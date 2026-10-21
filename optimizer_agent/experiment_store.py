"""SQLite registry: immutable evidence, transactional budgets and tree journal.

Tree snapshots contain Python pickles and must only be opened from a trusted
local campaign directory. SQLite WAL + BEGIN IMMEDIATE serialize all writers;
the tree snapshot and the processed-event flag commit in the same transaction.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import uuid


class BudgetExceeded(RuntimeError):
    pass


class IncompatibleCampaign(ValueError):
    pass


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ExperimentStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = self.path.parent / "artifacts"
        self._lock = threading.RLock()
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS campaigns (
                    run_id TEXT PRIMARY KEY, identity TEXT NOT NULL, snapshot BLOB,
                    tree_version INTEGER NOT NULL DEFAULT 0,
                    incumbent TEXT, incumbent_version INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS trials (
                    run_id TEXT NOT NULL, trial_id TEXT NOT NULL, status TEXT NOT NULL,
                    payload TEXT NOT NULL, result TEXT, evaluation_id TEXT,
                    PRIMARY KEY(run_id,trial_id));
                CREATE TABLE IF NOT EXISTS evaluations (
                    run_id TEXT NOT NULL, evaluation_id TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(run_id,evaluation_id));
                CREATE TABLE IF NOT EXISTS reward_claims (
                    run_id TEXT NOT NULL, evaluation_id TEXT NOT NULL, trial_id TEXT NOT NULL,
                    PRIMARY KEY(run_id,evaluation_id));
                CREATE TABLE IF NOT EXISTS integration_events (
                    run_id TEXT NOT NULL, trial_id TEXT NOT NULL,
                    processed INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(run_id,trial_id));
                CREATE TABLE IF NOT EXISTS candidates (
                    run_id TEXT NOT NULL, candidate_id TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(run_id,candidate_id));
                CREATE TABLE IF NOT EXISTS candidate_status (
                    run_id TEXT NOT NULL, candidate_id TEXT NOT NULL, valid INTEGER,
                    reason TEXT NOT NULL, PRIMARY KEY(run_id,candidate_id));
                CREATE TABLE IF NOT EXISTS reservations (
                    run_id TEXT NOT NULL, reservation_id TEXT NOT NULL,
                    purpose TEXT NOT NULL, amount TEXT NOT NULL, actual TEXT,
                    created_at REAL NOT NULL, expires_at REAL,
                    PRIMARY KEY(run_id,reservation_id));
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    kind TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, trial_id TEXT NOT NULL,
                    status TEXT NOT NULL, payload TEXT NOT NULL);
            """)

    @contextlib.contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        finally:
            db.close()

    @contextlib.contextmanager
    def transaction(self):
        with self._lock, self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def initialize_campaign(self, run_id, identity, budget_limits=None, finalization_reserve=None):
        identity = dict(identity, budget_limits=budget_limits or {},
                        finalization_reserve=finalization_reserve or {})
        serialized = canonical_json(identity)
        with self.transaction() as db:
            previous = db.execute("SELECT identity FROM campaigns WHERE run_id=?", (run_id,)).fetchone()
            if previous and previous["identity"] != serialized:
                raise IncompatibleCampaign("Contract, environment, search protocol or budgets changed; create a new run_id")
            db.execute("INSERT OR IGNORE INTO campaigns(run_id,identity) VALUES (?,?)", (run_id,serialized))
            if finalization_reserve and not db.execute("SELECT 1 FROM reservations WHERE run_id=? AND reservation_id='__finalization__'",(run_id,)).fetchone():
                self._reserve(db, run_id, "__finalization__", finalization_reserve, "finalization_pool", None)

    def _event(self, db, run_id, kind, payload):
        db.execute("INSERT INTO events(run_id,kind,payload,created_at) VALUES (?,?,?,?)",
                   (run_id, kind, canonical_json(payload), time.time()))

    def event(self, run_id, kind, payload):
        with self.transaction() as db:
            self._event(db, run_id, kind, payload)

    def events(self, run_id):
        with self.connection() as db:
            return [dict(row, payload=json.loads(row["payload"])) for row in
                    db.execute("SELECT * FROM events WHERE run_id=? ORDER BY event_id", (run_id,))]

    @staticmethod
    def _amount(amount):
        converted = {str(k): float(v) for k, v in amount.items()}
        if any(not math.isfinite(v) or v < 0 for v in converted.values()):
            raise ValueError("Budget amounts must be finite and nonnegative")
        return converted

    def _budget(self, db, run_id):
        row = db.execute("SELECT identity FROM campaigns WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        limits = json.loads(row["identity"])["budget_limits"]
        spent, reserved = {}, {}
        for reservation in db.execute("SELECT amount,actual FROM reservations WHERE run_id=?", (run_id,)):
            target = spent if reservation["actual"] is not None else reserved
            for key, value in json.loads(reservation["actual"] or reservation["amount"]).items():
                target[key] = target.get(key, 0) + value
        return {"limits": limits, "spent": spent, "reserved": reserved}

    def budget_state(self, run_id):
        with self.connection() as db:
            return self._budget(db, run_id)

    def _reserve(self, db, run_id, reservation_id, amount, purpose, expires_at):
        amount = self._amount(amount)
        existing = db.execute("SELECT amount,purpose FROM reservations WHERE run_id=? AND reservation_id=?",
                              (run_id,reservation_id)).fetchone()
        if existing:
            if json.loads(existing["amount"]) != amount or existing["purpose"] != purpose:
                raise ValueError("Reservation identity reused with different amounts or purpose")
            return reservation_id
        state = self._budget(db, run_id)
        for key, value in amount.items():
            if key in state["limits"] and value + state["spent"].get(key, 0) + state["reserved"].get(key, 0) > state["limits"][key] + 1e-12:
                raise BudgetExceeded(f"{key} budget exhausted")
        db.execute("INSERT INTO reservations VALUES (?,?,?,?,NULL,?,?)",
                   (run_id,reservation_id,purpose,canonical_json(amount),time.time(),expires_at))
        return reservation_id

    def reserve(self, run_id, reservation_id, amount, purpose="work", expires_at=None):
        with self.transaction() as db:
            return self._reserve(db, run_id, reservation_id, amount, purpose, expires_at)

    def _reconcile(self, db, run_id, reservation_id, actual):
        actual = self._amount(actual)
        row = db.execute("SELECT actual FROM reservations WHERE run_id=? AND reservation_id=?",
                         (run_id,reservation_id)).fetchone()
        if not row:
            raise KeyError(reservation_id)
        if row["actual"] is not None:
            if json.loads(row["actual"]) != actual:
                raise ValueError("A reconciled cost cannot be silently rewritten")
            return
        # Actual overruns are recorded, even if they exceed a limit. Subsequent
        # reservations fail rather than hiding consumed physical resources.
        db.execute("UPDATE reservations SET actual=? WHERE run_id=? AND reservation_id=?",
                   (canonical_json(actual),run_id,reservation_id))

    def reconcile(self, run_id, reservation_id, actual):
        with self.transaction() as db:
            self._reconcile(db, run_id, reservation_id, actual)

    def release_finalization(self, run_id):
        """Release protected funds only when entering the finalization stage."""
        with self.transaction() as db:
            row = db.execute("SELECT actual FROM reservations WHERE run_id=? AND reservation_id='__finalization__'", (run_id,)).fetchone()
            if row and row["actual"] is None:
                self._reconcile(db, run_id, "__finalization__", {})
                self._event(db, run_id, "finalization_reserve_released", {})

    def reserve_finalization(self,run_id,reservation_id,amount):
        """Atomically transfer protected funds to a finalization task."""
        amount = self._amount(amount)
        with self.transaction() as db:
            existing = db.execute("SELECT amount,purpose FROM reservations WHERE run_id=? AND reservation_id=?",(run_id,reservation_id)).fetchone()
            if existing:
                if json.loads(existing["amount"]) != amount or existing["purpose"] != "finalization":
                    raise ValueError("Finalization reservation identity reused")
                return reservation_id
            pool = db.execute("SELECT amount,actual FROM reservations WHERE run_id=? AND reservation_id='__finalization__'",(run_id,)).fetchone()
            if not pool or pool["actual"] is not None:
                raise BudgetExceeded("No protected finalization budget remains")
            remaining = json.loads(pool["amount"])
            if any(value > remaining.get(key,0)+1e-12 for key,value in amount.items()):
                raise BudgetExceeded("Finalization task exceeds protected funds")
            for key,value in amount.items():
                remaining[key] -= value
            db.execute("UPDATE reservations SET amount=? WHERE run_id=? AND reservation_id='__finalization__'",(canonical_json(remaining),run_id))
            self._reserve(db,run_id,reservation_id,amount,"finalization",None)
            return reservation_id

    def reserve_finalization_remaining(self,run_id,reservation_id):
        """Atomically claim protected funds plus all presently unused funds.

        Returning the actual reserved amounts avoids a separate read/release/
        reserve window. A repeated call with the same task ID returns the
        original reservation; it never allocates or consumes the budget twice.
        Only the protected pool's resource dimensions are transferred.
        """
        with self.transaction() as db:
            existing = db.execute("SELECT purpose,amount FROM reservations WHERE run_id=? AND reservation_id=?",(run_id,reservation_id)).fetchone()
            if existing:
                if existing["purpose"] != "finalization":
                    raise ValueError("Finalization identity reused by another purpose")
                return json.loads(existing["amount"])
            pool = db.execute("SELECT amount,actual FROM reservations WHERE run_id=? AND reservation_id='__finalization__'",(run_id,)).fetchone()
            if not pool or pool["actual"] is not None:
                raise BudgetExceeded("No protected finalization pool remains")
            protected = json.loads(pool["amount"])
            state = self._budget(db,run_id)
            amount = {}
            for resource,value in protected.items():
                if resource in state["limits"]:
                    amount[resource] = max(0.,state["limits"][resource]-state["spent"].get(resource,0)-state["reserved"].get(resource,0)+value)
                else:
                    amount[resource] = value
            if not amount or not any(value > 0 for value in amount.values()):
                raise BudgetExceeded("No finalization capacity remains")
            self._reconcile(db,run_id,"__finalization__",{})
            self._reserve(db,run_id,reservation_id,amount,"finalization",None)
            self._event(db,run_id,"finalization_budget_transferred",{"reservation_id":reservation_id,"amount":amount})
            return amount

    def expire_reservation(self, run_id, reservation_id, actual, *, worker_confirmed_stopped=False):
        if not worker_confirmed_stopped:
            raise ValueError("A timeout cannot release a GPU reservation until its worker is stopped")
        with self.transaction() as db:
            row = db.execute("SELECT expires_at FROM reservations WHERE run_id=? AND reservation_id=?",
                             (run_id,reservation_id)).fetchone()
            if not row or row["expires_at"] is None or row["expires_at"] > time.time():
                raise ValueError("Reservation has not expired")
            self._reconcile(db, run_id, reservation_id, actual)
            self._event(db, run_id, "reservation_expired", {"reservation_id":reservation_id})

    def archive_candidate(self, run_id, candidate):
        candidate = dict(candidate)
        candidate_id = str(candidate["candidate_id"])
        source = candidate.get("source", "")
        candidate["source_hash"] = hashlib.sha256(source.encode()).hexdigest()
        serialized = canonical_json(candidate)
        with self.transaction() as db:
            row = db.execute("SELECT payload FROM candidates WHERE run_id=? AND candidate_id=?", (run_id,candidate_id)).fetchone()
            if row and row["payload"] != serialized:
                raise ValueError("Candidate artifacts are immutable; allocate a new candidate_id")
            db.execute("INSERT OR IGNORE INTO candidates VALUES (?,?,?)", (run_id,candidate_id,serialized))
        # Content-addressed source paths cannot escape artifact_dir via IDs.
        self.atomic_artifact(f"sources/{candidate['source_hash']}.cu", source.encode())
        return candidate_id

    put_candidate = archive_candidate

    def get_candidate_source(self, candidate_id, run_id=None):
        with self.connection() as db:
            if run_id is None:
                rows = db.execute("SELECT run_id,payload FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchall()
            else:
                rows = db.execute("SELECT run_id,payload FROM candidates WHERE candidate_id=? AND run_id=?", (candidate_id,run_id)).fetchall()
            if len(rows) != 1:
                raise KeyError(f"Candidate missing or ambiguous: {candidate_id}")
            candidate = json.loads(rows[0]["payload"])
            candidate_run = rows[0]["run_id"]
            candidate["evaluations"] = []
            for record in db.execute("SELECT payload FROM evaluations WHERE run_id=?",(candidate_run,)):
                evaluation = json.loads(record["payload"])
                evidence_candidate = evaluation.get("candidate_id",evaluation.get("state",{}).get("candidate_id"))
                if evidence_candidate == candidate_id:
                    # The executable is already returned above; do not grow
                    # recursively nested copies of source/measurement history.
                    candidate["evaluations"].append({key:value for key,value in evaluation.items() if key != "state"})
            validity = db.execute("SELECT valid,reason FROM candidate_status WHERE run_id=? AND candidate_id=?",(candidate_run,candidate_id)).fetchone()
            if validity:
                candidate["current_validity"] = {"valid":bool(validity["valid"]),"reason":validity["reason"]}
            return candidate

    def set_candidate_validity(self, run_id, candidate_id, valid, reason):
        with self.transaction() as db:
            db.execute("INSERT OR REPLACE INTO candidate_status VALUES (?,?,?,?)", (run_id,candidate_id,int(valid),reason))
            self._event(db,run_id,"candidate_validity",{"candidate_id":candidate_id,"valid":valid,"reason":reason})

    def _add_evaluation(self, db, run_id, evaluation):
        evaluation_id = str(evaluation["evaluation_id"])
        payload = canonical_json(evaluation)
        old = db.execute("SELECT payload FROM evaluations WHERE run_id=? AND evaluation_id=?",(run_id,evaluation_id)).fetchone()
        if old and old["payload"] != payload:
            raise ValueError("Measurement identity reused for different evidence")
        db.execute("INSERT OR IGNORE INTO evaluations VALUES (?,?,?)",(run_id,evaluation_id,payload))
        return evaluation_id

    def add_evaluation(self, run_id, evaluation):
        with self.transaction() as db:
            return self._add_evaluation(db,run_id,evaluation)

    def get_evaluation(self, run_id, evaluation_id):
        with self.connection() as db:
            row = db.execute("SELECT payload FROM evaluations WHERE run_id=? AND evaluation_id=?",(run_id,evaluation_id)).fetchone()
            if not row:
                raise KeyError(evaluation_id)
            return json.loads(row["payload"])

    def new_attempt(self, run_id, trial_id, payload=None):
        attempt_id = uuid.uuid4().hex
        with self.transaction() as db:
            db.execute("INSERT INTO attempts VALUES (?,?,?,?,?)",(attempt_id,run_id,trial_id,"running",canonical_json(payload or {})))
            self._event(db,run_id,"attempt_started",{"trial_id":trial_id,"attempt_id":attempt_id})
        return attempt_id

    def finish_attempt(self, attempt_id, status, payload):
        with self.transaction() as db:
            db.execute("UPDATE attempts SET status=?,payload=? WHERE attempt_id=?",(status,canonical_json(payload),attempt_id))

    def incumbent(self, run_id):
        with self.connection() as db:
            row = db.execute("SELECT incumbent,incumbent_version FROM campaigns WHERE run_id=?",(run_id,)).fetchone()
            return {"candidate_id": row["incumbent"], "version":row["incumbent_version"]}

    def promote(self, run_id, candidate_id, expected_version, decision):
        if decision.get("status") != "improved" or not decision.get("evidence"):
            raise ValueError("Promotion requires an improved confirmation with evidence")
        with self.transaction() as db:
            row = db.execute("SELECT incumbent_version FROM campaigns WHERE run_id=?",(run_id,)).fetchone()
            if row["incumbent_version"] != expected_version:
                self._event(db,run_id,"promotion_stale",{"candidate_id":candidate_id,"expected_version":expected_version})
                return False
            if not db.execute("SELECT 1 FROM candidates WHERE run_id=? AND candidate_id=?",(run_id,candidate_id)).fetchone():
                raise KeyError(candidate_id)
            invalid = db.execute("SELECT valid FROM candidate_status WHERE run_id=? AND candidate_id=?",(run_id,candidate_id)).fetchone()
            if invalid is not None and invalid["valid"] == 0:
                raise ValueError("An invalidated candidate cannot be promoted")
            db.execute("UPDATE campaigns SET incumbent=?,incumbent_version=incumbent_version+1 WHERE run_id=?",(candidate_id,run_id))
            self._event(db,run_id,"promotion",{"candidate_id":candidate_id,"version":expected_version+1,"decision":decision})
            return True

    def atomic_artifact(self, relative_path, data):
        target = (self.artifact_dir / relative_path).resolve()
        if not target.is_relative_to(self.artifact_dir.resolve()):
            raise ValueError("Artifact path escapes campaign directory")
        target.parent.mkdir(parents=True,exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".artifact-",dir=target.parent)
        try:
            with os.fdopen(descriptor,"wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary,target)
            directory = os.open(target.parent,os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return target
