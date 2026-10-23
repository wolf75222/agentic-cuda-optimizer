"""Hold exactly the seventeen unseen FINAL tasks; no generation lock or payload."""
from pathlib import Path
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import time

repo = Path(__file__).resolve().parents[2]
study = repo / "results/quality-study-v2"
ready = repo / "work/live/hold-finals-only-20261023-ready.json"
marker = study / "release-finals-only-20261023-hold"
assert not ready.exists() and not marker.exists(), "Fresh receipt and release marker required"


def authorize(action, table, column, database, origin):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ and table == "tasks" and column in {"id", "kind", "state"}:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


held = []
try:
    # Require an idle controller during the short handoff. This lock is released
    # after the ready receipt; later generation dispatch remains possible.
    with (study / "controller.lock").open("rb") as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with contextlib.closing(sqlite3.connect((study / "suite.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("PRAGMA query_only=ON")
            db.set_authorizer(authorize)
            selected = db.execute("SELECT id,kind,state FROM tasks WHERE kind=? AND state=? ORDER BY id",
                                  ("final", "pending")).fetchall()
        assert len(selected) == 17, "Expected exactly seventeen unseen FINAL tasks"
        for task_id, kind, state in selected:
            path = study / "task-locks" / (hashlib.sha256(task_id.encode()).hexdigest() + ".lock")
            assert not path.is_symlink(), "Aliased task lock refused"
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            stream = os.fdopen(descriptor, "rb")
            try:
                assert stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "Task lock must be regular"
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                stream.close()
                raise
            held.append((task_id, stream))
        record = {
            "kind": "cooperative_hold_seventeen_unseen_finals_only",
            "pid": os.getpid(), "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "held_tasks": [task_id for task_id, stream in held],
            "held_generation_tasks": [], "release_marker": str(marker),
            "source_sha": "577d20a8b8a7de333309a32dddc0c6ce79104df9",
            "read_columns_allowlist": ["id", "kind", "state"],
            "model_calls": 0, "scheduler_calls": 0, "campaign_or_checkpoint_mutations": 0,
            "reason": "Keep held-out evaluation inaccessible while allowing reviewed post-model inspection/recovery and independent pre-model generation work.",
            "scope": "Existing advisory task-lock inodes only; no generation locks, process signals, submission, payload, checkpoint, seed, RNG or array read."
        }
        with ready.open("x") as output:
            json.dump(record, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        print(json.dumps({"pid": os.getpid(), "held_finals": len(held), "ready": str(ready)}), flush=True)
    while not marker.exists():
        time.sleep(1)
finally:
    for task_id, stream in held:
        fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()
print("FINAL-only hold released by its explicit marker", flush=True)
