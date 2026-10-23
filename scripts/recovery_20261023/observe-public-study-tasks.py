"""Read only allowlisted public task columns; never deserialize task payloads."""
import datetime
import contextlib
import hashlib
import json
import pathlib
import sqlite3
import sys

root = pathlib.Path(__file__).resolve().parents[2]
journal = root / "results/quality-study-v2/suite.sqlite"
output = pathlib.Path(sys.argv[1])
columns = {"", "id", "kind", "state", "attempt", "session", "job_id"}

def authorize(action, table, column, database, origin):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ and table == "tasks" and column in columns:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and column == "count":
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY

with contextlib.closing(sqlite3.connect(journal.as_uri() + "?mode=ro", uri=True)) as db:
    db.execute("PRAGMA query_only=ON")
    db.execute("BEGIN")
    db.set_authorizer(authorize)
    counts = [dict(zip(("kind", "state", "count"), row)) for row in db.execute(
        "SELECT kind,state,count(*) FROM tasks GROUP BY kind,state")]
    running = [dict(zip(("id", "kind", "state", "attempt", "session", "job_id"), row))
               for row in db.execute(
        "SELECT id,kind,state,attempt,session,job_id FROM tasks WHERE state=?", ("running",))]
    # Closing leaves the read transaction without an authorized COMMIT.

record = {
    "kind": "query_only_public_task_state_observation",
    "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "journal_path": str(journal),
    "read_columns_allowlist": sorted(columns),
    "sqlite_authorizer": "SELECT, tasks allowlisted columns and count only; all others denied",
    "counts": counts,
    "running": running,
    "observer_source_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    "scope": "Public task journal states only; no payload/result/seed/private input/RNG/checkpoint. Completed tasks do not prove successful methods or FINAL qualification. Mutable journal is not hashed as a cold artifact.",
}
raw = (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode()
with output.open("xb") as stream:
    stream.write(raw)
print(json.dumps({"receipt": str(output), "sha256": hashlib.sha256(raw).hexdigest(), "observation": record}, ensure_ascii=False))
