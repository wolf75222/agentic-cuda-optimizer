#!/usr/bin/env python3
"""Operational, exact-allowlist cleanup; opaque verified archives precede retirement."""
import datetime
import base64
import decimal
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import tempfile
import uuid

UID = 100267
CUTOFF_NS = 1784764800 * 10**9  # 2026-07-23 00:00 UTC; checked in main.
SCRATCH_NAMES = "249797 249798 250039 250047 250057 253182 253199 253202 253340 259030 259059 259103 259562 259621 259667 259669 259677 259705 259746 259783 259812 260080 260088 261628 261629 261668 263142 263147 263154 270608 272419".split()
HOME_NAMES = ["adc_hires_build", "adc_cpp.stale-20260529", "TP-20260331", "simplexe_gpu", "vec"]
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW


def digest_stream(f):
    h = hashlib.sha256()
    for block in iter(lambda: f.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def record(s):
    kind = "dir" if stat.S_ISDIR(s.st_mode) else "file" if stat.S_ISREG(s.st_mode) else "unsupported"
    return {"type": kind, "mode": stat.S_IMODE(s.st_mode), "uid": s.st_uid, "gid": s.st_gid,
            "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns, "size": s.st_size,
            "dev": s.st_dev, "ino": s.st_ino, "nlink": s.st_nlink, "allocated_bytes": s.st_blocks * 512}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def metadata_equal(s, expected):
    return record(s) == {k: v for k, v in expected.items() if k not in ("sha256", "xattrs_base64")}


def read_xattrs(fd, required=True):
    if not hasattr(os, "listxattr"):
        require(not required, "extended attribute inspection unavailable")
        return {}
    return {name: base64.b64encode(os.getxattr(fd, name)).decode("ascii") for name in sorted(os.listxattr(fd))}


def scan_root(parent_fd, name, uid, cutoff_ns, enforce_xattrs=True):
    rows = {}

    def visit(pfd, part, rel):
        s = os.stat(part, dir_fd=pfd, follow_symlinks=False)
        r = record(s)
        require(r["uid"] == uid, "foreign UID: " + rel)
        require(r["mtime_ns"] < cutoff_ns, "modified after cutoff: " + rel)
        require(r["type"] != "unsupported", "symlink/special file: " + rel)
        flags = DIR_FLAGS if r["type"] == "dir" else FILE_FLAGS
        fd = os.open(part, flags, dir_fd=pfd)
        try:
            require(metadata_equal(os.fstat(fd), r), "open/stat drift: " + rel)
            r["xattrs_base64"] = read_xattrs(fd, required=enforce_xattrs)
            if r["type"] == "dir":
                names = sorted(os.listdir(fd))
                for child in names:
                    require(child not in (".", "..") and "/" not in child, "invalid filename")
                    visit(fd, child, rel + "/" + child)
                require(sorted(os.listdir(fd)) == names, "directory entries drift: " + rel)
            else:
                require(s.st_nlink == 1, "hardlinked regular file: " + rel)
                with os.fdopen(os.dup(fd), "rb") as f:
                    r["sha256"] = digest_stream(f)
            require(metadata_equal(os.fstat(fd), r), "read/stat drift: " + rel)
            require(read_xattrs(fd, required=enforce_xattrs) == r["xattrs_base64"], "xattr read drift: " + rel)
            require(metadata_equal(os.stat(part, dir_fd=pfd, follow_symlinks=False), r), "path drift: " + rel)
        finally:
            os.close(fd)
        rows[rel] = r

    require(name and name == os.path.basename(name) and name not in (".", ".."), "invalid root")
    visit(parent_fd, name, name)
    return rows


def open_member(parent_fd, relative, rows):
    parts = relative.split("/")
    fd = os.dup(parent_fd)
    try:
        for i, part in enumerate(parts):
            rel = "/".join(parts[:i + 1])
            expected = rows[rel]
            nxt = os.open(part, DIR_FLAGS if expected["type"] == "dir" else FILE_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = nxt
            require(metadata_equal(os.fstat(fd), expected), "archive open drift: " + rel)
            require(read_xattrs(fd, required=False) == expected["xattrs_base64"], "archive xattr drift: " + rel)
        result = fd
        fd = None
        return result
    finally:
        if fd is not None:
            os.close(fd)


def write_json_exclusive(directory_fd, name, data):
    payload = (json.dumps(data, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
    try:
        with os.fdopen(os.dup(fd), "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    finally:
        os.close(fd)
    os.fsync(directory_fd)
    return hashlib.sha256(payload).hexdigest()


def make_archive(parent_fd, archive_fd, rows):
    with os.fdopen(os.dup(archive_fd), "wb") as raw:
        with tarfile.open(fileobj=raw, mode="w:gz", format=tarfile.PAX_FORMAT) as tf:
            for rel, expected in sorted(rows.items()):
                fd = open_member(parent_fd, rel, rows)
                try:
                    ti = tarfile.TarInfo(rel)
                    ti.mode, ti.uid, ti.gid = expected["mode"], expected["uid"], expected["gid"]
                    ti.type = tarfile.DIRTYPE if expected["type"] == "dir" else tarfile.REGTYPE
                    ti.size = 0 if expected["type"] == "dir" else expected["size"]
                    ti.mtime = expected["mtime_ns"] // 10**9
                    ti.pax_headers = {"mtime": str(decimal.Decimal(expected["mtime_ns"]) / decimal.Decimal(10**9))}
                    ti.pax_headers["agentic.cleanup.xattrs_base64"] = json.dumps(expected["xattrs_base64"], sort_keys=True, separators=(",", ":"))
                    if expected["type"] == "dir":
                        tf.addfile(ti)
                    else:
                        with os.fdopen(os.dup(fd), "rb") as f:
                            tf.addfile(ti, f)
                    require(metadata_equal(os.fstat(fd), expected), "archive read drift: " + rel)
                    require(read_xattrs(fd, required=False) == expected["xattrs_base64"], "archive xattr read drift: " + rel)
                finally:
                    os.close(fd)
        raw.flush()
        os.fsync(raw.fileno())


def verify_archive(archive_fd, rows):
    os.lseek(archive_fd, 0, os.SEEK_SET)
    seen = set()
    with os.fdopen(os.dup(archive_fd), "rb") as raw:
        with tarfile.open(fileobj=raw, mode="r:gz") as tf:
            for ti in tf:
                name = ti.name.rstrip("/")
                require(name in rows and name not in seen, "unknown/duplicate tar member")
                seen.add(name)
                r = rows[name]
                require((ti.isdir() if r["type"] == "dir" else ti.isreg()), "tar type mismatch")
                require((ti.mode, ti.uid, ti.gid) == (r["mode"], r["uid"], r["gid"]), "tar identity/mode mismatch")
                require(int(decimal.Decimal(ti.pax_headers["mtime"]) * 10**9) == r["mtime_ns"], "tar mtime mismatch")
                require(json.loads(ti.pax_headers["agentic.cleanup.xattrs_base64"]) == r["xattrs_base64"], "tar opaque xattr mismatch")
                require(ti.size == (0 if r["type"] == "dir" else r["size"]), "tar size mismatch")
                if r["type"] == "file":
                    with tf.extractfile(ti) as f:
                        require(digest_stream(f) == r["sha256"], "tar content mismatch")
    require(seen == set(rows), "missing tar members")
    os.lseek(archive_fd, 0, os.SEEK_SET)
    with os.fdopen(os.dup(archive_fd), "rb") as f:
        return digest_stream(f)


def retire_root(parent_fd, name, rows, deleted):
    def visit(pfd, part, rel):
        expected = rows[rel]
        require(metadata_equal(os.stat(part, dir_fd=pfd, follow_symlinks=False), expected), "retirement pre-stat drift: " + rel)
        fd = os.open(part, DIR_FLAGS if expected["type"] == "dir" else FILE_FLAGS, dir_fd=pfd)
        try:
            require(metadata_equal(os.fstat(fd), expected), "retirement open drift: " + rel)
            require(read_xattrs(fd, required=False) == expected["xattrs_base64"], "retirement xattr drift: " + rel)
            if expected["type"] == "file":
                require(metadata_equal(os.stat(part, dir_fd=pfd, follow_symlinks=False), expected), "retirement second-stat drift: " + rel)
                os.unlink(part, dir_fd=pfd)
                deleted.append(rel)
                return
            prefix = rel + "/"
            children = sorted(p[len(prefix):] for p in rows if p.startswith(prefix) and "/" not in p[len(prefix):])
            require(sorted(os.listdir(fd)) == children, "retirement directory drift: " + rel)
            for child in children:
                visit(fd, child, rel + "/" + child)
            require(os.listdir(fd) == [], "retirement directory nonempty: " + rel)
            final = os.fstat(fd)
            path_final = os.stat(part, dir_fd=pfd, follow_symlinks=False)
            require((final.st_dev, final.st_ino, final.st_uid, final.st_gid, final.st_mode) ==
                    (expected["dev"], expected["ino"], expected["uid"], expected["gid"], stat.S_IFDIR | expected["mode"]), "retirement directory identity drift")
            require((path_final.st_dev, path_final.st_ino, path_final.st_uid, path_final.st_gid, path_final.st_mode) ==
                    (final.st_dev, final.st_ino, final.st_uid, final.st_gid, final.st_mode), "retirement directory second-stat drift")
            require(read_xattrs(fd, required=False) == expected["xattrs_base64"], "retirement directory xattr drift")
            os.rmdir(part, dir_fd=pfd)
            deleted.append(rel)
        finally:
            os.close(fd)
    visit(parent_fd, name, name)


def quota():
    p = subprocess.run(["mmlsquota", "-u", "rmdraux", "--block-size", "K", "gpfs"], capture_output=True, text=True, timeout=30)
    return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def no_jobs():
    p = subprocess.run(["squeue", "-u", "rmdraux", "-h", "-o", "%i|%T|%Z|%o"], capture_output=True, text=True, timeout=20)
    require(p.returncode == 0 and p.stdout.strip() == "", "active jobs or scheduler unavailable")
    return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def execute_batch(parent, names, label, storage, uid=UID, cutoff_ns=CUTOFF_NS, live=True):
    require(os.path.realpath(parent) == parent, "parent must be real canonical path")
    require(os.path.realpath(storage) == storage, "archive parent must be real canonical path")
    ps, ss = os.lstat(parent), os.lstat(storage)
    require(stat.S_ISDIR(ps.st_mode) and ps.st_uid == uid, "source parent owner/type")
    require(stat.S_ISDIR(ss.st_mode) and ss.st_uid == uid and stat.S_IMODE(ss.st_mode) == 0o700, "archive parent must be private owned0700")
    parent_fd, storage_fd = os.open(parent, DIR_FLAGS), os.open(storage, DIR_FLAGS)
    report = {"label": label, "parent": parent, "requested_roots": names, "accepted_roots": [], "refused_roots": {}, "retired_roots": [], "deleted_entries": 0, "deleted_files": 0, "deleted_dirs": 0, "errors": [], "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    archive_fd = None
    rows = {}
    try:
        if live:
            report["squeue_before"] = no_jobs()
            report["quota_before"] = quota()
        for name in names:
            try:
                own = scan_root(parent_fd, name, uid, cutoff_ns, enforce_xattrs=live)
                rows.update(own)
                report["accepted_roots"].append(name)
            except Exception as e:
                report["refused_roots"][name] = str(e)
        require(rows, "no eligible roots: " + json.dumps(report["refused_roots"], sort_keys=True))
        token = label + "-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
        report["manifest_path"] = storage + "/" + token + "-manifest.json"
        report["manifest_sha256"] = write_json_exclusive(storage_fd, token + "-manifest.json", {"rows": rows, "cutoff_ns": cutoff_ns, "parent": parent, "label": label, "accepted_roots": report["accepted_roots"]})
        report["archive_path"] = storage + "/" + token + ".tar.gz"
        archive_fd = os.open(token + ".tar.gz", os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=storage_fd)
        make_archive(parent_fd, archive_fd, rows)
        os.fsync(storage_fd)
        report["archive_sha256"] = verify_archive(archive_fd, rows)
        report["archive_bytes"] = os.fstat(archive_fd).st_size
        fresh = {}
        for name in report["accepted_roots"]:
            fresh.update(scan_root(parent_fd, name, uid, cutoff_ns, enforce_xattrs=live))
        require(fresh == rows, "source changed after archive verification")
        report["archive_and_source_verified"] = True
        report["opaque_xattrs_policy"] = "Complete base64 values in manifest rows and agentic.cleanup.xattrs_base64 PAX header; restore with os.setxattr from these values. Ordinary tar extraction alone does not restore this custom metadata."
        report["entries_with_xattrs"] = sum(bool(r["xattrs_base64"]) for r in rows.values())
        report["source_entries"] = len(rows)
        report["source_files"] = sum(r["type"] == "file" for r in rows.values())
        report["source_dirs"] = sum(r["type"] == "dir" for r in rows.values())
        report["source_logical_bytes"] = sum(r["size"] for r in rows.values() if r["type"] == "file")
        report["source_allocated_bytes"] = sum(r["allocated_bytes"] for r in rows.values())
        if live:
            report["squeue_before_retirement"] = no_jobs()
        report["verified_phase_path"] = storage + "/" + token + "-verified-before-retirement.json"
        write_json_exclusive(storage_fd, token + "-verified-before-retirement.json", report)
        deleted = []
        for name in report["accepted_roots"]:
            try:
                retire_root(parent_fd, name, rows, deleted)
                report["retired_roots"].append(name)
            except Exception as e:
                report["errors"].append({"root": name, "error": str(e)})
        os.fsync(parent_fd)
        report["deleted_entries"] = len(deleted)
        report["deleted_files"] = sum(rows[n]["type"] == "file" for n in deleted)
        report["deleted_dirs"] = sum(rows[n]["type"] == "dir" for n in deleted)
        report["deleted_names"] = deleted
        report["retired_roots_absent"] = {n: not os.path.lexists(parent + "/" + n) for n in report["retired_roots"]}
        require(verify_archive(archive_fd, rows) == report["archive_sha256"], "archive changed after retirement")
        report["archive_verified_after_retirement"] = True
        if live:
            report["quota_after"] = quota()
            report["squeue_after"] = no_jobs()
        report["receipt_path"] = storage + "/" + token + "-receipt.json"
        report["receipt_sha256"] = write_json_exclusive(storage_fd, token + "-receipt.json", report)
        return report
    finally:
        if archive_fd is not None:
            os.close(archive_fd)
        os.close(storage_fd)
        os.close(parent_fd)


def self_test():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = str(Path(tmp).resolve())
        parent, storage = str(Path(tmp) / "source"), str(Path(tmp) / "archive")
        os.mkdir(parent, 0o700)
        os.mkdir(storage, 0o700)
        for name in ["good", "symlink", "hardlink", "new"]:
            os.mkdir(parent + "/" + name)
            Path(parent + "/" + name + "/f").write_bytes(b"opaque\x00fixture")
        os.symlink("f", parent + "/symlink/link")
        os.link(parent + "/hardlink/f", parent + "/hardlink/second")
        for p in Path(parent).rglob("*"):
            os.utime(p, ns=(1_000_000_000, 1_000_000_000), follow_symlinks=False)
        os.utime(parent + "/new/f", ns=(CUTOFF_NS + 1, CUTOFF_NS + 1))
        got = execute_batch(parent, ["good", "symlink", "hardlink", "new"], "selftest", storage, uid=os.getuid(), live=False)
        require(got["retired_roots"] == ["good"] and got["deleted_entries"] == 2, "fixture retirement count")
        require(set(got["refused_roots"]) == {"symlink", "hardlink", "new"}, "fixture refusals")
        require(all(os.path.lexists(parent + "/" + n) for n in ["symlink", "hardlink", "new"]), "fixture refused preservation")
        return {"self_test": "PASS", "checks": ["opaque archive equality", "full nanosecond mtime metadata", "source unchanged", "file+directory retirement", "symlink refused", "hardlink refused", "new descendant refused", "refused roots preserved", "archive reverified after retirement"]}


def main():
    global CUTOFF_NS
    CUTOFF_NS = int(datetime.datetime(2026, 7, 23, tzinfo=datetime.timezone.utc).timestamp()) * 10**9
    if "--self-test" in sys.argv:
        print(json.dumps(self_test(), sort_keys=True))
        return
    require(os.getuid() == UID, "must run as rmdraux")
    no_jobs()
    storage = "/gpfs/home/rmdraux/.cleanup-preserved"
    reports = []
    batches = [("/gpfs/scratch/rmdraux", SCRATCH_NAMES, "old-numeric-scratch"), ("/gpfs/home/rmdraux", HOME_NAMES, "old-authorized-home")]
    if "--home-only" in sys.argv:
        batches = batches[1:]
    for parent, names, label in batches:
        reports.append(execute_batch(parent, names, label, storage))
    compact = []
    for r in reports:
        compact.append({k: v for k, v in r.items() if k != "deleted_names"})
    print(json.dumps({"batches": compact}, sort_keys=True))


if __name__ == "__main__":
    main()
