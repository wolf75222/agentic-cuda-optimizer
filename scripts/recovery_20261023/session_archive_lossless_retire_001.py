"""Retire ONLY redundant originals in the private four-session rollback parent.

Requires root-approved publication proof and independent historical read-only
attach receipt. Verify the project copy first, then stream/hash and identity-check
each original before unlink through directory FDs. No path outside the exact
manifest is removed. Keep the empty rollback parent/.lock as evidence. No FINAL,
model, Slurm, GPU, SQL or source-code mutation API is used. Known-writer/private
namespace assumptions from publication remain; malicious same-UID protection is
not claimed. No payload is emitted or decoded.
"""
from __future__ import annotations
import argparse
import collections
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

spec = importlib.util.spec_from_file_location('published_helper', Path(__file__).with_name('session_archive_lossless_publish_003.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
m = p.m


class Progress:
    def __init__(self, root_fd):
        self.fd = os.open('retirement-progress.jsonl', os.O_RDWR | os.O_APPEND | os.O_CREAT | m.O_NOFOLLOW | m.O_CLOEXEC, 0o600, dir_fd=root_fd)
        s = os.fstat(self.fd)
        if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_nlink != 1:
            raise m.Refusal('Unsafe retirement progress')
        with os.fdopen(self.fd, 'rb', closefd=False) as stream:
            raw = stream.read()
        if raw and not raw.endswith(b'\n'):
            raise m.Refusal('Torn retirement progress preserved for review')
        self.events = [json.loads(line) for line in raw.splitlines()]
        if any(row.get('sequence') != index for index, row in enumerate(self.events)):
            raise m.Refusal('Retirement sequence mismatch')
        os.fsync(root_fd)

    def append(self, action, **fields):
        row = {'sequence': len(self.events), 'action': action, **fields}
        m.write_all(self.fd, m.raw_json(row))
        os.fsync(self.fd)
        self.events.append(row)

    def close(self):
        os.close(self.fd)


def validate_links(sfd, names, ops, uid):
    for name in names:
        s = p.entry(sfd, name)
        if s is None or not stat.S_ISLNK(s.st_mode) or s.st_uid != uid or m.inode_identity(s) != ops[name]['link_inode'] or os.readlink(name, dir_fd=sfd) != ops[name]['target']:
            raise m.Refusal('Published old namespace identity changed')


def absent(root_fd, key):
    try:
        with m.parent_and_name(root_fd, key) as (fd, name):
            return p.entry(fd, name) is None
    except FileNotFoundError:
        return True


def historical_receipt(raw, expected_sha, names, jobs=None, manifest=None, publication_sha=None):
    if m.digest(raw) != expected_sha:
        raise m.Refusal('Historical access receipt digest mismatch')
    value = json.loads(raw)
    if publication_sha is not None and (value.get('post_publication') is not True or value.get('publication_proof_sha256') != publication_sha):
        raise m.Refusal('Historical attach must explicitly bind this completed publication, not pre-transfer access')
    if value.get('kind') != 'read_only_four_closed_generation_historical_access_verification' or value.get('model_calls') != 0 or value.get('gpu_calls') != 0 or value.get('slurm_submissions') != 0 or value.get('original_scientific_artifacts_written') != 0:
        raise m.Refusal('Independent read-only historical access receipt required')
    rows = value.get('checks')
    if not isinstance(rows, list) or len(rows) != len(names) or {row.get('session_id') for row in rows} != set(names):
        raise m.Refusal('Historical access session scope mismatch')
    for row in rows:
        name, terminal = row['session_id'], row.get('terminal', {})
        if terminal.get('state') != 'COMPLETED' or terminal.get('exit_code') != '0:0' or terminal.get('terminal') is not True or row.get('job_id') != terminal.get('job_id') or (jobs is not None and row.get('job_id') != jobs[name]):
            raise m.Refusal('Historical terminal job identity mismatch')
        roles = row.get('remote_roles_equal_pinned_local_bytes', {})
        if set(roles) != {'job_id.json', 'snapshot.json', 'ready.json', 'build.json'}:
            raise m.Refusal('Historical access role proof incomplete')
        if manifest is not None and any(roles[role].get('sha256') != manifest['rows'][name + '/' + role]['sha256'] for role in roles):
            raise m.Refusal('Historical access bytes differ from retained source manifest')
    return value


def retire(source, destination, names, uid, source_gid, storage_gid, *,
           publication_proof_sha256, historical_access_receipt_sha256,
           acl_reader=m.gpfs_acl, fail=None):
    """No sole copy is removed. Only the exact closed rollback tree is retired.

    The caller has already authenticated the independent receipt and its hash.
    Live CLI performs that validation; synthetic tests use a synthetic authority.
    """
    names = tuple(names)
    sfd, dfd = m.open_absolute_directory(source), m.open_absolute_directory(destination)
    rfd = dlock = rlock = None
    progress = None
    try:
        ds = os.fstat(dfd)
        if uid != os.getuid() or ds.st_uid != uid or ds.st_gid != storage_gid or stat.S_IMODE(ds.st_mode) != 0o2700:
            raise m.Refusal('Private approved-account destination required')
        dlock = os.open('.lock', os.O_RDWR | m.O_NOFOLLOW | m.O_CLOEXEC, dir_fd=dfd)
        fcntl.flock(dlock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pub = m.read_json(dfd, 'publication-proof.json')
        manifest, plan = m.read_json(dfd, 'source-manifest.json'), m.read_json(dfd, 'publication-operations.json')
        pub_intent = m.read_json(dfd, 'publication-intent.json')
        if m.digest(m.raw_json(pub)) != publication_proof_sha256 or pub.get('status') != 'published_all_four_originals_retained_private_parent' or pub.get('source_manifest_sha256') != m.digest(m.raw_json(manifest)) or pub.get('operations_sha256') != m.digest(m.raw_json(plan)) or pub_intent.get('script_sha256') != m.digest(Path(p.__file__).read_bytes()):
            raise m.Refusal('Approved publication/manifest/helper authority mismatch')
        ops = plan['operations']
        if set(ops) != set(names) or pub.get('rollback_parent') != plan.get('rollback_parent'):
            raise m.Refusal('Exact published scope mismatch')
        m.valid_component(plan['rollback_parent'])
        if not plan['rollback_parent'].startswith('.aco-closed-rollback-'):
            raise m.Refusal('Rollback basename mismatch')
        rfd = os.open(plan['rollback_parent'], os.O_RDONLY | m.O_DIRECTORY | m.O_NOFOLLOW | m.O_CLOEXEC, dir_fd=sfd)
        rlock = os.open('.lock', os.O_RDWR | m.O_NOFOLLOW | m.O_CLOEXEC, dir_fd=rfd)
        fcntl.flock(rlock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for fd, expected_gid in ((dlock, storage_gid), (rlock, source_gid)):
            lock_stat = os.fstat(fd)
            if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_uid != uid or lock_stat.st_gid != expected_gid or lock_stat.st_nlink != 1 or stat.S_IMODE(lock_stat.st_mode) != 0o600:
                raise m.Refusal('Unsafe retirement advisory lock')
        p.validate_parent(sfd, rfd, plan, uid, source_gid, acl_reader)
        validate_links(sfd, names, ops, uid)
        p.verify_target(dfd, names, manifest, uid, storage_gid, acl_reader)
        intent = {'schema': 1, 'kind': 'retire_only_verified_redundant_four_generation_originals',
                  'source': str(source), 'destination': str(destination), 'names': list(names),
                  'publication_proof_sha256': publication_proof_sha256,
                  'historical_access_receipt_sha256': historical_access_receipt_sha256,
                  'manifest_sha256': m.digest(m.raw_json(manifest)),
                  'script_sha256': m.digest(Path(__file__).read_bytes()),
                  'rollback_parent': plan['rollback_parent'], 'rollback_parent_inode': plan['rollback_parent_inode']}
        if 'retirement-intent.json' in os.listdir(dfd):
            if m.read_json(dfd, 'retirement-intent.json') != intent:
                raise m.Refusal('Retirement intent collision')
        else:
            # Last full original global hardlink/opaque-byte proof before any
            # source payload unlink; all four originals must still be present.
            p.verify_originals(sfd, rfd, names, ops, manifest, uid, source_gid, acl_reader)
            m.atomic_json(dfd, 'retirement-intent.json', intent)
        progress = Progress(dfd)
        done = {}
        for event in progress.events:
            if event.get('action') != 'retired_entry':
                raise m.Refusal('Unrecognized retirement progress action')
            if event['action'] == 'retired_entry':
                key = event.get('path')
                if key not in manifest['rows'] or key in done or event.get('kind') != manifest['rows'][key]['kind'] or event.get('inode') != manifest['source_identities'][key][:2]:
                    raise m.Refusal('Retirement progress scope/identity mismatch')
                done[key] = event
        active = m.read_json(dfd, 'retirement-active.json') if 'retirement-active.json' in os.listdir(dfd) else None
        canonical = {key: group[0] for group in manifest['hardlink_classes'] for key in group}
        remaining = collections.Counter(canonical[key] for key, row in manifest['rows'].items() if row['kind'] == 'regular_file' and key not in done)
        files = [key for key, row in manifest['rows'].items() if row['kind'] == 'regular_file']
        dirs = sorted((key for key, row in manifest['rows'].items() if row['kind'] == 'directory'), key=lambda key: (key.count('/'), key), reverse=True)
        for key in files + dirs:
            row = manifest['rows'][key]
            expected_inode = manifest['source_identities'][key][:2]
            if key in done:
                if not absent(rfd, key):
                    raise m.Refusal('Previously retired entry reappeared')
                if active is not None and active.get('path') == key:
                    if active.get('inode') != expected_inode or active.get('kind') != row['kind']:
                        raise m.Refusal('Committed retirement active identity mismatch')
                    m.atomic_json(dfd, 'retirement-active.json', None, replace=True)
                    active = None
                continue
            if active is not None and active.get('path') != key:
                raise m.Refusal('Retirement active operation order mismatch')
            with m.parent_and_name(rfd, key) as (parent, name):
                if '/' in key:
                    parent_key = key.rsplit('/', 1)[0]
                    parent_stat = os.fstat(parent)
                    if m.inode_identity(parent_stat) != manifest['source_identities'][parent_key][:2] or parent_stat.st_uid != uid or parent_stat.st_gid != source_gid or stat.S_IMODE(parent_stat.st_mode) != manifest['rows'][parent_key]['mode']:
                        raise m.Refusal('Original parent directory identity changed')
                else:
                    p.validate_parent(sfd, rfd, plan, uid, source_gid, acl_reader)
                s = p.entry(parent, name)
                operation = {'path': key, 'kind': row['kind'], 'inode': expected_inode}
                if s is None:
                    if active != operation:
                        raise m.Refusal('Unexplained missing original entry')
                    # The prior process may have stopped after unlink/rmdir but
                    # before its progress append. Reaffirm durable absence.
                    os.fsync(parent)
                    progress.append('retired_entry', **operation, recovered_absence=True)
                else:
                    if m.inode_identity(s) != expected_inode or s.st_uid != uid or s.st_gid != source_gid or stat.S_IMODE(s.st_mode) != row['mode']:
                        raise m.Refusal('Original entry ownership/mode/inode changed')
                    if row['kind'] == 'regular_file':
                        if not stat.S_ISREG(s.st_mode) or s.st_nlink != remaining[canonical[key]] or s.st_size != row['size'] or s.st_mtime_ns != row['mtime_ns']:
                            raise m.Refusal('Original file size/mtime/global remaining hardlink count changed')
                        fd = os.open(name, os.O_RDONLY | m.O_NOFOLLOW | m.O_CLOEXEC | os.O_NONBLOCK, dir_fd=parent)
                        try:
                            if m.identity(os.fstat(fd)) != m.identity(s) or m.hash_fd(fd) != row['sha256'] or m.xattr_metadata(fd) != row['xattrs'] or acl_reader(fd, os.fstat(fd)) != row['acl']:
                                raise m.Refusal('Original file byte/attribute identity mismatch before unlink')
                            if m.identity(os.fstat(fd)) != m.identity(s) or m.identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) != m.identity(s):
                                raise m.Refusal('Original file changed during pre-unlink verification')
                            m.atomic_json(dfd, 'retirement-active.json', operation, replace=True)
                            active = operation
                            os.unlink(name, dir_fd=parent)
                        finally:
                            os.close(fd)
                    else:
                        if not stat.S_ISDIR(s.st_mode):
                            raise m.Refusal('Original directory type changed')
                        fd = os.open(name, os.O_RDONLY | m.O_DIRECTORY | m.O_NOFOLLOW | m.O_CLOEXEC, dir_fd=parent)
                        try:
                            if m.inode_identity(os.fstat(fd)) != expected_inode or os.listdir(fd) or m.xattr_metadata(fd) != row['xattrs'] or acl_reader(fd, os.fstat(fd)) != row['acl']:
                                raise m.Refusal('Original directory not empty/identity changed')
                            m.atomic_json(dfd, 'retirement-active.json', operation, replace=True)
                            active = operation
                            os.rmdir(name, dir_fd=parent)
                        finally:
                            os.close(fd)
                    os.fsync(parent)
                    if fail == 'after_remove_before_progress':
                        raise m.InjectedCrash(fail)
                    progress.append('retired_entry', **operation, recovered_absence=False)
                    if fail == 'after_progress_before_clear':
                        raise m.InjectedCrash(fail)
                done[key] = progress.events[-1]
                if row['kind'] == 'regular_file':
                    remaining[canonical[key]] -= 1
                m.atomic_json(dfd, 'retirement-active.json', None, replace=True)
                active = None
        if set(done) != set(manifest['rows']) or set(os.listdir(rfd)) != {'.lock'} or active is not None:
            raise m.Refusal('Retirement incomplete or unexpected private entry')
        validate_links(sfd, names, ops, uid)
        p.verify_target(dfd, names, manifest, uid, storage_gid, acl_reader)
        m.verify_namespace(source, sfd)
        m.verify_namespace(destination, dfd)
        result = {'schema': 1, 'status': 'redundant_originals_retired_project_copy_reverified',
                  'source': str(source), 'destination': str(destination),
                  'publication_proof_sha256': publication_proof_sha256,
                  'historical_access_receipt_sha256': historical_access_receipt_sha256,
                  'retired_entries_observed': len(done),
                  'retired_files_observed': len(files), 'retired_directories_observed': len(dirs),
                  'recovered_absent_entries': sum(bool(row['recovered_absence']) for row in done.values()),
                  'completed_remove_calls_durably_recorded': sum(not row['recovered_absence'] for row in done.values()),
                  'original_unique_inode_count': len(manifest['hardlink_classes']) + len(dirs),
                  'quota_delta_not_inferred_from_entry_count': True,
                  'empty_private_rollback_parent_and_lock_retained': plan['rollback_parent'],
                  'project_copy_final_semantic_sha256': m.digest(m.raw_json(m.semantic(manifest, storage_gid))),
                  'unrelated_path_operations': 0, 'model_calls': 0, 'gpu_calls': 0, 'slurm_submissions': 0}
        if 'retirement-proof.json' in os.listdir(dfd):
            if m.read_json(dfd, 'retirement-proof.json') != result:
                raise m.Refusal('Retirement proof replay mismatch')
            return {**result, 'replay': True}
        m.atomic_json(dfd, 'retirement-proof.json', result)
        return {**result, 'replay': False}
    finally:
        if progress is not None:
            progress.close()
        for fd in (rlock, rfd, dlock, dfd, sfd):
            if fd is not None:
                os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication-proof-sha256', required=True)
    parser.add_argument('--historical-access-receipt', type=Path, required=True)
    parser.add_argument('--historical-access-receipt-sha256', required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--preflight-sha256', required=True)
    parser.add_argument('--acknowledge-redundant-retirement', required=True, choices=['only-four-closed-rollback-copies-after-independent-readonly-attach'])
    args = parser.parse_args()
    if sys.platform != 'linux' or os.getuid() != m.UID or m.STORAGE_GID not in os.getgroups():
        raise m.Refusal('Live retirement requires exact Linux owner/project group')
    m.live_preflight(args.preflight, args.preflight_sha256)
    dfd = m.open_absolute_directory(m.DESTINATION)
    try:
        manifest = m.read_json(dfd, 'source-manifest.json')
    finally:
        os.close(dfd)
    historical_receipt(args.historical_access_receipt.read_bytes(), args.historical_access_receipt_sha256, m.NAMES, m.JOBS, manifest, args.publication_proof_sha256)
    print(json.dumps(retire(m.SOURCE, m.DESTINATION, m.NAMES, m.UID, m.SOURCE_GID, m.STORAGE_GID,
                            publication_proof_sha256=args.publication_proof_sha256,
                            historical_access_receipt_sha256=args.historical_access_receipt_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
