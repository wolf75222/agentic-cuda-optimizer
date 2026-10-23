"""Closed-session publication using features available on the observed GPFS.

Actual synthetic GPFS RENAME_EXCHANGE and absent-target RENAME_NOREPLACE failed
EINVAL. We do not retry/pretend support. An original directory is moved with
ordinary rename into a fresh private 0700 scratch rollback parent under flock;
then os.symlink creates the old pathname atomically WITHOUT overwrite. A brief
missing-path window exists. The private destination, advisory locks, absence of
known writers and exhaustive before/after checks are operational assumptions,
not protection against a malicious same-UID actor. All originals are retained.
No payload unlink or retirement command, Slurm/model/GPU/SQL API exists here.
"""
from __future__ import annotations
import argparse
import collections
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import uuid

spec = importlib.util.spec_from_file_location('staging_helper', Path(__file__).with_name('session_archive_lossless_live_003.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def entry(fd, name):
    try:
        return os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def state(sfd, rfd, name, op):
    original, backup = entry(sfd, name), entry(rfd, name)
    here = original is not None and stat.S_ISDIR(original.st_mode) and m.inode_identity(original) == op['source_inode']
    there = backup is not None and stat.S_ISDIR(backup.st_mode) and m.inode_identity(backup) == op['source_inode']
    if here and backup is None:
        return 'original'
    if original is None and there:
        return 'source_renamed'
    if original is not None and there and op.get('link_inode') is not None and stat.S_ISLNK(original.st_mode) and m.inode_identity(original) == op['link_inode'] and os.readlink(name, dir_fd=sfd) == op['target']:
        return 'published'
    raise m.Refusal('Ambiguous/replaced namespace; no overwrite, all copies preserved')


def verify_target(dfd, names, manifest, uid, gid, acl_reader):
    with m.child_directory(dfd, 'sessions') as fd:
        value = m.inventory(fd, names, uid, gid, acl_reader)
        if sorted(os.listdir(fd)) != sorted(names) or m.semantic(value) != m.semantic(manifest, gid):
            raise m.Refusal('Project copy drifted')


def verify_originals(sfd, rfd, names, operations, manifest, uid, gid, acl_reader):
    """One global hardlink inventory spanning original and rollback parent FDs.

    No synthetic symlink view and no monkeypatch of filesystem APIs is used.
    Only the first directory-FD choice differs from the reviewed staging walk.
    """
    selected = {name: (sfd if state(sfd, rfd, name, operations[name]) == 'original' else rfd) for name in names}
    rows, physical, cached = {}, {}, {}
    groups = collections.defaultdict(list)
    def visit(parent_fd, name, key):
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        directory = stat.S_ISDIR(before.st_mode)
        if not directory and not stat.S_ISREG(before.st_mode):
            raise m.Refusal('Special/symlink original entry')
        fd = os.open(name, os.O_RDONLY | m.O_NOFOLLOW | m.O_CLOEXEC | (m.O_DIRECTORY if directory else os.O_NONBLOCK), dir_fd=parent_fd)
        try:
            s = os.fstat(fd)
            if m.identity(s) != m.identity(before) or s.st_uid != uid or s.st_gid != gid:
                raise m.Refusal('Original entry owner/identity changed')
            inode = (s.st_dev, s.st_ino)
            if inode not in cached:
                cached[inode] = (None if directory else m.hash_fd(fd), m.xattr_metadata(fd), acl_reader(fd, s))
            content, attrs, acl = cached[inode]
            row = {'kind': 'directory' if directory else 'regular_file', 'mode': stat.S_IMODE(s.st_mode),
                   'uid': uid, 'gid': gid, 'mtime_ns': s.st_mtime_ns, 'xattrs': attrs, 'acl': acl}
            if not directory:
                row.update(size=s.st_size, sha256=content)
                groups[inode].append(key)
            rows[key], physical[key] = row, m.identity(s)
            if directory:
                children = sorted(os.listdir(fd))
                for child in children:
                    m.valid_component(child)
                    visit(fd, child, key + '/' + child)
                if sorted(os.listdir(fd)) != children:
                    raise m.Refusal('Original directory entry set changed')
            if m.identity(os.fstat(fd)) != m.identity(s) or m.identity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != m.identity(s):
                raise m.Refusal('Original entry changed during verification')
        finally:
            os.close(fd)
    for name in names:
        visit(selected[name], name, name)
    hardlinks = []
    for paths in groups.values():
        paths.sort()
        if any(physical[key][-1] != len(paths) for key in paths):
            raise m.Refusal('Unexplained external original hardlink')
        hardlinks.append(paths)
    value = {'rows': dict(sorted(rows.items())), 'hardlink_classes': sorted(hardlinks), 'source_identities': dict(sorted(physical.items()))}
    if m.semantic(value) != m.semantic(manifest):
        raise m.Refusal('Original byte/metadata/hardlink graph drift')
    for key, old in manifest['source_identities'].items():
        actual = value['source_identities'][key]
        if key in names and operations[key]['source_rename_intended']:
            # A recorded ordinary rename can update root ctime, including a
            # guarded rollback. All other physical and semantic fields remain.
            if actual[:8] != old[:8] or actual[9:] != old[9:]:
                raise m.Refusal('Renamed root identity drift')
        elif actual != old:
            raise m.Refusal('Original physical entry drift')


def validate_parent(sfd, rfd, plan, uid, gid, acl_reader):
    s = os.fstat(rfd)
    named = os.stat(plan['rollback_parent'], dir_fd=sfd, follow_symlinks=False)
    if not stat.S_ISDIR(s.st_mode) or m.inode_identity(s) != plan['rollback_parent_inode'] or m.inode_identity(named) != m.inode_identity(s) or s.st_uid != uid or s.st_gid != gid or stat.S_IMODE(s.st_mode) != 0o700:
        raise m.Refusal('Private rollback parent identity/permissions changed')
    acl_reader(rfd, s)
    m.xattr_metadata(rfd)
    if set(os.listdir(rfd)) - (set(plan['operations']) | {'.lock'}):
        raise m.Refusal('Unexpected private rollback entry preserved for review')


def publish(source, destination, names, uid, source_gid, storage_gid, *,
            copy_proof_sha256, manifest_sha256, native_failure_receipt_sha256,
            acl_reader=m.gpfs_acl, rename_fn=os.rename, symlink_fn=os.symlink,
            fail=None):
    names = tuple(names)
    sfd, dfd = m.open_absolute_directory(source), m.open_absolute_directory(destination)
    rfd = project_lock = rollback_lock = None
    journal = None
    try:
        ds = os.fstat(dfd)
        if uid != os.getuid() or ds.st_uid != uid or ds.st_gid != storage_gid or stat.S_IMODE(ds.st_mode) != 0o2700:
            raise m.Refusal('Private approved-account destination required')
        project_lock = os.open('.lock', os.O_RDWR | m.O_NOFOLLOW | m.O_CLOEXEC, dir_fd=dfd)
        fcntl.flock(project_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proof, manifest = m.read_json(dfd, 'copy-proof.json'), m.read_json(dfd, 'source-manifest.json')
        if m.digest(m.raw_json(proof)) != copy_proof_sha256 or m.digest(m.raw_json(manifest)) != manifest_sha256 or proof.get('status') != 'staged_verified_no_publication' or proof.get('source_manifest_sha256') != manifest_sha256 or proof.get('script_sha256') != m.digest(Path(m.__file__).read_bytes()):
            raise m.Refusal('Approved copy/manifest/helper authority mismatch')
        if not isinstance(native_failure_receipt_sha256, str) or not re.fullmatch('[0-9a-f]{64}', native_failure_receipt_sha256):
            raise m.Refusal('Observed native feature failure receipt hash required')
        intent = {'schema': 3, 'kind': 'private_rollback_parent_rename_then_symlink_noclobber',
                  'source': str(source), 'destination': str(destination), 'names': list(names),
                  'uid': uid, 'source_gid': source_gid, 'storage_gid': storage_gid,
                  'copy_proof_sha256': copy_proof_sha256, 'manifest_sha256': manifest_sha256,
                  'native_failure_receipt_sha256': native_failure_receipt_sha256,
                  'script_sha256': m.digest(Path(__file__).read_bytes()),
                  'namespace_gap_possible': True, 'malicious_same_uid_actor_protection_claimed': False}
        journal = m.Journal(dfd)
        if 'publication-intent.json' in os.listdir(dfd):
            if m.read_json(dfd, 'publication-intent.json') != intent:
                raise m.Refusal('Publication intent collision')
        else:
            m.atomic_json(dfd, 'publication-intent.json', intent)
        if 'publication-operations.json' not in os.listdir(dfd):
            plan = {'rollback_parent': '.aco-closed-rollback-' + uuid.uuid4().hex,
                    'rollback_parent_inode': None,
                    'operations': {name: {'name': name, 'target': str(Path(destination) / 'sessions' / name),
                                          'source_inode': manifest['source_identities'][name][:2],
                                          'source_rename_intended': False, 'link_inode': None} for name in names}}
            m.atomic_json(dfd, 'publication-operations.json', plan)
            os.mkdir(plan['rollback_parent'], 0o700, dir_fd=sfd)
            rfd = os.open(plan['rollback_parent'], os.O_RDONLY | m.O_DIRECTORY | m.O_NOFOLLOW | m.O_CLOEXEC, dir_fd=sfd)
            os.fchown(rfd, uid, source_gid)
            os.fchmod(rfd, 0o700)
            rollback_lock = os.open('.lock', os.O_RDWR | os.O_CREAT | os.O_EXCL | m.O_NOFOLLOW | m.O_CLOEXEC, 0o600, dir_fd=rfd)
            os.fchown(rollback_lock, uid, source_gid)
            os.fsync(rollback_lock)
            os.fsync(rfd)
            os.fsync(sfd)
            plan['rollback_parent_inode'] = m.inode_identity(os.fstat(rfd))
            m.atomic_json(dfd, 'publication-operations.json', plan, replace=True)
            journal.append('private_rollback_parent_ready', name=plan['rollback_parent'], inode=plan['rollback_parent_inode'])
        else:
            plan = m.read_json(dfd, 'publication-operations.json')
            if plan.get('rollback_parent_inode') is None:
                raise m.Refusal('Unjournaled rollback parent creation preserved for review')
            m.valid_component(plan['rollback_parent'])
            if not plan['rollback_parent'].startswith('.aco-closed-rollback-'):
                raise m.Refusal('Rollback parent basename mismatch')
            rfd = os.open(plan['rollback_parent'], os.O_RDONLY | m.O_DIRECTORY | m.O_NOFOLLOW | m.O_CLOEXEC, dir_fd=sfd)
            rollback_lock = os.open('.lock', os.O_RDWR | m.O_NOFOLLOW | m.O_CLOEXEC, dir_fd=rfd)
        for fd, expected_gid in ((project_lock, storage_gid), (rollback_lock, source_gid)):
            ls = os.fstat(fd)
            if not stat.S_ISREG(ls.st_mode) or ls.st_uid != uid or ls.st_gid != expected_gid or stat.S_IMODE(ls.st_mode) != 0o600 or ls.st_nlink != 1:
                raise m.Refusal('Unsafe advisory lock')
        fcntl.flock(rollback_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_parent(sfd, rfd, plan, uid, source_gid, acl_reader)
        operations = plan['operations']
        if set(operations) != set(names):
            raise m.Refusal('Exact session operation scope mismatch')
        for name, op in operations.items():
            if op.get('name') != name or op.get('target') != str(Path(destination) / 'sessions' / name) or op.get('source_inode') != manifest['source_identities'][name][:2] or type(op.get('source_rename_intended')) is not bool:
                raise m.Refusal('Operation identity/target mismatch')
            state(sfd, rfd, name, op)
        verify_target(dfd, names, manifest, uid, storage_gid, acl_reader)
        verify_originals(sfd, rfd, names, operations, manifest, uid, source_gid, acl_reader)
        if 'publication-proof.json' in os.listdir(dfd):
            done = m.read_json(dfd, 'publication-proof.json')
            if done.get('status') != 'published_all_four_originals_retained_private_parent' or done.get('operations_sha256') != m.digest(m.raw_json(plan)) or any(state(sfd, rfd, name, operations[name]) != 'published' for name in names):
                raise m.Refusal('Committed publication proof mismatch')
            return {**done, 'replay': True, 'source_renames': 0, 'symlink_creations': 0}
        renames = links = 0
        for name in names:
            op = operations[name]
            current = state(sfd, rfd, name, op)
            if current == 'original':
                validate_parent(sfd, rfd, plan, uid, source_gid, acl_reader)
                if entry(rfd, name) is not None:
                    raise m.Refusal('Rollback destination collision')
                op['source_rename_intended'] = True
                m.atomic_json(dfd, 'publication-operations.json', plan, replace=True)
                journal.append('ordinary_source_rename_intended', name=name, inode=op['source_inode'])
                # This is safe under the documented private-namespace and known-
                # writer assumptions, not a fabricated NOREPLACE syscall.
                rename_fn(name, name, src_dir_fd=sfd, dst_dir_fd=rfd)
                os.fsync(sfd)
                os.fsync(rfd)
                renames += 1
                if fail == 'after_source_rename':
                    raise m.InjectedCrash(fail)
                journal.append('ordinary_source_renamed', name=name)
                current = state(sfd, rfd, name, op)
            if current == 'source_renamed':
                try:
                    symlink_fn(op['target'], name, dir_fd=sfd)
                except Exception:
                    # EEXIST/unknown entry never triggers an overwrite. If the
                    # old path remains absent and exact original is retained,
                    # restore it under the same known-writer assumption.
                    if state(sfd, rfd, name, op) == 'source_renamed':
                        validate_parent(sfd, rfd, plan, uid, source_gid, acl_reader)
                        rename_fn(name, name, src_dir_fd=rfd, dst_dir_fd=sfd)
                        os.fsync(sfd)
                        os.fsync(rfd)
                        journal.append('ordinary_source_rename_rolled_back', name=name)
                    raise
                os.fsync(sfd)
                links += 1
                if fail == 'after_symlink_unjournaled':
                    raise m.InjectedCrash(fail)
                op['link_inode'] = m.inode_identity(os.stat(name, dir_fd=sfd, follow_symlinks=False))
                m.atomic_json(dfd, 'publication-operations.json', plan, replace=True)
                if fail == 'after_link_inode_durable':
                    raise m.InjectedCrash(fail)
                if state(sfd, rfd, name, op) != 'published':
                    raise m.Refusal('Published link identity mismatch')
                journal.append('old_path_symlink_published', name=name, inode=op['link_inode'])
            if state(sfd, rfd, name, op) != 'published':
                raise m.Refusal('Session publication incomplete')
            if fail == 'after_session':
                raise m.InjectedCrash(fail)
        verify_originals(sfd, rfd, names, operations, manifest, uid, source_gid, acl_reader)
        verify_target(dfd, names, manifest, uid, storage_gid, acl_reader)
        validate_parent(sfd, rfd, plan, uid, source_gid, acl_reader)
        m.verify_namespace(source, sfd)
        m.verify_namespace(destination, dfd)
        done = {'schema': 3, 'status': 'published_all_four_originals_retained_private_parent',
                'source': str(source), 'destination': str(destination),
                'copy_proof_sha256': copy_proof_sha256, 'source_manifest_sha256': manifest_sha256,
                'operations_sha256': m.digest(m.raw_json(plan)),
                'rollback_parent': plan['rollback_parent'],
                'rollback_parent_inode': plan['rollback_parent_inode'],
                'publication_protocol': 'ordinary rename into fresh private locked0700 scratch parent, followed by atomic no-clobber symlink creation',
                'publication_atomic': False, 'namespace_gap_possible': True,
                'native_renameat2_operations': 0,
                'native_failure_receipt_sha256': native_failure_receipt_sha256,
                'source_payload_unlinks': 0, 'retirement_authorized': False,
                'historical_readonly_attach_still_required': True,
                'malicious_same_uid_actor_protection_claimed': False}
        journal.append('all_publication_verified', operations_sha256=done['operations_sha256'])
        m.atomic_json(dfd, 'publication-proof.json', done)
        return {**done, 'replay': False, 'source_renames': renames, 'symlink_creations': links}
    finally:
        if journal is not None:
            journal.close()
        for fd in (rollback_lock, rfd, project_lock, dfd, sfd):
            if fd is not None:
                os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('copy-proof-sha256', 'manifest-sha256', 'native-failure-receipt-sha256', 'preflight-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--native-failure-receipt', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--acknowledge-publication-only', required=True, choices=['four-closed-roots-private-rollback-rename-and-symlink-with-gap-retain-originals'])
    args = parser.parse_args()
    if sys.platform != 'linux' or os.getuid() != m.UID or m.STORAGE_GID not in os.getgroups():
        raise m.Refusal('Live publication requires exact Linux owner/project group')
    m.live_preflight(args.preflight, args.preflight_sha256)
    raw = args.native_failure_receipt.read_bytes()
    if m.digest(raw) != args.native_failure_receipt_sha256:
        raise m.Refusal('Native failure receipt digest mismatch')
    observed = json.loads(raw).get('probe', {})
    if observed.get('kind') != 'actual_gpfs_rename_noreplace_probe' or observed.get('passed') is not False or observed.get('real_session_operations') != 0 or observed.get('preceding_exchange_probe_failed_errno') != 22 or not any(row.get('target') == 'c' and row.get('errno') == 22 for row in observed.get('rows', [])):
        raise m.Refusal('Matching actual unsupported-native-feature evidence required')
    print(json.dumps(publish(m.SOURCE, m.DESTINATION, m.NAMES, m.UID, m.SOURCE_GID, m.STORAGE_GID,
                             copy_proof_sha256=args.copy_proof_sha256,
                             manifest_sha256=args.manifest_sha256,
                             native_failure_receipt_sha256=args.native_failure_receipt_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
