"""Isolated GPFS SELinux SET/readback/restore probe, never a real-tree relabel.

Read ONLY source metadata and one build directory's security.selinux bytes.
Create exclusively fresh synthetic project objects, keep every result, emit only
metadata and hashes. No privilege change, removexattr, source/real-target writes,
SQL, model, Slurm or GPU operation exists. Absence is not replaced by a made-up
label; permission denial or mismatched readback is a concrete negative result.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import uuid

SOURCE = Path('/gpfs/scratch/rmdraux/agentic-cuda-optimizer-20261021/sessions')
PARTIAL = Path('/gpfs/projet/r250127/rmdraux-agentic-cuda-optimizer-retained-20261022')
NAMES = {'suite-gen-168ebe9470e795deffa60085e0db8815', 'suite-gen-77d89ccf16da23c828d8aa6503eaaf74', 'suite-gen-384811d9ca7a729c0f7ce04ea82c1aef', 'suite-gen-10621470f80015cf6eab7c53689c5f60'}
UID, SOURCE_GID, GID = 100267, 100267, 200109
STAGE_SHA = '6c591396031e8b6575bd46dced4b2444f90868324789182d105479b25e3e2c65'
ATTRIBUTE = 'security.selinux'
FIXTURE = b'SYNTHETIC CAPABILITY PROBE ONLY\n'
NODATA = {getattr(__import__('errno'), 'ENODATA', 61), getattr(__import__('errno'), 'ENOATTR', 93)}


class Refusal(RuntimeError):
    pass


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def raw_json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def label_summary(value):
    return None if value is None else {'size': len(value), 'sha256': sha(value)}


def identity(s):
    return [s.st_dev, s.st_ino, stat.S_IFMT(s.st_mode), s.st_uid, s.st_gid, stat.S_IMODE(s.st_mode), s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink]


def snapshot(fd, labels):
    s = os.fstat(fd)
    return {'identity': identity(s), 'uid': s.st_uid, 'gid': s.st_gid,
            'mode': oct(stat.S_IMODE(s.st_mode)), 'label': label_summary(labels.get(fd)),
            'fixture_sha256': sha(FIXTURE) if stat.S_ISREG(s.st_mode) else None}


def write_all(fd, raw):
    while raw:
        n = os.write(fd, raw)
        if n <= 0:
            raise OSError('Short write')
        raw = raw[n:]


def component(name):
    if not isinstance(name, str) or not name or name in ('.', '..') or '/' in name or '\x00' in name:
        raise Refusal('Unsafe path component')
    return name


def directory(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise Refusal('Absolute canonical directory required')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in path.parts[1:]:
            nxt = os.open(component(name), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def read_metadata_json(fd, name):
    file_fd = os.open(component(name), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != UID or before.st_nlink != 1:
            raise Refusal('Unsafe stage metadata JSON')
        with os.fdopen(file_fd, 'rb', closefd=False) as stream:
            raw = stream.read()
        if identity(os.fstat(file_fd)) != identity(before):
            raise Refusal('Stage metadata changed during read')
        return json.loads(raw), sha(raw)
    finally:
        os.close(file_fd)


class Labels:
    def get(self, fd):
        try:
            return os.getxattr(fd, ATTRIBUTE)
        except OSError as error:
            if error.errno in NODATA:
                return None
            raise

    def replace(self, fd, value):
        if value is None:
            raise Refusal('No removexattr or invented label permitted')
        os.setxattr(fd, ATTRIBUTE, value, os.XATTR_REPLACE)


class Journal:
    def __init__(self, root_fd, uid, gid):
        self.root_fd = root_fd
        self.fd = os.open('phases.jsonl', os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=root_fd)
        os.fchown(self.fd, uid, gid)
        self.sequence = 0
        os.fsync(self.fd)
        os.fsync(root_fd)

    def append(self, phase, **fields):
        write_all(self.fd, raw_json({'sequence': self.sequence, 'phase': phase, **fields}))
        os.fsync(self.fd)
        self.sequence += 1

    def close(self):
        os.close(self.fd)


def exercise(root_fd, name, kind, desired, uid, gid, journal, labels=None):
    """Only an empty dir or explicit synthetic byte fixture under fresh root."""
    labels = labels or Labels()
    if kind == 'directory':
        os.mkdir(name, 0o700, dir_fd=root_fd)
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        mode = 0o700
    else:
        fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=root_fd)
        write_all(fd, FIXTURE)
        mode = 0o600
    result = {'name': name, 'kind': kind, 'qualified': False, 'set_attempts': 0, 'restore_attempts': 0}
    initial = None
    try:
        os.fchown(fd, uid, gid)
        os.fchmod(fd, mode)
        os.fsync(fd)
        os.fsync(root_fd)
        initial = labels.get(fd)
        result.update(before=snapshot(fd, labels), desired_label=label_summary(desired),
                      real_transition_required=initial != desired)
        journal.append('fixture_baseline', name=name, before=result['before'], desired_label=result['desired_label'])
        if initial is None or desired is None:
            result['refusal'] = 'Absent label needs a separate explicit preservation policy; no remove/create is attempted.'
            return result
        if initial == desired:
            result['refusal'] = 'Identical-label rewrite cannot qualify a distinct-label transition.'
            return result
        journal.append('set_intent', name=name, from_label=label_summary(initial), to_label=label_summary(desired))
        result['set_attempts'] += 1
        try:
            labels.replace(fd, desired)
            os.fsync(fd)
            actual = labels.get(fd)
            result['after_set'] = snapshot(fd, labels)
            journal.append('set_returned', name=name, label=label_summary(actual), exact_readback=actual == desired)
            if actual != desired:
                result['refusal'] = 'SET returned without exact source-label readback.'
                return result
            # Match the stage ownership/mode/mtime operations, then re-open the
            # same synthetic inode. Never execute these against the source FD.
            old_inode = identity(os.fstat(fd))[:2]
            os.fchown(fd, uid, gid)
            os.fchmod(fd, mode)
            stamp = 1700000000000000000
            os.utime(fd, ns=(stamp, stamp))
            os.fsync(fd)
            reopened = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | (os.O_DIRECTORY if kind == 'directory' else os.O_NONBLOCK), dir_fd=root_fd)
            try:
                if identity(os.fstat(reopened))[:2] != old_inode or labels.get(reopened) != desired:
                    raise Refusal('Label/inode changed after metadata operations and reopen')
                if kind == 'regular_file':
                    os.lseek(reopened, 0, os.SEEK_SET)
                    if os.read(reopened, len(FIXTURE) + 1) != FIXTURE:
                        raise Refusal('Synthetic fixture bytes changed')
            finally:
                os.close(reopened)
            retained_name = name + '.installed' if kind == 'regular_file' else name + '.renamed'
            if kind == 'regular_file':
                os.link(name, retained_name, src_dir_fd=root_fd, dst_dir_fd=root_fd, follow_symlinks=False)
                os.fsync(root_fd)
                linked = os.stat(retained_name, dir_fd=root_fd, follow_symlinks=False)
                if identity(linked)[:2] != old_inode or linked.st_nlink != 2:
                    raise Refusal('Synthetic hardlink install identity mismatch')
                os.unlink(name, dir_fd=root_fd)
            else:
                try:
                    os.stat(retained_name, dir_fd=root_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise Refusal('Synthetic rename destination collision')
                os.rename(name, retained_name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
            os.fsync(root_fd)
            transported = os.open(retained_name, os.O_RDONLY | os.O_NOFOLLOW | (os.O_DIRECTORY if kind == 'directory' else os.O_NONBLOCK), dir_fd=root_fd)
            try:
                if identity(os.fstat(transported))[:2] != old_inode or labels.get(transported) != desired:
                    raise Refusal('Synthetic install/rename changed inode or label')
                if kind == 'regular_file' and os.read(transported, len(FIXTURE) + 1) != FIXTURE:
                    raise Refusal('Synthetic installed bytes changed')
            finally:
                os.close(transported)
            result['retained_name'] = retained_name
            result['transport_verified'] = True
            result['qualified'] = True
            journal.append('reopen_readback_verified', name=name, inode=old_inode, label=label_summary(desired))
        except OSError as error:
            result.update(refusal='SET/readback/metadata capability failure', errno=error.errno, error_type=type(error).__name__, qualified=False)
            journal.append('capability_refused', name=name, errno=error.errno, error_type=type(error).__name__)
        except Refusal as error:
            result.update(refusal=str(error), qualified=False)
            journal.append('verification_refused', name=name, reason=str(error))
    finally:
        try:
            if initial is not None:
                now = labels.get(fd)
                if now != initial:
                    journal.append('restore_initial_label_intent', name=name, initial_label=label_summary(initial))
                    result['restore_attempts'] += 1
                    try:
                        labels.replace(fd, initial)
                        os.fsync(fd)
                    except OSError as error:
                        result.update(qualified=False, restoration_errno=error.errno, restoration_error_type=type(error).__name__)
                result['initial_label_restored_exactly'] = labels.get(fd) == initial
                if not result['initial_label_restored_exactly']:
                    result['qualified'] = False
                result['after_restore'] = snapshot(fd, labels)
                journal.append('restore_checked', name=name, restored_exactly=result['initial_label_restored_exactly'], label=result['after_restore']['label'])
        finally:
            os.fsync(root_fd)
            os.close(fd)
    return result


def main():
    if sys.platform != 'linux' or os.getuid() != UID or GID not in os.getgroups():
        raise Refusal('Exact Linux owner/project group required')
    script_sha = os.environ.get('ACO_PROBE_SOURCE_SHA', '')
    if len(script_sha) != 64 or any(ch not in '0123456789abcdef' for ch in script_sha):
        raise Refusal('Source hash pin required')
    partial_fd, parent_fd = directory(PARTIAL), directory(PARTIAL.parent)
    source_fd = probe_fd = journal = None
    result = None
    try:
        intent, intent_sha = read_metadata_json(partial_fd, 'intent.json')
        manifest, manifest_sha = read_metadata_json(partial_fd, 'source-manifest.json')
        if intent.get('script_sha256') != STAGE_SHA or set(intent.get('names', [])) != NAMES or intent.get('source') != str(SOURCE) or intent.get('destination') != str(PARTIAL):
            raise Refusal('Exact preserved stage003 authority mismatch')
        labels = Labels()
        inherited_parent = labels.get(parent_fd)
        candidates = [key for key, row in sorted(manifest['rows'].items()) if key.split('/')[0] in NAMES and key.split('/')[-1] == 'build' and row['kind'] == 'directory' and ({k: v for k, v in row['xattrs'].get(ATTRIBUTE, {}).items() if k in ('size', 'sha256')} or None) != label_summary(inherited_parent)]
        if not candidates:
            raise Refusal('No manifest-pinned build variant differs from project parent; no invented transition')
        key = candidates[0]
        source_fd = directory(SOURCE / key)
        before_source = identity(os.fstat(source_fd))
        desired = labels.get(source_fd)
        wanted = manifest['rows'][key]['xattrs'].get(ATTRIBUTE)
        wanted_summary = None if wanted is None else {k: wanted[k] for k in ('size', 'sha256')}
        if before_source != manifest['source_identities'][key] or before_source[3:5] != [UID, SOURCE_GID] or label_summary(desired) != wanted_summary:
            raise Refusal('Source build identity/label drift since stage003 preparation')
        name = '.aco-selinux-capability-probe-' + uuid.uuid4().hex
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        probe_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        os.fchown(probe_fd, UID, GID)
        os.fchmod(probe_fd, 0o700)
        os.fsync(probe_fd)
        os.fsync(parent_fd)
        journal = Journal(probe_fd, UID, GID)
        authority = {'kind': 'isolated_gpfs_selinux_set_readback_restore_probe', 'path': str(PARTIAL.parent / name),
                     'script_sha256': script_sha, 'stage003_intent_sha256': intent_sha, 'stage003_manifest_sha256': manifest_sha,
                     'source_build_role': key, 'source_build_identity': before_source, 'source_label': label_summary(desired),
                     'project_parent_label': label_summary(inherited_parent), 'source_mutations': 0, 'real_staging_target_mutations': 0,
                     'probe_uid': UID, 'probe_gid': GID, 'probe_mode': '0700', 'numeric_source_payloads_read': False}
        journal.append('probe_authorized', **authority)
        rows = [exercise(probe_fd, 'fixture.bin', 'regular_file', desired, UID, GID, journal, labels),
                exercise(probe_fd, 'fixture-directory', 'directory', desired, UID, GID, journal, labels)]
        unchanged = identity(os.fstat(source_fd)) == before_source and labels.get(source_fd) == desired
        result = {**authority, 'observed_at_unix': time.time(), 'rows': rows,
                  'source_build_unchanged': unchanged, 'qualified': unchanged and all(row['qualified'] for row in rows),
                  'probe_objects_retained': True, 'slurm_submissions': 0, 'model_calls': 0, 'gpu_calls': 0,
                  'scope': 'One exact manifest-pinned build label on synthetic file+directory only; no permission to relabel any real archive object.'}
        journal.append('probe_complete', qualified=result['qualified'], source_build_unchanged=unchanged)
        fd = os.open('receipt.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=probe_fd)
        try:
            os.fchown(fd, UID, GID)
            write_all(fd, raw_json(result))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(probe_fd)
        os.fsync(parent_fd)
        print(json.dumps(result, sort_keys=True))
    finally:
        if journal is not None:
            journal.close()
        for fd in (source_fd, probe_fd, partial_fd, parent_fd):
            if fd is not None:
                os.close(fd)
    if result is None or not result['qualified']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
