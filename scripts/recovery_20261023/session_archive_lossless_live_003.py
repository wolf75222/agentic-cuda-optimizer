"""Opaque, staging-only copy of four pinned, closed GENERATION transport trees.

No source rename/unlink, symlink publication, scheduler, model, SQL or GPU API is
implemented. Live CLI is Linux/GPFS-only. Synthetic tests inject the ACL reader.
Native GPFS ACLs are inspected with mmgetacl -k native: translating NFSv4 ACLs
to POSIX would silently discard information and is deliberately forbidden.
Only basic, mode-representable native POSIX ACLs and user.* xattrs are supported;
security.selinux is allowed solely when inherited byte-exact on every target
inode, never by relabelling, setting, removing or ignoring that attribute.
extended/default/NFSv4 ACLs and other xattrs stop before any source mutation
(there is no source mutation). All payloads are streamed opaquely.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import uuid


SOURCE = Path('/gpfs/scratch/rmdraux/agentic-cuda-optimizer-20261021/sessions')
DESTINATION = Path('/gpfs/projet/r250127/rmdraux-agentic-cuda-optimizer-retained-20261022')
NAMES = (
    'suite-gen-168ebe9470e795deffa60085e0db8815',
    'suite-gen-77d89ccf16da23c828d8aa6503eaaf74',
    'suite-gen-384811d9ca7a729c0f7ce04ea82c1aef',
    'suite-gen-10621470f80015cf6eab7c53689c5f60',
)
JOBS = dict(zip(NAMES, ('775251', '775252', '775253', '775254')))
UID = 100267
SOURCE_GID = 100267
STORAGE_GID = 200109
MMGETACL = '/usr/lpp/mmfs/bin/mmgetacl'
O_DIRECTORY = getattr(os, 'O_DIRECTORY', 0)
O_NOFOLLOW = getattr(os, 'O_NOFOLLOW', 0)
O_CLOEXEC = getattr(os, 'O_CLOEXEC', 0)


class Refusal(RuntimeError):
    pass


class InjectedCrash(RuntimeError):
    pass


def raw_json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(s):
    return [s.st_dev, s.st_ino, stat.S_IFMT(s.st_mode), s.st_uid, s.st_gid,
            stat.S_IMODE(s.st_mode), s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink]


def inode_identity(s):
    return [s.st_dev, s.st_ino]


def valid_component(name):
    if not isinstance(name, str) or not name or name in ('.', '..') or '/' in name or '\x00' in name or '\n' in name:
        raise Refusal('Unsafe path component')
    return name


def parts(key):
    return [valid_component(p) for p in key.split('/')]


def open_absolute_directory(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise Refusal('Absolute canonical directory path required')
    fd = os.open('/', os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
    try:
        for name in path.parts[1:]:
            nxt = os.open(valid_component(name), os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


@contextlib.contextmanager
def child_directory(root_fd, key):
    fd = os.dup(root_fd)
    try:
        for name in parts(key):
            nxt = os.open(name, os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = nxt
        yield fd
    finally:
        os.close(fd)


@contextlib.contextmanager
def parent_and_name(root_fd, key):
    components = parts(key)
    if len(components) == 1:
        yield root_fd, components[0]
    else:
        with child_directory(root_fd, '/'.join(components[:-1])) as fd:
            yield fd, components[-1]


def open_file(root_fd, key, flags=os.O_RDONLY):
    with parent_and_name(root_fd, key) as (parent, name):
        return os.open(name, flags | O_NOFOLLOW | O_CLOEXEC | os.O_NONBLOCK, dir_fd=parent)


def write_all(fd, raw):
    view = memoryview(raw)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise OSError('Short write')
        view = view[n:]


def atomic_json(root_fd, name, value, *, replace=False):
    """Private root only. No overwrite for immutable manifests/intent/proof."""
    valid_component(name)
    temporary = '.atomic-' + uuid.uuid4().hex
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600, dir_fd=root_fd)
    try:
        write_all(fd, raw_json(value))
        os.fsync(fd)
    finally:
        os.close(fd)
    if replace:
        os.rename(temporary, name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
    else:
        # Hardlink is an atomic no-clobber publication on this one filesystem.
        os.link(temporary, name, src_dir_fd=root_fd, dst_dir_fd=root_fd, follow_symlinks=False)
        os.unlink(temporary, dir_fd=root_fd)
    os.fsync(root_fd)


def read_json(root_fd, name):
    fd = os.open(valid_component(name), os.O_RDONLY | O_NOFOLLOW | O_CLOEXEC | os.O_NONBLOCK, dir_fd=root_fd)
    try:
        s = os.fstat(fd)
        if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or s.st_uid != os.getuid():
            raise Refusal('Unsafe bookkeeping file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read()
        if identity(os.fstat(fd)) != identity(s):
            raise Refusal('Bookkeeping file changed during read')
        return json.loads(raw)
    finally:
        os.close(fd)


def parse_native_acl(raw, mode, *, default=False):
    """Only native POSIX three-mode-entry ACLs; never a translated ACL."""
    text = raw.decode('utf-8', errors='strict')
    if 'nfs' in text.lower():
        raise Refusal('Native NFSv4 ACL cannot be preserved by this utility')
    entries = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if default:
        if entries:
            raise Refusal('Default ACL is unsupported')
        return []
    if len(entries) != 3:
        raise Refusal('Extended or unrecognized native ACL')
    result = []
    for index, role in enumerate(('user', 'group', 'other')):
        match = re.fullmatch(role + r'::([r-][w-][x-])([c-]?)', entries[index])
        expected = ''.join(ch if mode & bit else '-' for ch, bit in zip('rwx', (4 << (6-3*index), 2 << (6-3*index), 1 << (6-3*index))))
        if match is None or match.group(1) != expected or (match.group(2) == 'c' and role != 'user'):
            raise Refusal('Native ACL is not basic mode-representable POSIX')
        result.append(entries[index])
    return result


def gpfs_acl(fd, s):
    def invoke(default=False):
        command = [MMGETACL] + (['-d'] if default else []) + ['-k', 'native', '/proc/self/fd/' + str(fd)]
        result = subprocess.run(command, pass_fds=(fd,), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0 or result.stderr:
            # No stdout/stderr contents: metadata failure may include opaque path names.
            raise Refusal('Native GPFS ACL inspection failed (exit=%s, stderr_sha256=%s)' % (result.returncode, digest(result.stderr)))
        return parse_native_acl(result.stdout, stat.S_IMODE(s.st_mode), default=default)
    access = invoke()
    default = invoke(True) if stat.S_ISDIR(s.st_mode) else []
    return {'access': access, 'default': default, 'inspection': 'mmgetacl -k native'}


def xattr_metadata(fd):
    result = {}
    for name in sorted(os.listxattr(fd)):
        if not name.startswith('user.') and name != 'security.selinux':
            raise Refusal('Unsupported extended attribute name: ' + name)
        value = os.getxattr(fd, name)
        result[name] = {'size': len(value), 'sha256': digest(value),
                        'preservation': 'inherited_exact' if name == 'security.selinux' else 'copy'}
    return result


def verify_inherited_attributes(fd, row):
    actual = {name: value for name, value in xattr_metadata(fd).items() if value['preservation'] == 'inherited_exact'}
    wanted = {name: value for name, value in row['xattrs'].items() if value['preservation'] == 'inherited_exact'}
    if actual != wanted:
        raise Refusal('Inherited security label absent/different; relabelling is forbidden')


def hash_fd(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    h = hashlib.sha256()
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            return h.hexdigest()
        h.update(block)


def inventory(root_fd, names, uid, gid, acl_reader=gpfs_acl):
    """Opaque content and global hardlink graph, traversed through directory FDs."""
    rows = {}
    physical = {}
    groups = collections.defaultdict(list)
    cached = {}

    def visit(parent_fd, name, key):
        s0 = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        directory = stat.S_ISDIR(s0.st_mode)
        if not (directory or stat.S_ISREG(s0.st_mode)):
            raise Refusal('Symlink or special source/destination entry: ' + key)
        fd = os.open(name, os.O_RDONLY | O_NOFOLLOW | O_CLOEXEC | (O_DIRECTORY if directory else os.O_NONBLOCK), dir_fd=parent_fd)
        try:
            s = os.fstat(fd)
            if identity(s) != identity(s0) or s.st_uid != uid or s.st_gid != gid:
                raise Refusal('Entry identity/owner changed or outside policy: ' + key)
            inode = (s.st_dev, s.st_ino)
            if inode in cached:
                content, attrs, acl = cached[inode]
            else:
                attrs = xattr_metadata(fd)
                acl = acl_reader(fd, s)
                content = None if directory else hash_fd(fd)
                cached[inode] = (content, attrs, acl)
            row = {'kind': 'directory' if directory else 'regular_file',
                   'mode': stat.S_IMODE(s.st_mode), 'uid': s.st_uid, 'gid': s.st_gid,
                   'mtime_ns': s.st_mtime_ns, 'xattrs': attrs, 'acl': acl}
            if not directory:
                row.update(size=s.st_size, sha256=content)
                groups[inode].append(key)
            rows[key] = row
            physical[key] = identity(s)
            if directory:
                children = sorted(os.listdir(fd))
                for child in children:
                    valid_component(child)
                    visit(fd, child, key + '/' + child)
                if sorted(os.listdir(fd)) != children:
                    raise Refusal('Directory entries changed during inventory: ' + key)
            if identity(os.fstat(fd)) != identity(s) or identity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != identity(s):
                raise Refusal('Entry changed during inventory: ' + key)
        finally:
            os.close(fd)

    for name in names:
        valid_component(name)
        visit(root_fd, name, name)
    hardlinks = []
    for inode, paths in groups.items():
        paths.sort()
        if any(physical[path][-1] != len(paths) for path in paths):
            raise Refusal('Unexplained external hardlink')
        hardlinks.append(paths)
    return {'rows': dict(sorted(rows.items())), 'hardlink_classes': sorted(hardlinks),
            'source_identities': dict(sorted(physical.items()))}


def semantic(manifest, gid=None):
    result = json.loads(json.dumps({k: v for k, v in manifest.items() if k != 'source_identities'}))
    if gid is not None:
        for row in result['rows'].values():
            row['gid'] = gid
    return result


class Journal:
    def __init__(self, root_fd):
        self.root_fd = root_fd
        self.fd = os.open('journal.jsonl', os.O_RDWR | os.O_CREAT | os.O_APPEND | O_NOFOLLOW | O_CLOEXEC, 0o600, dir_fd=root_fd)
        s = os.fstat(self.fd)
        if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_nlink != 1:
            raise Refusal('Unsafe journal')
        os.lseek(self.fd, 0, os.SEEK_SET)
        with os.fdopen(self.fd, 'rb', closefd=False) as stream:
            raw = stream.read()
        if raw and not raw.endswith(b'\n'):
            raise Refusal('Torn journal preserved; manual bookkeeping review required')
        self.events = [json.loads(line) for line in raw.splitlines()]
        for index, event in enumerate(self.events):
            if event.get('sequence') != index:
                raise Refusal('Journal sequence mismatch')
        os.fsync(root_fd)

    def append(self, action, **fields):
        event = {'sequence': len(self.events), 'action': action, **fields}
        write_all(self.fd, raw_json(event))
        os.fsync(self.fd)
        self.events.append(event)

    def close(self):
        os.close(self.fd)


def apply_metadata(fd, row, source_fd, gid):
    verify_inherited_attributes(fd, row)
    # Ownership/group first because chown may clear set-ID bits.
    os.fchown(fd, row['uid'], gid)
    os.fchmod(fd, row['mode'])
    for name, wanted in row['xattrs'].items():
        value = os.getxattr(source_fd, name)
        if {'size': len(value), 'sha256': digest(value), 'preservation': wanted['preservation']} != wanted:
            raise Refusal('Source xattr drift during copy')
        if wanted['preservation'] == 'copy':
            os.setxattr(fd, name, value)
        elif os.getxattr(fd, name) != value:
            raise Refusal('Inherited security label drift; relabelling is forbidden')
    os.utime(fd, ns=(row['mtime_ns'], row['mtime_ns']))
    os.fsync(fd)


def install_regular_noclobber(src_fd, src, dst_fd, dst, *, fail=None):
    """Regular files only: atomic no-clobber hardlink, then unlink temp.

    Actual GPFS renameat2(RENAME_NOREPLACE) and RENAME_EXCHANGE both returned
    EINVAL for absent-target moves in the synthetic root probes. Do not call
    those unsupported operations. The two-name hardlink window is recoverable
    only through the exact durable active inode, never through a name pattern.
    """
    before = os.stat(src, dir_fd=src_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or before.st_nlink != 1:
        raise Refusal('Regular single-link owned temp required for installation')
    os.link(src, dst, src_dir_fd=src_fd, dst_dir_fd=dst_fd, follow_symlinks=False)
    os.fsync(dst_fd)
    if fail == 'after_file_link_before_temp_unlink':
        raise InjectedCrash(fail)
    after_src = os.stat(src, dir_fd=src_fd, follow_symlinks=False)
    after_dst = os.stat(dst, dir_fd=dst_fd, follow_symlinks=False)
    if inode_identity(after_src) != inode_identity(before) or inode_identity(after_dst) != inode_identity(before) or after_src.st_nlink != 2 or after_dst.st_nlink != 2 or identity(after_src) != identity(after_dst):
        raise Refusal('New installed file/temp pair identity mismatch; retain both')
    os.unlink(src, dir_fd=src_fd)
    os.fsync(src_fd)


def reconcile_installed_pair(root_fd, pfd, name, index, key, active, existing, row, uid, gid):
    if active.get('kind') != 'copy' or active.get('path') != key or active.get('inode') != inode_identity(existing) or active.get('temporary') != '.copy-' + str(index):
        raise Refusal('Installed active copy identity mismatch')
    try:
        temporary = os.stat(active['temporary'], dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        if existing.st_nlink != 1:
            raise Refusal('Installed active copy has an unexplained external link')
        return
    if not stat.S_ISREG(temporary.st_mode) or temporary.st_uid != uid or temporary.st_gid != gid or inode_identity(temporary) != active['inode'] or identity(temporary) != identity(existing) or temporary.st_nlink != 2:
        raise Refusal('Installed target/temp pair is not the exact journaled two-link inode')
    # check_leaf has already streamed and verified this same inode. Re-stat both
    # names directly before unlinking only the exact private temporary alias.
    if identity(os.stat(name, dir_fd=pfd, follow_symlinks=False)) != identity(existing) or identity(os.stat(active['temporary'], dir_fd=root_fd, follow_symlinks=False)) != identity(temporary):
        raise Refusal('Installed pair changed before temporary alias retirement')
    os.unlink(active['temporary'], dir_fd=root_fd)
    os.fsync(root_fd)
    os.fsync(pfd)


def source_file_checked(source_fd, key, manifest):
    fd = open_file(source_fd, key)
    if identity(os.fstat(fd)) != manifest['source_identities'][key]:
        os.close(fd)
        raise Refusal('Source drift before copying: ' + key)
    return fd


def check_leaf(fd, row, gid, acl_reader):
    s = os.fstat(fd)
    if not stat.S_ISREG(s.st_mode) or s.st_uid != row['uid'] or s.st_gid != gid:
        raise Refusal('Staged file owner/type mismatch')
    if (stat.S_IMODE(s.st_mode), s.st_size, s.st_mtime_ns) != (row['mode'], row['size'], row['mtime_ns']):
        raise Refusal('Staged file metadata mismatch')
    if hash_fd(fd) != row['sha256'] or xattr_metadata(fd) != row['xattrs'] or acl_reader(fd, s) != row['acl']:
        raise Refusal('Staged file byte/attribute/ACL mismatch')
    if identity(os.fstat(fd)) != identity(s):
        raise Refusal('Staged file changed during verification')


def stage_copy(source, destination, names, uid, source_gid, storage_gid, *,
               acl_reader=gpfs_acl, authorization=None, fail=None):
    """Resume only this exact private journal. Source bytes never modified.

    Explicit crash windows with ambiguous, unjournaled creations fail closed.
    No directory is recursively deleted, even when validation fails.
    """
    names = tuple(names)
    if len(set(names)) != len(names) or not names or uid != os.getuid():
        raise Refusal('Invalid source scope/UID')
    source_fd = open_absolute_directory(source)
    parent_fd = open_absolute_directory(Path(destination).parent)
    root_fd = lock_fd = sessions_fd = None
    journal = None
    try:
        # Native/default ACL compatibility of the real project parent is proved
        # before creation. Extended inheritance is never stripped to make it pass.
        parent_stat = os.fstat(parent_fd)
        acl_reader(parent_fd, parent_stat)
        # Audit ALL source ACLs/xattrs/global hardlinks before destination mkdir.
        # Source bytes remain opaque, including during this inspection.
        now = inventory(source_fd, names, uid, source_gid, acl_reader)
        if authorization is not None and 'role_sha256' in authorization:
            for session, roles in authorization['role_sha256'].items():
                for role, expected in roles.items():
                    row = now['rows'].get(session + '/' + role)
                    if row is None or row.get('kind') != 'regular_file' or row.get('sha256') != expected:
                        raise Refusal('Pinned historical transport role changed before staging')
        basename = valid_component(Path(destination).name)
        expected_intent = {'schema': 1, 'kind': 'opaque_generation_staging_only',
                           'source': str(source), 'destination': str(destination), 'names': list(names),
                           'uid': uid, 'source_gid': source_gid, 'storage_gid': storage_gid,
                           'authorization': authorization, 'source_mutation_operations': 0,
                           'script_sha256': digest(Path(__file__).read_bytes())}
        try:
            os.mkdir(basename, 0o700, dir_fd=parent_fd)
            created = True
            os.fsync(parent_fd)
        except FileExistsError:
            created = False
        root_fd = os.open(basename, os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC, dir_fd=parent_fd)
        if created:
            os.fchown(root_fd, uid, storage_gid)
            os.fchmod(root_fd, 0o2700)
            os.fsync(root_fd)
        root_stat = os.fstat(root_fd)
        if root_stat.st_uid != uid or root_stat.st_gid != storage_gid or stat.S_IMODE(root_stat.st_mode) != 0o2700:
            raise Refusal('Destination root must be private and charged to the approved group')
        lock_fd = os.open('.lock', os.O_RDWR | os.O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600, dir_fd=root_fd)
        lock_stat = os.fstat(lock_fd)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1 or lock_stat.st_uid != uid:
            raise Refusal('Unsafe lock file')
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if created:
            atomic_json(root_fd, 'intent.json', expected_intent)
        elif read_json(root_fd, 'intent.json') != expected_intent:
            raise Refusal('Existing destination intent collision')
        allowed = {'.lock', 'intent.json', 'source-manifest.json', 'journal.jsonl', 'active.json', 'sessions', 'copy-proof.json'}
        resume_active = read_json(root_fd, 'active.json') if 'active.json' in os.listdir(root_fd) else None
        if resume_active is not None and resume_active.get('kind') == 'copy':
            temporary = resume_active.get('temporary')
            if not isinstance(temporary, str) or not re.fullmatch(r'\.copy-[0-9]+', temporary):
                raise Refusal('Invalid active temporary filename')
            allowed.add(temporary)
        if set(os.listdir(root_fd)) - allowed:
            raise Refusal('Unrecognized destination bookkeeping entry; preserve and review')
        journal = Journal(root_fd)
        if 'source-manifest.json' in os.listdir(root_fd):
            manifest = read_json(root_fd, 'source-manifest.json')
            if now != manifest:
                raise Refusal('Source drift since durable preparation')
        else:
            manifest = now
            atomic_json(root_fd, 'source-manifest.json', manifest)
            journal.append('source_manifest_fsynced', sha256=digest(raw_json(manifest)))
        if fail == 'after_manifest':
            raise InjectedCrash(fail)
        if 'copy-proof.json' in os.listdir(root_fd):
            proof = read_json(root_fd, 'copy-proof.json')
            with child_directory(root_fd, 'sessions') as check_fd:
                if semantic(inventory(check_fd, names, uid, storage_gid, acl_reader)) != semantic(manifest, storage_gid):
                    raise Refusal('Committed destination drift')
            if proof['source_manifest_sha256'] != digest(raw_json(manifest)):
                raise Refusal('Committed proof mismatch')
            verify_namespace(source, source_fd)
            verify_namespace(destination, root_fd)
            return {**proof, 'replay': True, 'copy_operations': 0}
        mkdir_events = {event['path']: event['inode'] for event in journal.events if event['action'] == 'mkdir_done'}
        if 'sessions' not in os.listdir(root_fd):
            journal.append('mkdir_intent', path='sessions')
            os.mkdir('sessions', 0o700, dir_fd=root_fd)
            with child_directory(root_fd, 'sessions') as tmp_fd:
                os.fchown(tmp_fd, uid, storage_gid)
                os.fchmod(tmp_fd, 0o700)
                os.fsync(tmp_fd)
                entry_inode = inode_identity(os.fstat(tmp_fd))
            os.fsync(root_fd)
            journal.append('mkdir_done', path='sessions', inode=entry_inode)
            mkdir_events['sessions'] = entry_inode
        sessions_fd = os.open('sessions', os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC, dir_fd=root_fd)
        if inode_identity(os.fstat(sessions_fd)) != mkdir_events.get('sessions'):
            raise Refusal('Unjournaled/replaced staging sessions root')
        directories = [(key, row) for key, row in manifest['rows'].items() if row['kind'] == 'directory']
        for key, row in sorted(directories, key=lambda item: (item[0].count('/'), item[0])):
            with parent_and_name(sessions_fd, key) as (pfd, name):
                try:
                    s = os.stat(name, dir_fd=pfd, follow_symlinks=False)
                except FileNotFoundError:
                    journal.append('mkdir_intent', path=key)
                    os.mkdir(name, 0o700, dir_fd=pfd)
                    fd = os.open(name, os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC, dir_fd=pfd)
                    try:
                        verify_inherited_attributes(fd, row)
                        os.fchown(fd, uid, storage_gid)
                        os.fchmod(fd, 0o700)
                        os.fsync(fd)
                        s = os.fstat(fd)
                    finally:
                        os.close(fd)
                    os.fsync(pfd)
                    journal.append('mkdir_done', path=key, inode=inode_identity(s))
                    mkdir_events[key] = inode_identity(s)
                if not stat.S_ISDIR(s.st_mode) or s.st_uid != uid or s.st_gid != storage_gid or inode_identity(s) != mkdir_events.get(key):
                    raise Refusal('Directory collision or unjournaled creation: ' + key)
        done = {event['path']: event['inode'] for event in journal.events if event['action'] == 'file_done'}
        canonical = {path: group[0] for group in manifest['hardlink_classes'] for path in group}
        copied = 0
        for index, (key, row) in enumerate(manifest['rows'].items()):
            if row['kind'] != 'regular_file':
                continue
            active = read_json(root_fd, 'active.json') if 'active.json' in os.listdir(root_fd) else None
            if active is not None and active['path'] != key and key not in done:
                raise Refusal('Active operation order mismatch')
            with parent_and_name(sessions_fd, key) as (pfd, name):
                try:
                    existing = os.stat(name, dir_fd=pfd, follow_symlinks=False)
                except FileNotFoundError:
                    existing = None
                if existing is not None:
                    admitted_inode = done.get(key) or (active or {}).get('inode')
                    if inode_identity(existing) != admitted_inode:
                        raise Refusal('Unjournaled destination collision: ' + key)
                    fd = os.open(name, os.O_RDONLY | O_NOFOLLOW | O_CLOEXEC | os.O_NONBLOCK, dir_fd=pfd)
                    try:
                        check_leaf(fd, row, storage_gid, acl_reader)
                    finally:
                        os.close(fd)
                    if key not in done:
                        if canonical[key] == key:
                            reconcile_installed_pair(root_fd, pfd, name, index, key, active or {}, existing, row, uid, storage_gid)
                        journal.append('file_done', path=key, inode=inode_identity(existing))
                        done[key] = inode_identity(existing)
                        atomic_json(root_fd, 'active.json', None, replace=True)
                    elif active is not None and active.get('path') == key:
                        # The durable file_done record may precede the atomic
                        # active clear when the process stops. Authenticate that
                        # exact already-committed operation before clearing it.
                        if active.get('inode') != done[key]:
                            raise Refusal('Committed active inode mismatch')
                        if canonical[key] == key:
                            if active.get('kind') != 'copy' or active.get('temporary') != '.copy-' + str(index):
                                raise Refusal('Committed active copy mismatch')
                            try:
                                os.stat(active['temporary'], dir_fd=root_fd, follow_symlinks=False)
                            except FileNotFoundError:
                                pass
                            else:
                                raise Refusal('Committed copy retains an unexpected temporary inode')
                        else:
                            owner = canonical[key]
                            if active.get('kind') != 'hardlink' or active.get('owner') != owner or done.get(owner) != done[key]:
                                raise Refusal('Committed active hardlink mismatch')
                        journal.append('committed_active_reconciled', path=key, inode=done[key])
                        atomic_json(root_fd, 'active.json', None, replace=True)
                    continue
                if key in done:
                    raise Refusal('Previously committed destination file missing')
                if canonical[key] != key:
                    owner = canonical[key]
                    if owner not in done:
                        raise Refusal('Hardlink canonical owner is not committed')
                    operation = {'path': key, 'kind': 'hardlink', 'inode': done[owner], 'owner': owner}
                    if active is not None and active != operation:
                        raise Refusal('Hardlink active-state mismatch')
                    atomic_json(root_fd, 'active.json', operation, replace=True)
                    with parent_and_name(sessions_fd, owner) as (opfd, oname):
                        if inode_identity(os.stat(oname, dir_fd=opfd, follow_symlinks=False)) != done[owner]:
                            raise Refusal('Hardlink owner replaced')
                        os.link(oname, name, src_dir_fd=opfd, dst_dir_fd=pfd, follow_symlinks=False)
                    os.fsync(pfd)
                    if fail == 'after_hardlink':
                        raise InjectedCrash(fail)
                    journal.append('file_done', path=key, inode=done[owner])
                    done[key] = done[owner]
                    atomic_json(root_fd, 'active.json', None, replace=True)
                    continue
                temporary = '.copy-' + str(index)
                if active is None:
                    operation = {'path': key, 'kind': 'copy', 'temporary': temporary, 'inode': None}
                    atomic_json(root_fd, 'active.json', operation, replace=True)
                    fd = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600, dir_fd=root_fd)
                    try:
                        verify_inherited_attributes(fd, row)
                        os.fchown(fd, uid, storage_gid)
                        os.fsync(fd)
                        os.fsync(root_fd)
                        operation['inode'] = inode_identity(os.fstat(fd))
                        atomic_json(root_fd, 'active.json', operation, replace=True)
                    except BaseException:
                        os.close(fd)
                        raise
                else:
                    operation = active
                    if operation.get('kind') != 'copy' or operation.get('temporary') != temporary or operation.get('inode') is None:
                        raise Refusal('Unjournaled temporary creation preserved for manual review')
                    # A prior completed copy may already have restored mode0444.
                    # Open read-only first, authenticate the private temp inode,
                    # then temporarily restore owner-write before recopying it.
                    readonly = os.open(temporary, os.O_RDONLY | O_NOFOLLOW | O_CLOEXEC | os.O_NONBLOCK, dir_fd=root_fd)
                    try:
                        rs = os.fstat(readonly)
                        if not stat.S_ISREG(rs.st_mode) or rs.st_nlink != 1 or rs.st_uid != uid or rs.st_gid != storage_gid or inode_identity(rs) != operation['inode']:
                            raise Refusal('Temporary inode collision')
                        verify_inherited_attributes(readonly, row)
                        os.fchmod(readonly, 0o600)
                    finally:
                        os.close(readonly)
                    fd = os.open(temporary, os.O_RDWR | O_NOFOLLOW | O_CLOEXEC | os.O_NONBLOCK, dir_fd=root_fd)
                try:
                    s = os.fstat(fd)
                    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or s.st_uid != uid or s.st_gid != storage_gid or inode_identity(s) != operation['inode']:
                        raise Refusal('Temporary inode collision')
                    source_file = source_file_checked(source_fd, key, manifest)
                    try:
                        os.ftruncate(fd, 0)
                        os.lseek(fd, 0, os.SEEK_SET)
                        while True:
                            block = os.read(source_file, 1024 * 1024)
                            if not block:
                                break
                            write_all(fd, block)
                            if fail == 'during_copy':
                                os.fsync(fd)
                                raise InjectedCrash(fail)
                        if identity(os.fstat(source_file)) != manifest['source_identities'][key] or hash_fd(fd) != row['sha256']:
                            raise Refusal('Source drift or copy digest mismatch')
                        apply_metadata(fd, row, source_file, storage_gid)
                    finally:
                        os.close(source_file)
                    check_leaf(fd, row, storage_gid, acl_reader)
                    if fail == 'after_copy':
                        raise InjectedCrash(fail)
                    install_regular_noclobber(root_fd, temporary, pfd, name, fail=fail)
                    os.fsync(pfd)
                    os.fsync(root_fd)
                    if fail == 'after_install':
                        raise InjectedCrash(fail)
                    journal.append('file_done', path=key, inode=operation['inode'])
                    done[key] = operation['inode']
                    atomic_json(root_fd, 'active.json', None, replace=True)
                    copied += 1
                finally:
                    os.close(fd)
        # Restore directories bottom-up after the final child creation. A crash
        # here is replayable; no completed payload file is recopied.
        for key, row in sorted(directories, key=lambda item: item[0].count('/'), reverse=True):
            with child_directory(sessions_fd, key) as fd, child_directory(source_fd, key) as source_dir:
                if inode_identity(os.fstat(fd)) != mkdir_events[key] or identity(os.fstat(source_dir)) != manifest['source_identities'][key]:
                    raise Refusal('Directory identity drift before metadata restoration')
                apply_metadata(fd, row, source_dir, storage_gid)
        os.fsync(sessions_fd)
        if fail == 'before_verify':
            raise InjectedCrash(fail)
        after = inventory(source_fd, names, uid, source_gid, acl_reader)
        if after != manifest:
            raise Refusal('Source drift during staging')
        target = inventory(sessions_fd, names, uid, storage_gid, acl_reader)
        if semantic(target) != semantic(manifest, storage_gid) or sorted(os.listdir(sessions_fd)) != sorted(names):
            raise Refusal('Full destination byte/metadata/hardlink graph mismatch')
        if read_json(root_fd, 'active.json') is not None:
            raise Refusal('Unexpected unfinished copy operation')
        verify_namespace(source, source_fd)
        verify_namespace(destination, root_fd)
        proof = {'schema': 1, 'status': 'staged_verified_no_publication',
                 'source_manifest_sha256': digest(raw_json(manifest)),
                 'destination_semantic_sha256': digest(raw_json(semantic(target))),
                 'source_semantic_sha256': digest(raw_json(semantic(manifest))),
                 'entry_count': len(manifest['rows']),
                 'file_count': sum(r['kind'] == 'regular_file' for r in manifest['rows'].values()),
                 'hardlink_class_count': len(manifest['hardlink_classes']),
                 'source_namespace_unchanged': True, 'source_mutation_operations': 0,
                 'original_gid_retained_in_source_manifest': source_gid,
                 'destination_storage_gid': storage_gid,
                 'file_install_protocol': 'atomic no-clobber hardlink followed by authenticated temporary alias unlink',
                 'renameat2_not_used_for_staging': 'Actual GPFS probes returned EINVAL for NOREPLACE absent-target move and EXCHANGE; no fallback success is fabricated.',
                 'native_acl_reference': 'https://www.ibm.com/docs/en/storage-scale/6.0.1?topic=reference-mmgetacl-command',
                 'gid_policy': 'Storage accounting group only; UID, permission bits, mtime, content, xattrs, native basic ACL and hardlink classes unchanged.',
                 'authorization': authorization, 'script_sha256': expected_intent['script_sha256']}
        journal.append('copy_verified', source_manifest_sha256=proof['source_manifest_sha256'])
        atomic_json(root_fd, 'copy-proof.json', proof)
        return {**proof, 'replay': False, 'copy_operations': copied}
    finally:
        if journal is not None:
            journal.close()
        for fd in (sessions_fd, lock_fd, root_fd, parent_fd, source_fd):
            if fd is not None:
                os.close(fd)


def verify_namespace(path, expected_fd):
    other = open_absolute_directory(path)
    try:
        if inode_identity(os.fstat(other)) != inode_identity(os.fstat(expected_fd)):
            raise Refusal('Absolute namespace root changed during staging')
    finally:
        os.close(other)


def exchange(fd, a, b):
    if sys.platform != 'linux':
        raise Refusal('Exchange probe requires Linux renameat2')
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, 'renameat2', None)
    if function is None:
        raise Refusal('renameat2 unavailable')
    if function(fd, os.fsencode(a), fd, os.fsencode(b), 2) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def probe_exchange(parent, uid, gid):
    """Exclusive synthetic names only. Preserve the probe tree after roundtrip.

    No real session path is passed to renameat2 or removed. No recursive cleanup.
    A failed/unsupported exchange leaves all probe evidence for review.
    """
    pfd = open_absolute_directory(parent)
    probe_fd = None
    try:
        name = '.aco-synthetic-exchange-probe-' + uuid.uuid4().hex
        os.mkdir(name, 0o700, dir_fd=pfd)
        probe_fd = os.open(name, os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC, dir_fd=pfd)
        os.fchown(probe_fd, uid, gid)
        os.fchmod(probe_fd, 0o700)
        os.mkdir('directory', 0o700, dir_fd=probe_fd)
        os.symlink('directory', 'symlink', dir_fd=probe_fd)
        before_dir = inode_identity(os.stat('directory', dir_fd=probe_fd, follow_symlinks=False))
        before_link = inode_identity(os.stat('symlink', dir_fd=probe_fd, follow_symlinks=False))
        atomic_json(probe_fd, 'intent.json', {'kind': 'synthetic_directory_symlink_exchange_roundtrip', 'directory_inode': before_dir, 'link_inode': before_link, 'real_session_operations': 0})
        exchange(probe_fd, 'directory', 'symlink')
        os.fsync(probe_fd)
        if not stat.S_ISLNK(os.stat('directory', dir_fd=probe_fd, follow_symlinks=False).st_mode) or inode_identity(os.stat('symlink', dir_fd=probe_fd, follow_symlinks=False)) != before_dir:
            raise Refusal('First synthetic exchange verification failed')
        exchange(probe_fd, 'directory', 'symlink')
        os.fsync(probe_fd)
        if inode_identity(os.stat('directory', dir_fd=probe_fd, follow_symlinks=False)) != before_dir or inode_identity(os.stat('symlink', dir_fd=probe_fd, follow_symlinks=False)) != before_link or os.readlink('symlink', dir_fd=probe_fd) != 'directory':
            raise Refusal('Synthetic exchange roundtrip identity mismatch')
        result = {'status': 'synthetic_exchange_roundtrip_verified', 'path': str(Path(parent) / name), 'real_session_operations': 0, 'probe_tree_retained': True}
        atomic_json(probe_fd, 'receipt.json', result)
        os.fsync(pfd)
        return result
    finally:
        if probe_fd is not None:
            os.close(probe_fd)
        os.close(pfd)


def live_preflight(path, expected_sha):
    raw = Path(path).read_bytes()
    if digest(raw) != expected_sha:
        raise Refusal('Authorization/preflight digest mismatch')
    proof = json.loads(raw)
    if proof.get('schema') != 1 or proof.get('generation_only') is not True or proof.get('no_active_owned_jobs') is not True or proof.get('no_known_source_writers') is not True:
        raise Refusal('Explicit generation-only/known-writer preflight is required')
    if proof.get('terminal_jobs') != {name: {'job_id': JOBS[name], 'state': 'COMPLETED', 'exit_code': '0:0'} for name in NAMES}:
        raise Refusal('Exact four closed job preflight mismatch')
    if proof.get('source') != str(SOURCE) or proof.get('destination') != str(DESTINATION):
        raise Refusal('Exact source/destination preflight mismatch')
    observed = proof.get('observed_at_unix')
    if type(observed) not in (int, float) or not 0 <= time.time() - observed <= 3600:
        raise Refusal('Fresh preflight (within one hour) required before staging')
    pins = proof.get('historical_receipt_sha256')
    if not isinstance(pins, dict) or not pins or not all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in pins.values()):
        raise Refusal('Historical receipt hash pins required')
    roles = proof.get('role_sha256')
    expected_roles = {'job_id.json', 'snapshot.json', 'ready.json', 'build.json'}
    if not isinstance(roles, dict) or set(roles) != set(NAMES) or not all(
            isinstance(value, dict) and set(value) == expected_roles and
            all(isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h) for h in value.values()) for value in roles.values()):
        raise Refusal('Exact four historical transport role hash maps required')
    # Fresh queue/process observations may be renewed after a crash. The intent
    # pins stable historical authorities, rather than making a fresh observation
    # hash collide with an otherwise identical recoverable transaction.
    return {'historical_receipt_sha256': pins, 'terminal_job_ids': list(JOBS.values()),
            'role_sha256': roles, 'generation_only': True}


def metadata_probe():
    """Eight exact metadata roles plus the project parent; no payload output."""
    source_fd = open_absolute_directory(SOURCE)
    parent_fd = open_absolute_directory(DESTINATION.parent)
    rows = {}
    try:
        for name in NAMES:
            for suffix in ('', '/job_id.json'):
                key = name + suffix
                if suffix:
                    fd = open_file(source_fd, key)
                else:
                    with child_directory(source_fd, key) as dfd:
                        fd = os.dup(dfd)
                try:
                    s = os.fstat(fd)
                    if s.st_uid != UID or s.st_gid != SOURCE_GID:
                        raise Refusal('Probe source owner mismatch')
                    rows[key] = {'uid': s.st_uid, 'gid': s.st_gid,
                                 'mode': oct(stat.S_IMODE(s.st_mode)),
                                 'xattrs': xattr_metadata(fd), 'acl': gpfs_acl(fd, s)}
                finally:
                    os.close(fd)
        parent_s = os.fstat(parent_fd)
        rows['project_parent'] = {'uid': parent_s.st_uid, 'gid': parent_s.st_gid,
                                  'mode': oct(stat.S_IMODE(parent_s.st_mode)),
                                  'acl': gpfs_acl(parent_fd, parent_s)}
        return {'kind': 'read_only_exact_role_metadata_probe', 'rows': rows,
                'source_mutation_operations': 0, 'payload_bytes_emitted': 0,
                'coverage': 'Eight exact source roles and project parent only; not whole-tree ACL/xattr qualification.'}
    finally:
        os.close(parent_fd)
        os.close(source_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    stage = sub.add_parser('stage')
    stage.add_argument('--preflight', type=Path, required=True)
    stage.add_argument('--preflight-sha256', required=True)
    stage.add_argument('--acknowledge-staging-only', choices=['four-closed-generation-sessions-no-source-mutation'], required=True)
    probe = sub.add_parser('probe-exchange')
    probe.add_argument('--acknowledge-synthetic-only', choices=['no-real-session-exchange'], required=True)
    sub.add_parser('metadata-probe')
    args = parser.parse_args()
    if sys.platform != 'linux' or os.getuid() != UID or STORAGE_GID not in os.getgroups():
        raise Refusal('Live CLI requires Linux, exact owner UID and approved project group membership')
    if args.command == 'stage':
        authorization = live_preflight(args.preflight, args.preflight_sha256)
        result = stage_copy(SOURCE, DESTINATION, NAMES, UID, SOURCE_GID, STORAGE_GID, authorization=authorization)
    elif args.command == 'probe-exchange':
        result = probe_exchange(DESTINATION.parent, UID, STORAGE_GID)
    else:
        result = metadata_probe()
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
