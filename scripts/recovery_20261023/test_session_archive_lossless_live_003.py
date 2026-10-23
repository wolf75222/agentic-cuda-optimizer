"""Synthetic only: no network, campaign files, Slurm, GPFS or private values."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('live_staging', Path(__file__).with_name('session_archive_lossless_live_003.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def synthetic_acl(fd, s):
    return {'access': [oct(stat.S_IMODE(s.st_mode))], 'default': [], 'inspection': 'synthetic-only'}


class StagingTests(unittest.TestCase):
    def setUp(self):
        # CPython on this Mac does not expose Linux's FD-xattr API. The live CLI
        # is Linux-only; synthetic tests do not pretend to qualify GPFS attrs.
        if not hasattr(m.os, 'listxattr'):
            self.xattr_patch = patch.object(m.os, 'listxattr', return_value=[], create=True)
            self.xattr_patch.start()
            self.addCleanup(self.xattr_patch.stop)
        self.tmp = tempfile.TemporaryDirectory(prefix='opaque-stage-live-test-')
        # macOS's tempfile root uses /var -> /private/var. Resolve only our new
        # synthetic fixture; the live traversal deliberately rejects symlinks.
        self.base = Path(self.tmp.name).resolve()
        self.source = self.base / 'scratch' / 'sessions'
        self.source.mkdir(parents=True)
        self.project = self.base / 'project'
        self.project.mkdir()
        self.destination = self.project / 'private-staging'
        self.names = ('generation-a', 'generation-b')
        for name in self.names:
            directory = self.source / name
            directory.mkdir()
            (directory / 'requests').mkdir()
            (directory / 'job_id.json').write_bytes(b'opaque transport receipt fixture\n')
        self.original = self.source / self.names[0] / 'requests' / 'blob.bin'
        self.original.write_bytes(bytes(range(256)) * 32)
        os.chmod(self.original, 0o444)
        os.link(self.original, self.source / self.names[1] / 'requests' / 'blob.bin')
        self.uid = os.getuid()
        self.gid = os.getgid()
        self.old = self.inventory(self.source)

    def tearDown(self):
        self.tmp.cleanup()

    def inventory(self, root):
        fd = m.open_absolute_directory(root)
        try:
            return m.inventory(fd, self.names, self.uid, self.gid, synthetic_acl)
        finally:
            os.close(fd)

    def stage(self, **kwargs):
        return m.stage_copy(self.source, self.destination, self.names, self.uid, self.gid, self.gid,
                            acl_reader=synthetic_acl, authorization={'synthetic': True}, **kwargs)

    def test_full_opaque_copy_cross_session_hardlink_and_replay(self):
        result = self.stage()
        self.assertEqual(result['status'], 'staged_verified_no_publication')
        self.assertFalse(result['replay'])
        self.assertEqual(result['copy_operations'], 3)
        copied = self.inventory(self.destination / 'sessions')
        self.assertEqual(m.semantic(copied), m.semantic(self.old))
        self.assertEqual(self.inventory(self.source), self.old)
        self.assertFalse((self.source / self.names[0]).is_symlink())
        a = self.destination / 'sessions' / self.names[0] / 'requests' / 'blob.bin'
        b = self.destination / 'sessions' / self.names[1] / 'requests' / 'blob.bin'
        self.assertEqual(a.stat().st_ino, b.stat().st_ino)
        self.assertEqual(a.stat().st_nlink, 2)
        before = {p: p.read_bytes() for p in self.destination.iterdir() if p.is_file()}
        repeated = self.stage()
        self.assertTrue(repeated['replay'])
        self.assertEqual(repeated['copy_operations'], 0)
        self.assertEqual(before, {p: p.read_bytes() for p in self.destination.iterdir() if p.is_file()})

    def test_resume_after_manifest(self):
        with self.assertRaises(m.InjectedCrash):
            self.stage(fail='after_manifest')
        self.assertTrue(self.stage()['status'].startswith('staged_verified'))
        self.assertEqual(self.inventory(self.source), self.old)

    def test_resume_partial_copy_uses_journaled_temp_only(self):
        with self.assertRaises(m.InjectedCrash):
            self.stage(fail='during_copy')
        active = json.loads((self.destination / 'active.json').read_bytes())
        self.assertIsNotNone(active['inode'])
        self.assertTrue((self.destination / active['temporary']).is_file())
        self.stage()
        self.assertEqual(self.inventory(self.source), self.old)

    def test_resume_after_readonly_temp_completed(self):
        with self.assertRaises(m.InjectedCrash):
            self.stage(fail='after_copy')
        self.stage()
        self.assertEqual(self.inventory(self.source), self.old)

    def test_resume_after_atomic_install_before_journal(self):
        with self.assertRaises(m.InjectedCrash):
            self.stage(fail='after_install')
        self.stage()
        self.assertEqual(self.inventory(self.source), self.old)

    def test_resume_after_file_hardlink_before_temp_unlink(self):
        with self.assertRaises(m.InjectedCrash):
            self.stage(fail='after_file_link_before_temp_unlink')
        active = json.loads((self.destination / 'active.json').read_bytes())
        temp = self.destination / active['temporary']
        target = self.destination / 'sessions' / active['path']
        self.assertEqual(temp.stat().st_ino, target.stat().st_ino)
        self.assertEqual(temp.stat().st_nlink, 2)
        self.stage()
        self.assertFalse(temp.exists())
        self.assertEqual(self.inventory(self.source), self.old)

    def test_extra_external_alias_of_active_pair_refused_preserving_all(self):
        with self.assertRaises(m.InjectedCrash):
            self.stage(fail='after_file_link_before_temp_unlink')
        active = json.loads((self.destination / 'active.json').read_bytes())
        temp = self.destination / active['temporary']
        os.link(temp, self.base / 'external-unexplained-target-alias')
        with self.assertRaisesRegex(m.Refusal, 'exact journaled two-link'):
            self.stage()
        self.assertTrue(temp.exists())

    def test_replaced_temporary_of_active_pair_refused_preserving_all(self):
        with self.assertRaises(m.InjectedCrash):
            self.stage(fail='after_file_link_before_temp_unlink')
        active = json.loads((self.destination / 'active.json').read_bytes())
        temp = self.destination / active['temporary']
        temp.unlink()
        temp.write_bytes(b'unknown inode; preserve')
        with self.assertRaisesRegex(m.Refusal, 'exact journaled two-link'):
            self.stage()
        self.assertEqual(temp.read_bytes(), b'unknown inode; preserve')

    def test_resume_after_cross_session_hardlink_before_journal(self):
        with self.assertRaises(m.InjectedCrash):
            self.stage(fail='after_hardlink')
        self.stage()
        self.assertEqual(m.semantic(self.inventory(self.destination / 'sessions')), m.semantic(self.old))

    def crash_after_done(self, wanted):
        append = m.Journal.append
        def failing(journal, action, **fields):
            append(journal, action, **fields)
            if action == 'file_done' and fields['path'] == wanted:
                raise m.InjectedCrash('file_done durable before active clear')
        with patch.object(m.Journal, 'append', new=failing):
            with self.assertRaises(m.InjectedCrash):
                self.stage()
        self.assertEqual(json.loads((self.destination / 'active.json').read_bytes())['path'], wanted)
        self.stage()
        self.assertIsNone(json.loads((self.destination / 'active.json').read_bytes()))
        self.assertEqual(self.inventory(self.source), self.old)
        self.assertTrue(self.stage()['replay'])

    def test_resume_file_done_before_active_clear_first_copy(self):
        self.crash_after_done(self.names[0] + '/job_id.json')

    def test_resume_file_done_before_active_clear_last_hardlink(self):
        self.crash_after_done(self.names[1] + '/requests/blob.bin')

    def test_resume_file_done_before_active_clear_intermediate_copy(self):
        self.crash_after_done(self.names[0] + '/requests/blob.bin')

    def test_committed_active_incompatible_kind_refused(self):
        append = m.Journal.append
        def failing(journal, action, **fields):
            append(journal, action, **fields)
            if action == 'file_done':
                raise m.InjectedCrash('stop')
        with patch.object(m.Journal, 'append', new=failing):
            with self.assertRaises(m.InjectedCrash):
                self.stage()
        active = json.loads((self.destination / 'active.json').read_bytes())
        active['kind'] = 'hardlink'
        (self.destination / 'active.json').write_bytes(m.raw_json(active))
        with self.assertRaisesRegex(m.Refusal, 'Committed active copy mismatch'):
            self.stage()
        self.assertEqual(self.inventory(self.source), self.old)

    def test_resume_after_directory_metadata_before_verification(self):
        with self.assertRaises(m.InjectedCrash):
            self.stage(fail='before_verify')
        self.stage()
        self.assertEqual(self.inventory(self.source), self.old)

    def test_source_drift_refuses_without_rewrite(self):
        with self.assertRaises(m.InjectedCrash):
            self.stage(fail='after_manifest')
        (self.source / self.names[0] / 'new').write_bytes(b'new')
        with self.assertRaisesRegex(m.Refusal, 'Source drift'):
            self.stage()
        self.assertFalse((self.source / self.names[0]).is_symlink())

    def test_target_content_drift_refuses_replay(self):
        self.stage()
        target = self.destination / 'sessions' / self.names[0] / 'job_id.json'
        target.write_bytes(b'drift')
        with self.assertRaisesRegex(m.Refusal, 'Committed destination drift'):
            self.stage()
        self.assertEqual(self.inventory(self.source), self.old)

    def test_unknown_target_collision_is_preserved(self):
        with self.assertRaises(m.InjectedCrash):
            self.stage(fail='after_manifest')
        (self.destination / 'unrecognized').write_bytes(b'do not delete')
        with self.assertRaisesRegex(m.Refusal, 'Unrecognized'):
            self.stage()
        self.assertEqual((self.destination / 'unrecognized').read_bytes(), b'do not delete')

    def test_preexisting_root_without_intent_refuses(self):
        self.destination.mkdir(mode=0o700)
        with self.assertRaises((m.Refusal, FileNotFoundError)):
            self.stage()
        self.assertEqual(self.inventory(self.source), self.old)

    def test_unjournaled_partial_directory_is_preserved_and_refused(self):
        with self.assertRaises(m.InjectedCrash):
            self.stage(fail='after_manifest')
        (self.destination / 'sessions').mkdir()
        with self.assertRaisesRegex(m.Refusal, 'Unjournaled'):
            self.stage()
        self.assertTrue((self.destination / 'sessions').is_dir())

    def test_external_source_hardlink_refuses(self):
        os.link(self.original, self.base / 'unselected-alias')
        with self.assertRaisesRegex(m.Refusal, 'external hardlink'):
            self.stage()

    def test_symlink_source_file_and_directory_refuse(self):
        for directory in (False, True):
            link = self.source / self.names[0] / 'unexpected'
            link.symlink_to(self.base, target_is_directory=directory)
            with self.assertRaisesRegex(m.Refusal, 'Symlink or special'):
                self.stage()
            link.unlink()

    def test_parent_symlink_is_not_followed(self):
        alias = self.base / 'alias'
        alias.symlink_to(self.project, target_is_directory=True)
        with self.assertRaises(OSError):
            m.stage_copy(self.source, alias / 'bad', self.names, self.uid, self.gid, self.gid, acl_reader=synthetic_acl)
        self.assertFalse((self.project / 'bad').exists())

    def test_native_acl_parser_refuses_extended_default_nfs_and_mode_mismatch(self):
        self.assertEqual(len(m.parse_native_acl(b'#owner:opaque\nuser::rwxc\ngroup::r-x-\nother::---\n', 0o750)), 3)
        for raw, mode, default in ((b'#NFSv4 ACL\n', 0o700, False),
                                   (b'user::rwx\nuser:123:rwx\ngroup::---\nother::---\n', 0o700, False),
                                   (b'user::rwx\n', 0o700, True),
                                   (b'user::rwx\ngroup::rwx\nother::---\n', 0o700, False)):
            with self.assertRaises(m.Refusal):
                m.parse_native_acl(raw, mode, default=default)

    def test_unsupported_xattr_refuses(self):
        with patch.object(m.os, 'listxattr', return_value=['security.synthetic']):
            with self.assertRaisesRegex(m.Refusal, 'Unsupported extended attribute'):
                self.stage()

    def source_inodes(self):
        return {p.stat().st_ino for name in self.names for p in [self.source / name, *(self.source / name).rglob('*')]}

    def test_security_selinux_inherited_exact_never_set_or_relabelled(self):
        with patch.object(m.os, 'listxattr', return_value=['security.selinux'], create=True), \
             patch.object(m.os, 'getxattr', return_value=b'synthetic-security-label', create=True), \
             patch.object(m.os, 'setxattr', side_effect=AssertionError('Never set SELinux labels'), create=True):
            self.stage()
            self.assertTrue(self.stage()['replay'])

    def test_security_selinux_absent_or_different_target_refused(self):
        source_inodes = self.source_inodes()
        for absent in (True, False):
            destination = self.project / ('label-absent' if absent else 'label-different')
            def names(fd):
                return ['security.selinux'] if not absent or os.fstat(fd).st_ino in source_inodes else []
            def value(fd, name):
                return b'source-label' if os.fstat(fd).st_ino in source_inodes else b'other-label'
            with patch.object(m.os, 'listxattr', side_effect=names, create=True), \
                 patch.object(m.os, 'getxattr', side_effect=value, create=True), \
                 patch.object(m.os, 'setxattr', side_effect=AssertionError('No label writes'), create=True):
                with self.assertRaisesRegex(m.Refusal, 'Inherited security label'):
                    m.stage_copy(self.source, destination, self.names, self.uid, self.gid, self.gid, acl_reader=synthetic_acl)

    def test_other_security_and_trusted_attributes_remain_refused(self):
        for attribute in ('security.other', 'trusted.synthetic', 'system.posix_acl_access'):
            with patch.object(m.os, 'listxattr', return_value=[attribute], create=True):
                with self.assertRaisesRegex(m.Refusal, 'Unsupported extended attribute'):
                    self.stage()

    def test_user_attributes_are_copied_and_verified_opaqely(self):
        attrs = {inode: {'user.synthetic': b'opaque-synthetic-xattr'} for inode in self.source_inodes()}
        def names(fd):
            return list(attrs.get(os.fstat(fd).st_ino, {}))
        def read(fd, name):
            return attrs[os.fstat(fd).st_ino][name]
        def write(fd, name, value):
            self.assertTrue(name.startswith('user.'))
            attrs.setdefault(os.fstat(fd).st_ino, {})[name] = value
        with patch.object(m.os, 'listxattr', side_effect=names, create=True), \
             patch.object(m.os, 'getxattr', side_effect=read, create=True), \
             patch.object(m.os, 'setxattr', side_effect=write, create=True):
            self.stage()
            self.assertTrue(self.stage()['replay'])

    def test_foreign_private_gid_is_refused(self):
        self.stage()
        with self.assertRaisesRegex(m.Refusal, 'charged to the approved group'):
            m.stage_copy(self.source, self.destination, self.names, self.uid, self.gid, self.gid + 1, acl_reader=synthetic_acl)

    def test_wrong_intent_authorization_refuses(self):
        self.stage()
        with self.assertRaisesRegex(m.Refusal, 'intent collision'):
            m.stage_copy(self.source, self.destination, self.names, self.uid, self.gid, self.gid, acl_reader=synthetic_acl, authorization={'different': True})

    def test_historical_role_drift_refuses_before_destination_creation(self):
        with self.assertRaisesRegex(m.Refusal, 'Pinned historical transport role'):
            m.stage_copy(self.source, self.destination, self.names, self.uid, self.gid, self.gid,
                         acl_reader=synthetic_acl, authorization={'role_sha256': {self.names[0]: {'job_id.json': '0'*64}}})
        self.assertFalse(self.destination.exists())

    def test_torn_journal_refused_without_silent_truncation(self):
        with self.assertRaises(m.InjectedCrash):
            self.stage(fail='after_manifest')
        journal = self.destination / 'journal.jsonl'
        with journal.open('ab') as stream:
            stream.write(b'{"torn":')
        raw = journal.read_bytes()
        with self.assertRaisesRegex(m.Refusal, 'Torn journal'):
            self.stage()
        self.assertEqual(raw, journal.read_bytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
