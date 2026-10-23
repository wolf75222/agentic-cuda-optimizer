"""Synthetic private-parent publication tests. No remote or GPU actions."""
import importlib.util
import json
import os
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('pub003', Path(__file__).with_name('session_archive_lossless_publish_003.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
fspec = importlib.util.spec_from_file_location('fixtures003', Path(__file__).with_name('test_session_archive_lossless_live_003.py'))
fixtures = importlib.util.module_from_spec(fspec)
fspec.loader.exec_module(fixtures)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.StagingTests(methodName='test_full_opaque_copy_cross_session_hardlink_and_replay')
        self.fixture.setUp()
        self.fixture.stage()
        self.source, self.destination, self.names = self.fixture.source, self.fixture.destination, self.fixture.names
        self.manifest = json.loads((self.destination / 'source-manifest.json').read_bytes())
        self.proof = json.loads((self.destination / 'copy-proof.json').read_bytes())

    def tearDown(self):
        self.fixture.tearDown()
        self.fixture.doCleanups()

    def publish(self, **kwargs):
        return p.publish(self.source, self.destination, self.names, self.fixture.uid, self.fixture.gid, self.fixture.gid,
                         copy_proof_sha256=p.m.digest(p.m.raw_json(self.proof)),
                         manifest_sha256=p.m.digest(p.m.raw_json(self.manifest)),
                         native_failure_receipt_sha256='a'*64,
                         acl_reader=fixtures.synthetic_acl, **kwargs)

    def test_all_published_retains_originals_and_global_links_then_replays(self):
        result = self.publish()
        self.assertEqual((result['source_renames'], result['symlink_creations']), (2, 2))
        self.assertFalse(result['publication_atomic'])
        self.assertTrue(result['namespace_gap_possible'])
        parent = self.source / result['rollback_parent']
        self.assertEqual(parent.stat().st_mode & 0o7777, 0o700)
        for name in self.names:
            self.assertTrue((self.source / name).is_symlink())
            self.assertEqual(os.readlink(self.source / name), str(self.destination / 'sessions' / name))
            self.assertTrue((parent / name).is_dir())
        a, b = (parent / name / 'requests/blob.bin' for name in self.names)
        self.assertEqual(a.stat().st_ino, b.stat().st_ino)
        self.assertEqual(a.stat().st_nlink, 2)
        self.assertEqual(result['source_payload_unlinks'], 0)
        self.assertFalse(self.source.is_symlink())
        self.assertEqual(self.publish()['source_renames'], 0)

    def test_resume_after_source_rename_with_gap(self):
        with self.assertRaises(p.m.InjectedCrash):
            self.publish(fail='after_source_rename')
        self.assertFalse((self.source / self.names[0]).exists())
        self.assertTrue((self.source / self.names[1]).is_dir())
        result = self.publish()
        self.assertEqual((result['source_renames'], result['symlink_creations']), (1, 2))

    def test_resume_after_durable_link_inode_before_journal(self):
        with self.assertRaises(p.m.InjectedCrash):
            self.publish(fail='after_link_inode_durable')
        result = self.publish()
        self.assertEqual((result['source_renames'], result['symlink_creations']), (1, 1))

    def test_resume_after_first_session(self):
        with self.assertRaises(p.m.InjectedCrash):
            self.publish(fail='after_session')
        result = self.publish()
        self.assertEqual((result['source_renames'], result['symlink_creations']), (1, 1))

    def test_unrecorded_symlink_creation_refuses_preserving_all(self):
        with self.assertRaises(p.m.InjectedCrash):
            self.publish(fail='after_symlink_unjournaled')
        plan = json.loads((self.destination / 'publication-operations.json').read_bytes())
        with self.assertRaisesRegex(p.m.Refusal, 'Ambiguous'):
            self.publish()
        self.assertTrue((self.source / plan['rollback_parent'] / self.names[0]).is_dir())
        self.assertTrue((self.source / self.names[0]).is_symlink())

    def test_ordinary_symlink_error_restores_root_without_overwrite(self):
        once = [False]
        def fail_once(target, name, *, dir_fd):
            if not once[0]:
                once[0] = True
                raise PermissionError('synthetic link creation failure')
            os.symlink(target, name, dir_fd=dir_fd)
        with self.assertRaises(PermissionError):
            self.publish(symlink_fn=fail_once)
        self.assertTrue((self.source / self.names[0]).is_dir())
        self.assertFalse((self.source / self.names[0]).is_symlink())
        self.assertEqual(self.publish()['source_renames'], 2)

    def test_occupied_gap_is_preserved_and_rollback_refuses(self):
        def occupy(target, name, *, dir_fd):
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=dir_fd)
            os.write(fd, b'unknown occupant preserved')
            os.close(fd)
            raise FileExistsError('occupied')
        with self.assertRaisesRegex(p.m.Refusal, 'Ambiguous'):
            self.publish(symlink_fn=occupy)
        self.assertEqual((self.source / self.names[0]).read_bytes(), b'unknown occupant preserved')
        plan = json.loads((self.destination / 'publication-operations.json').read_bytes())
        self.assertTrue((self.source / plan['rollback_parent'] / self.names[0]).is_dir())

    def test_source_drift_refused_before_any_rename(self):
        (self.source / self.names[0] / 'new').write_bytes(b'drift')
        with self.assertRaisesRegex(p.m.Refusal, 'Original byte'):
            self.publish()
        self.assertTrue(all(not (self.source / name).is_symlink() for name in self.names))

    def test_target_drift_refused_before_any_rename(self):
        (self.destination / 'sessions' / self.names[0] / 'job_id.json').write_bytes(b'drift')
        with self.assertRaisesRegex(p.m.Refusal, 'Project copy drifted'):
            self.publish()
        self.assertTrue(all(not (self.source / name).is_symlink() for name in self.names))

    def test_changed_private_parent_mode_refused(self):
        with self.assertRaises(p.m.InjectedCrash):
            self.publish(fail='after_source_rename')
        plan = json.loads((self.destination / 'publication-operations.json').read_bytes())
        os.chmod(self.source / plan['rollback_parent'], 0o755)
        with self.assertRaisesRegex(p.m.Refusal, 'Private rollback parent'):
            self.publish()

    def test_changed_published_link_refuses_without_removing_backup(self):
        result = self.publish()
        name = self.names[0]
        (self.source / name).unlink()
        (self.source / name).symlink_to(self.destination / 'sessions' / self.names[1])
        with self.assertRaisesRegex(p.m.Refusal, 'Ambiguous'):
            self.publish()
        self.assertTrue((self.source / result['rollback_parent'] / name).is_dir())

    def test_unrecorded_private_parent_creation_is_preserved_and_refused(self):
        self.publish()
        plan = json.loads((self.destination / 'publication-operations.json').read_bytes())
        plan['rollback_parent_inode'] = None
        (self.destination / 'publication-operations.json').write_bytes(p.m.raw_json(plan))
        with self.assertRaisesRegex(p.m.Refusal, 'Unjournaled rollback parent'):
            self.publish()


if __name__ == '__main__':
    unittest.main(verbosity=2)
