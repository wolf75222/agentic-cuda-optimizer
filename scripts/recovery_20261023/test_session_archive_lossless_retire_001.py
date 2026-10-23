"""Synthetic only. Removal touches redundant copies in private temp fixtures."""
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('retirement', Path(__file__).with_name('session_archive_lossless_retire_001.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
fspec = importlib.util.spec_from_file_location('publication_fixtures', Path(__file__).with_name('test_session_archive_lossless_publish_003.py'))
fixtures = importlib.util.module_from_spec(fspec)
fspec.loader.exec_module(fixtures)


class RetirementTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PublicationTests(methodName='test_all_published_retains_originals_and_global_links_then_replays')
        self.fixture.setUp()
        self.fixture.publish()
        self.source, self.destination, self.names = self.fixture.source, self.fixture.destination, self.fixture.names
        self.original = self.fixture.fixture.old
        self.pub = json.loads((self.destination / 'publication-proof.json').read_bytes())
        self.pub_sha = r.m.digest(r.m.raw_json(self.pub))
        self.rollback = self.source / self.pub['rollback_parent']

    def tearDown(self):
        self.fixture.tearDown()

    def retire(self, **kwargs):
        return r.retire(self.source, self.destination, self.names, self.fixture.fixture.uid, self.fixture.fixture.gid, self.fixture.fixture.gid,
                        publication_proof_sha256=self.pub_sha,
                        historical_access_receipt_sha256='a'*64,
                        acl_reader=fixtures.fixtures.synthetic_acl, **kwargs)

    def assert_copy(self):
        copied = self.fixture.fixture.inventory(self.destination / 'sessions')
        self.assertEqual(r.m.semantic(copied), r.m.semantic(self.original))
        for name in self.names:
            self.assertTrue((self.source / name).is_symlink())

    def test_only_redundant_originals_removed_copy_and_symlinks_preserved(self):
        unrelated = self.source / 'unrelated-other-task'
        unrelated.mkdir()
        (unrelated / 'keep').write_bytes(b'keep')
        result = self.retire()
        self.assertEqual(result['retired_entries_observed'], len(self.original['rows']))
        self.assertEqual(result['retired_files_observed'], 4)
        self.assertEqual(set(os.listdir(self.rollback)), {'.lock'})
        self.assertEqual((unrelated / 'keep').read_bytes(), b'keep')
        self.assert_copy()
        repeated = self.retire()
        self.assertTrue(repeated['replay'])

    def test_resume_after_unlink_before_progress_keeps_global_hardlink_counts(self):
        with self.assertRaises(r.m.InjectedCrash):
            self.retire(fail='after_remove_before_progress')
        result = self.retire()
        self.assertEqual(result['recovered_absent_entries'], 1)
        self.assert_copy()

    def test_resume_after_progress_before_active_clear(self):
        with self.assertRaises(r.m.InjectedCrash):
            self.retire(fail='after_progress_before_clear')
        self.retire()
        self.assert_copy()

    def test_resume_after_directory_rmdir_before_progress(self):
        append = r.Progress.append
        once = [False]
        def fail_on_dir(progress, action, **fields):
            if fields.get('kind') == 'directory' and not once[0]:
                once[0] = True
                raise r.m.InjectedCrash('directory removed before progress')
            append(progress, action, **fields)
        with patch.object(r.Progress, 'append', new=fail_on_dir):
            with self.assertRaises(r.m.InjectedCrash):
                self.retire()
        self.assertEqual(self.retire()['recovered_absent_entries'], 1)
        self.assert_copy()

    def test_unknown_extra_original_stops_before_any_remove(self):
        (self.rollback / self.names[0] / 'unknown').write_bytes(b'preserve')
        with self.assertRaisesRegex(r.m.Refusal, 'Original byte'):
            self.retire()
        self.assertTrue((self.rollback / self.names[0] / 'unknown').is_file())
        self.assertTrue((self.rollback / self.names[0] / 'job_id.json').is_file())

    def test_drifted_project_copy_stops_before_any_remove(self):
        (self.destination / 'sessions' / self.names[0] / 'job_id.json').write_bytes(b'drift')
        with self.assertRaisesRegex(r.m.Refusal, 'Project copy drifted'):
            self.retire()
        self.assertTrue((self.rollback / self.names[0] / 'job_id.json').is_file())

    def test_external_original_hardlink_refused_before_any_remove(self):
        os.link(self.rollback / self.names[0] / 'requests/blob.bin', self.source / 'unexpected-alias')
        with self.assertRaisesRegex(r.m.Refusal, 'external original hardlink'):
            self.retire()
        self.assertTrue((self.rollback / self.names[0] / 'job_id.json').is_file())

    def test_unknown_missing_file_on_resume_refused(self):
        with self.assertRaises(r.m.InjectedCrash):
            self.retire(fail='after_progress_before_clear')
        (self.rollback / self.names[1] / 'job_id.json').unlink()
        with self.assertRaisesRegex(r.m.Refusal, 'Unexplained missing'):
            self.retire()
        self.assert_copy()

    def test_original_changed_on_resume_refuses_before_its_unlink(self):
        with self.assertRaises(r.m.InjectedCrash):
            self.retire(fail='after_progress_before_clear')
        remaining = self.rollback / self.names[1] / 'job_id.json'
        remaining.write_bytes(b'changed')
        with self.assertRaisesRegex(r.m.Refusal, 'size/mtime'):
            self.retire()
        self.assertEqual(remaining.read_bytes(), b'changed')
        self.assert_copy()

    def test_pretransfer_receipt_cannot_authorize_retirement(self):
        raw = r.m.raw_json({'kind': 'read_only_four_closed_generation_historical_access_verification'})
        with self.assertRaisesRegex(r.m.Refusal, 'explicitly bind'):
            r.historical_receipt(raw, r.m.digest(raw), self.names, publication_sha=self.pub_sha)


if __name__ == '__main__':
    unittest.main(verbosity=2)
