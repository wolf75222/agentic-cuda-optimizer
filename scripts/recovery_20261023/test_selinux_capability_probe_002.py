"""Synthetic label API only; no real SELinux modification or remote action."""
import errno
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('label_probe', Path(__file__).with_name('selinux_capability_probe_002.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
INITIAL, DESIRED = b'opaque initial synthetic label', b'opaque distinct synthetic label'


class FakeLabels:
    def __init__(self, initial=INITIAL, deny=False, mismatch=False, deny_restore=False):
        self.initial, self.deny, self.mismatch, self.deny_restore = initial, deny, mismatch, deny_restore
        self.values = {}
        self.sets = []
    def get(self, fd):
        return self.values.get(os.fstat(fd).st_ino, self.initial)
    def replace(self, fd, value):
        self.sets.append((os.fstat(fd).st_ino, value))
        if self.deny or (self.deny_restore and value == INITIAL):
            raise PermissionError(errno.EPERM, 'synthetic denial')
        self.values[os.fstat(fd).st_ino] = b'opaque wrong label' if self.mismatch and value == DESIRED else value


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='label-capability-synthetic-')
        self.root = Path(self.tmp.name).resolve()
        self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.journal = m.Journal(self.fd, os.getuid(), os.getgid())
    def tearDown(self):
        self.journal.close()
        os.close(self.fd)
        self.tmp.cleanup()
    def run_probe(self, labels=None, kind='regular_file', desired=DESIRED):
        return m.exercise(self.fd, 'fixture', kind, desired, os.getuid(), os.getgid(), self.journal, labels or FakeLabels())
    def test_distinct_file_label_set_install_readback_restore(self):
        result = self.run_probe()
        self.assertTrue(result['qualified'])
        self.assertTrue(result['initial_label_restored_exactly'])
        self.assertTrue(result['transport_verified'])
        self.assertEqual((self.root / result['retained_name']).read_bytes(), m.FIXTURE)
        self.assertNotIn(DESIRED.decode(), json.dumps(result))
    def test_distinct_directory_label_set_rename_readback_restore(self):
        result = self.run_probe(kind='directory')
        self.assertTrue(result['qualified'])
        self.assertTrue(result['initial_label_restored_exactly'])
        self.assertTrue((self.root / result['retained_name']).is_dir())
    def test_permission_denial_is_retained_concrete_negative(self):
        result = self.run_probe(FakeLabels(deny=True))
        self.assertFalse(result['qualified'])
        self.assertEqual(result['errno'], errno.EPERM)
        self.assertTrue(result['initial_label_restored_exactly'])
        self.assertEqual(result['set_attempts'], 1)
        self.assertEqual(result['restore_attempts'], 0)
    def test_wrong_readback_refuses_then_restores_initial(self):
        result = self.run_probe(FakeLabels(mismatch=True))
        self.assertFalse(result['qualified'])
        self.assertTrue(result['initial_label_restored_exactly'])
        self.assertEqual(result['restore_attempts'], 1)
    def test_restore_denied_never_claims_success(self):
        result = self.run_probe(FakeLabels(deny_restore=True))
        self.assertFalse(result['qualified'])
        self.assertFalse(result['initial_label_restored_exactly'])
        self.assertEqual(result['restoration_errno'], errno.EPERM)
    def test_absent_and_identical_labels_never_qualify_transition(self):
        result = self.run_probe(FakeLabels(initial=None))
        self.assertFalse(result['qualified'])
        self.assertEqual(result['set_attempts'], 0)
        second = m.exercise(self.fd, 'other', 'directory', INITIAL, os.getuid(), os.getgid(), self.journal, FakeLabels())
        self.assertFalse(second['qualified'])
        self.assertEqual(second['set_attempts'], 0)

    def test_manifest_absence_and_effective_label_are_separate_authorities(self):
        from unittest.mock import patch
        with patch.object(m, 'label_summary', return_value=m.DIAGNOSTIC_EFFECTIVE_LABEL):
            authority = m.source_label_authority(m.DIAGNOSTIC_BUILD, {'xattrs': {}}, [], b'opaque-effective-label')
        self.assertFalse(authority['enumerated'])
        self.assertFalse(authority['stored_attribute_absence_reproduction_qualified'])
        self.assertEqual(authority['diagnostic_sha256'], m.SOURCE_DIAGNOSTIC_SHA)

    def test_unknown_effective_label_or_changed_enumeration_refused(self):
        with self.assertRaisesRegex(m.Refusal, 'lacks exact readonly diagnostic'):
            m.source_label_authority(m.DIAGNOSTIC_BUILD, {'xattrs': {}}, [], b'unverified-different-label')
        with self.assertRaisesRegex(m.Refusal, 'enumerated xattr names differ'):
            m.source_label_authority(m.DIAGNOSTIC_BUILD, {'xattrs': {}}, [m.ATTRIBUTE], b'other')


if __name__ == '__main__':
    unittest.main(verbosity=2)
