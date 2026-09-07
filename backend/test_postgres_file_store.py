import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from postgres_file_store import PostgresFileStore


class FileStoreTests(unittest.TestCase):
    def setUp(self):
        self.connection = MagicMock()
        self.connection.__enter__.return_value = self.connection
        self.patch = patch.object(PostgresFileStore, "_connect", return_value=self.connection)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.store = PostgresFileStore("test")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "invoice.png"
        self.source.write_bytes(b"invoice fixture")

    def test_insert_and_retrieve_without_local_file(self):
        self.connection.execute.return_value.fetchone.side_effect = [None, (0,), (b"invoice fixture",)]
        reference = self.store.put(self.source)
        self.source.unlink()
        self.assertEqual(self.store.get(reference), b"invoice fixture")
        self.assertTrue(any("INSERT INTO stored_files" in call.args[0]
                            for call in self.connection.execute.call_args_list))

    def test_duplicate_does_not_consume_quota(self):
        self.store.quota_bytes = 0
        self.connection.execute.return_value.fetchone.return_value = (1,)
        self.store.put(self.source)
        self.assertFalse(any("INSERT INTO stored_files" in call.args[0]
                             for call in self.connection.execute.call_args_list))

    def test_quota_rejected(self):
        self.store.quota_bytes = 1
        self.connection.execute.return_value.fetchone.side_effect = [None, (0,)]
        with self.assertRaisesRegex(ValueError, "Quota"):
            self.store.put(self.source)

    def test_oversized_file_rejected(self):
        self.store.max_file_bytes = 1
        with self.assertRaisesRegex(ValueError, "volumineux"):
            self.store.put(self.source)

    def test_invalid_reference(self):
        with self.assertRaises(ValueError):
            self.store.get("../../secret")

    def test_missing_file(self):
        self.connection.execute.return_value.fetchone.return_value = None
        with self.assertRaises(FileNotFoundError):
            self.store.get(self.store.PREFIX + "a" * 64)


if __name__ == "__main__":
    unittest.main()
