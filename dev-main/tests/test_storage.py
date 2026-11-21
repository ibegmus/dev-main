import os
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import storage


class StorageMigrationTests(unittest.TestCase):
    def test_legacy_copy_preserves_records_and_never_overwrites_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); old = root / 'app.db'; new = root / 'data' / 'app.db'
            with closing(sqlite3.connect(old)) as conn, conn:
                conn.execute('CREATE TABLE marker(value TEXT)')
                conn.execute("INSERT INTO marker VALUES ('old')")
            with patch.object(storage, 'DEFAULT_DATABASE', new), patch.object(storage, 'LEGACY_DATABASE', old):
                storage.prepare_database_path(new)
                with closing(sqlite3.connect(new)) as conn, conn:
                    self.assertEqual(conn.execute('SELECT value FROM marker').fetchone()[0], 'old')
                    conn.execute("UPDATE marker SET value='new'")
                storage.prepare_database_path(new)
            with closing(sqlite3.connect(new)) as conn, conn:
                self.assertEqual(conn.execute('SELECT value FROM marker').fetchone()[0], 'new')
            with closing(sqlite3.connect(old)) as conn, conn:
                self.assertEqual(conn.execute('SELECT value FROM marker').fetchone()[0], 'old')

    def test_windows_file_operations_only_run_after_connections_close(self):
        self._check_windows_handles(invalid=False)

    def test_failed_backup_releases_handles_before_windows_cleanup(self):
        self._check_windows_handles(invalid=True)

    def _check_windows_handles(self, invalid):
        connections = []
        real_connect = sqlite3.connect
        real_replace = os.replace
        real_unlink = Path.unlink

        class TrackedConnection(sqlite3.Connection):
            closed = False

            def close(self):
                super().close()
                self.closed = True

        def connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs, factory=TrackedConnection)
            connections.append(conn)  # Keep references alive; do not rely on GC.
            return conn

        def assert_no_open_handles():
            if any(not conn.closed for conn in connections):
                raise PermissionError('Windows: SQLite file is still open')

        def replace(*args):
            assert_no_open_handles()
            return real_replace(*args)

        def unlink(path, *args, **kwargs):
            if path.name.startswith('.migration-'):
                assert_no_open_handles()
            return real_unlink(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); old = root / 'app.db'; new = root / 'data' / 'app.db'
            if invalid:
                old.write_bytes(b'invalid database')
            else:
                with closing(real_connect(old)) as conn, conn:
                    conn.execute('CREATE TABLE marker(value TEXT)')
                    conn.execute("INSERT INTO marker VALUES ('preserved')")
            try:
                with patch.object(storage, 'DEFAULT_DATABASE', new), patch.object(storage, 'LEGACY_DATABASE', old), \
                        patch.object(storage.sqlite3, 'connect', side_effect=connect), \
                        patch.object(storage.os, 'replace', side_effect=replace), \
                        patch.object(Path, 'unlink', autospec=True, side_effect=unlink):
                    if invalid:
                        with self.assertRaises(sqlite3.DatabaseError):
                            storage.prepare_database_path(new)
                    else:
                        storage.prepare_database_path(new)
                self.assertEqual(len(connections), 2)
                self.assertTrue(all(conn.closed for conn in connections))
                self.assertEqual(list(new.parent.glob('.migration-*')), [])
                self.assertTrue(old.exists())
                if not invalid:
                    with closing(real_connect(new)) as conn:
                        self.assertEqual(conn.execute('SELECT value FROM marker').fetchone()[0], 'preserved')
            finally:
                for conn in connections:
                    conn.close()

    def test_custom_database_does_not_import_legacy_records(self):
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / 'private' / 'custom.db'
            storage.prepare_database_path(custom)
            self.assertTrue(custom.parent.exists())
            self.assertFalse(custom.exists())

    def test_invalid_legacy_database_is_preserved_without_partial_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); old = root/'app.db'; new = root/'data'/'app.db'
            old.write_bytes(b'invalid database')
            with patch.object(storage, 'DEFAULT_DATABASE', new), patch.object(storage, 'LEGACY_DATABASE', old):
                with self.assertRaises(sqlite3.DatabaseError):
                    storage.prepare_database_path(new)
            self.assertFalse(new.exists())
            self.assertEqual(old.read_bytes(), b'invalid database')
            self.assertEqual(list(new.parent.glob('.migration-*')), [])


if __name__ == '__main__': unittest.main()
