"""Transactions must release file handles without waiting for garbage collection."""
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import db


class DatabaseConnectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        p = patch.object(db, 'DB_FILE', str(Path(self.temp.name) / 'test.db'))
        p.start(); self.addCleanup(p.stop)
        with db.get_conn() as conn:
            conn.execute('CREATE TABLE marker(value TEXT)')

    def assert_closed(self, conn):
        with self.assertRaises(sqlite3.ProgrammingError):
            conn.execute('SELECT 1')

    def test_success_commits_and_closes_even_with_a_live_reference(self):
        with db.get_conn() as conn:
            conn.execute("INSERT INTO marker VALUES ('saved')")
        self.assert_closed(conn)
        with db.get_conn() as check:
            self.assertEqual(check.execute('SELECT value FROM marker').fetchone()[0], 'saved')
        self.assert_closed(check)

    def test_exception_rolls_back_and_releases_the_connection(self):
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            with db.get_conn() as conn:
                conn.execute("INSERT INTO marker VALUES ('discard')")
                raise RuntimeError('request failed')
        self.assert_closed(conn)
        with db.get_conn() as check:
            self.assertEqual(check.execute('SELECT COUNT(*) FROM marker').fetchone()[0], 0)

    def test_setup_failure_also_closes_the_handle(self):
        connections = []
        real_connect = sqlite3.connect

        class BrokenSetup(sqlite3.Connection):
            closed = False
            def execute(self, *args, **kwargs):
                raise RuntimeError('pragma failed')
            def close(self):
                self.closed = True
                super().close()

        def connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs, factory=BrokenSetup)
            connections.append(conn)
            return conn

        with patch.object(db.sqlite3, 'connect', side_effect=connect):
            with self.assertRaisesRegex(RuntimeError, 'pragma failed'):
                with db.get_conn():
                    self.fail('connection should not be yielded')
        self.assertTrue(connections[0].closed)


if __name__ == '__main__': unittest.main()
