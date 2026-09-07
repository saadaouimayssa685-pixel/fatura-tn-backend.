import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from auth_database import connect_auth, PostgresAuthConnection


class AuthDatabaseTests(unittest.TestCase):
    def test_sqlite_survives_new_connection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "auth.sqlite3")
            with connect_auth("", path) as db:
                db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT UNIQUE)")
                db.execute("INSERT INTO users VALUES (?, ?)", (1, "test@example.com"))
            with connect_auth("", path) as db:
                self.assertEqual(db.execute("SELECT email FROM users").fetchone()[0], "test@example.com")

    def test_sqlite_rolls_back_failed_transaction(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "auth.sqlite3")
            with connect_auth("", path) as db:
                db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
            with self.assertRaises(sqlite3.IntegrityError):
                with connect_auth("", path) as db:
                    db.execute("INSERT INTO users VALUES (1)")
                    db.execute("INSERT INTO users VALUES (1)")
            with connect_auth("", path) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM users").fetchone()[0], 0)

    def test_postgres_insert_returns_id_without_interpolating_values(self):
        connection = Mock()
        connection.execute.return_value.fetchone.return_value = (42,)
        db = PostgresAuthConnection(connection)
        email = "literal?quote'@example.com"
        result = db.execute("INSERT INTO users (email) VALUES (?)", (email,))
        self.assertEqual(result.lastrowid, 42)
        connection.execute.assert_called_once_with(
            "INSERT INTO users (email) VALUES (%s) RETURNING id", (email,)
        )

    def test_postgres_schema_uses_precise_timestamps(self):
        connection = Mock()
        db = PostgresAuthConnection(connection)
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL)")
        self.assertIn("BIGSERIAL PRIMARY KEY", connection.execute.call_args.args[0])
        self.assertIn("DOUBLE PRECISION", connection.execute.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
