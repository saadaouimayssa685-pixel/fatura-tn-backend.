"""Authentication connections with the same transaction API on SQLite and PostgreSQL."""
from contextlib import contextmanager
import sqlite3


class PostgresAuthConnection:
    def __init__(self, connection):
        self.connection = connection

    def execute(self, statement, parameters=()):
        # Only application-owned authentication SQL is passed to this adapter.
        statement = statement.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY")
        statement = statement.replace("REAL NOT NULL", "DOUBLE PRECISION NOT NULL")
        statement = statement.replace("?", "%s")
        insert_user = statement.lstrip().startswith("INSERT INTO users (")
        if insert_user:
            statement = statement.rstrip().rstrip(";") + " RETURNING id"
        cursor = self.connection.execute(statement, parameters)
        if insert_user:
            return InsertedUser(cursor.fetchone()[0])
        return cursor


class InsertedUser:
    def __init__(self, user_id):
        self.lastrowid = user_id


@contextmanager
def connect_auth(database_url, sqlite_path):
    if database_url:
        import psycopg

        try:
            with psycopg.connect(database_url, connect_timeout=10) as connection:
                yield PostgresAuthConnection(connection)
        except psycopg.errors.UniqueViolation as error:
            raise sqlite3.IntegrityError("Authentication record already exists") from error
    else:
        connection = sqlite3.connect(sqlite_path)
        try:
            with connection:
                yield connection
        finally:
            connection.close()
