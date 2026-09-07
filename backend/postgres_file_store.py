"""Bounded durable storage for the small hosted deployment."""

import hashlib
import os
import re
from pathlib import Path


class PostgresFileStore:
    PREFIX = "postgres-file:"

    def __init__(self, database_url):
        self.database_url = database_url
        self.max_file_bytes = int(os.getenv("MAX_FILE_SIZE", "10485760"))
        self.quota_bytes = int(os.getenv("FILE_STORAGE_QUOTA_BYTES", "104857600"))
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS stored_files (
                    digest TEXT PRIMARY KEY,
                    content BYTEA NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

    def _connect(self):
        import psycopg
        return psycopg.connect(self.database_url, connect_timeout=10)

    def put(self, source_path):
        with Path(source_path).open("rb") as source:
            content = source.read(self.max_file_bytes + 1)
        if len(content) > self.max_file_bytes:
            raise ValueError("Fichier trop volumineux pour le stockage.")
        digest = hashlib.sha256(content).hexdigest()
        with self._connect() as connection:
            # Serialize quota checks so concurrent uploads cannot exceed the limit.
            connection.execute("LOCK TABLE stored_files IN SHARE ROW EXCLUSIVE MODE")
            existing = connection.execute(
                "SELECT 1 FROM stored_files WHERE digest = %s", (digest,)
            ).fetchone()
            if not existing:
                used = connection.execute(
                    "SELECT COALESCE(SUM(octet_length(content)), 0) FROM stored_files"
                ).fetchone()[0]
                if used + len(content) > self.quota_bytes:
                    raise ValueError("Quota de stockage des fichiers atteint.")
                connection.execute(
                    "INSERT INTO stored_files (digest, content) VALUES (%s, %s)",
                    (digest, content),
                )
        return self.PREFIX + digest

    def get(self, reference):
        digest = reference.removeprefix(self.PREFIX)
        if not reference.startswith(self.PREFIX) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Reference de fichier invalide.")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT content FROM stored_files WHERE digest = %s", (digest,)
            ).fetchone()
        if row is None:
            raise FileNotFoundError("Fichier introuvable dans la base.")
        return bytes(row[0])
