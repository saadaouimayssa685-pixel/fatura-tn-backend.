import json
import os
import sqlite3
from typing import Any, Dict, List, Optional


class SQLiteRAGStore:
    """Small local SQLite store for documents, chunks, and embeddings."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_type TEXT,
                    file_path TEXT,
                    text_length INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER,
                    chunk_text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
            connection.commit()

    def add_document(self, filename: str, file_type: str, file_path: str, text_length: int, metadata: Dict[str, Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO documents (filename, file_type, file_path, text_length, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (filename, file_type, file_path, text_length, json.dumps(metadata, ensure_ascii=False)),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def add_chunk(
        self,
        document_id: int,
        chunk_index: int,
        page_number: Optional[int],
        chunk_text: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO chunks (document_id, chunk_index, page_number, chunk_text, embedding_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    chunk_index,
                    page_number,
                    chunk_text,
                    json.dumps(embedding),
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    d.*,
                    COUNT(c.id) AS chunks_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC
                """
            ).fetchall()

        return [self._document_from_row(row) for row in rows]

    def get_chunks(self, document_id: Optional[int] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM chunks"
        params = []
        if document_id is not None:
            query += " WHERE document_id = ?"
            params.append(document_id)

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, params).fetchall()

        return [self._chunk_from_row(row) for row in rows]

    def _document_from_row(self, row) -> Dict[str, Any]:
        payload = dict(row)
        payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
        return payload

    def _chunk_from_row(self, row) -> Dict[str, Any]:
        payload = dict(row)
        payload["embedding"] = json.loads(payload.pop("embedding_json") or "[]")
        payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
        return payload
