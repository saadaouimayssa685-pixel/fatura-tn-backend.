import json
from typing import Any, Dict, List, Optional


class PostgresRAGStore:
    """PostgreSQL store for RAG documents, chunks and embeddings.

    Embeddings are stored as JSONB for this MVP. Later, this table can migrate
    to pgvector while keeping the same service API.
    """

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._init_db()

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as error:
            raise RuntimeError(
                "PostgreSQL RAG mode requires psycopg. Install dependencies with: pip install 'psycopg[binary]'"
            ) from error

        return psycopg.connect(self.database_url, row_factory=dict_row)

    def _init_db(self):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_documents (
                        id BIGSERIAL PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_type TEXT,
                        file_path TEXT,
                        text_length INTEGER NOT NULL DEFAULT 0,
                        metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        id BIGSERIAL PRIMARY KEY,
                        document_id BIGINT NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        page_number INTEGER,
                        chunk_text TEXT NOT NULL,
                        embedding_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                        metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_pg_rag_chunks_document_id ON rag_chunks(document_id)")
            connection.commit()

    def add_document(self, filename: str, file_type: str, file_path: str, text_length: int, metadata: Dict[str, Any]) -> int:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO rag_documents (filename, file_type, file_path, text_length, metadata_json)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    RETURNING id
                    """,
                    (filename, file_type, file_path, text_length, json.dumps(metadata, ensure_ascii=False)),
                )
                document_id = int(cursor.fetchone()["id"])
            connection.commit()
        return document_id

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
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO rag_chunks (
                        document_id, chunk_index, page_number, chunk_text, embedding_json, metadata_json
                    )
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    RETURNING id
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
                chunk_id = int(cursor.fetchone()["id"])
            connection.commit()
        return chunk_id

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT d.*, COUNT(c.id) AS chunks_count
                    FROM rag_documents d
                    LEFT JOIN rag_chunks c ON c.document_id = d.id
                    GROUP BY d.id
                    ORDER BY d.created_at DESC
                    """
                )
                rows = cursor.fetchall()
        return [self._document_from_row(row) for row in rows]

    def get_chunks(self, document_id: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                if document_id is None:
                    cursor.execute("SELECT * FROM rag_chunks ORDER BY document_id ASC, chunk_index ASC")
                else:
                    cursor.execute(
                        "SELECT * FROM rag_chunks WHERE document_id = %s ORDER BY chunk_index ASC",
                        (document_id,),
                    )
                rows = cursor.fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def _document_from_row(self, row) -> Dict[str, Any]:
        payload = dict(row)
        payload["metadata"] = self._json_value(payload.pop("metadata_json"), {})
        return payload

    def _chunk_from_row(self, row) -> Dict[str, Any]:
        payload = dict(row)
        payload["embedding"] = self._json_value(payload.pop("embedding_json"), [])
        payload["metadata"] = self._json_value(payload.pop("metadata_json"), {})
        return payload

    def _json_value(self, value: Any, default: Any) -> Any:
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return default
