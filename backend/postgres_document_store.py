import json
import os
from postgres_file_store import PostgresFileStore
from typing import Any, Dict, List, Optional


class PostgresDocumentStore:
    """PostgreSQL store for semi-structured and unstructured document records."""

    def __init__(self, database_url: str, documents_dir: str = "data/document_ai/documents"):
        self.database_url = database_url
        self.files = PostgresFileStore(database_url)
        self.documents_dir = documents_dir
        os.makedirs(documents_dir, exist_ok=True)
        self._init_db()

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as error:
            raise RuntimeError(
                "PostgreSQL mode requires psycopg. Install dependencies with: pip install 'psycopg[binary]'"
            ) from error

        return psycopg.connect(self.database_url, row_factory=dict_row)

    def _init_db(self):
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ai_documents (
                        id BIGSERIAL PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        document_type TEXT NOT NULL DEFAULT 'unknown',
                        is_invoice BOOLEAN NOT NULL DEFAULT FALSE,
                        classification_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
                        extraction_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
                        extracted_text TEXT NOT NULL DEFAULT '',
                        fields_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        detections_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                        rag_document_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        invoice_id BIGINT,
                        metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        status TEXT NOT NULL DEFAULT 'draft',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_pg_ai_documents_document_type ON ai_documents(document_type)"
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_pg_ai_documents_is_invoice ON ai_documents(is_invoice)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_pg_ai_documents_invoice_id ON ai_documents(invoice_id)")
            connection.commit()

    def create_document(
        self,
        filename: str,
        file_type: str,
        source_file_path: str,
        document_type: str,
        is_invoice: bool,
        classification_confidence: float,
        extraction_confidence: float,
        extracted_text: str,
        fields: Dict[str, Any],
        detections: List[Dict[str, Any]],
        rag_document: Optional[Dict[str, Any]],
        invoice_id: Optional[int],
        metadata: Dict[str, Any],
        status: str = "draft",
    ) -> Dict[str, Any]:
        stored_path = self._persist_file(source_file_path, filename)

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO ai_documents (
                        filename, file_type, file_path, document_type, is_invoice,
                        classification_confidence, extraction_confidence, extracted_text,
                        fields_json, detections_json, rag_document_json, invoice_id,
                        metadata_json, status
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s,
                        %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::jsonb, %s
                    )
                    RETURNING id
                    """,
                    (
                        filename,
                        file_type,
                        stored_path,
                        document_type,
                        bool(is_invoice),
                        classification_confidence,
                        extraction_confidence,
                        extracted_text,
                        json.dumps(fields, ensure_ascii=False),
                        json.dumps(detections, ensure_ascii=False),
                        json.dumps(rag_document or {}, ensure_ascii=False),
                        invoice_id,
                        json.dumps(metadata, ensure_ascii=False),
                        status,
                    ),
                )
                document_id = int(cursor.fetchone()["id"])
            connection.commit()

        return self.get_document(document_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM ai_documents ORDER BY created_at DESC")
                rows = cursor.fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_document(self, document_id: int) -> Dict[str, Any]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM ai_documents WHERE id = %s", (document_id,))
                row = cursor.fetchone()

        if row is None:
            raise ValueError(f"Document {document_id} not found")

        return self._row_to_document(row)

    def _persist_file(self, source_file_path: str, filename: str) -> str:
        return self.files.put(source_file_path)

    def _row_to_document(self, row) -> Dict[str, Any]:
        payload = dict(row)
        payload["is_invoice"] = bool(payload["is_invoice"])
        payload["fields"] = self._json_value(payload.pop("fields_json"), {})
        payload["detections"] = self._json_value(payload.pop("detections_json"), [])
        payload["rag_document"] = self._json_value(payload.pop("rag_document_json"), {})
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
