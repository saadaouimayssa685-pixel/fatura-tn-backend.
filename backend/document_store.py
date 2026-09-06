import json
import os
import shutil
import sqlite3
from typing import Any, Dict, List, Optional


class DocumentStore:
    """SQLite store for semi-structured and unstructured document AI records.

    The relational columns keep stable operational metadata. Variable AI outputs
    such as OCR text, object detections, extracted fields, RAG links and raw
    model responses are stored as JSON/TEXT so the schema can evolve without a
    migration for every new document type.
    """

    def __init__(
        self,
        db_path: str = "data/document_ai/documents.sqlite3",
        documents_dir: str = "data/document_ai/documents",
    ):
        self.db_path = db_path
        self.documents_dir = documents_dir
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(documents_dir, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    document_type TEXT NOT NULL DEFAULT 'unknown',
                    is_invoice INTEGER NOT NULL DEFAULT 0,
                    classification_confidence REAL NOT NULL DEFAULT 0,
                    extraction_confidence REAL NOT NULL DEFAULT 0,
                    extracted_text TEXT NOT NULL DEFAULT '',
                    fields_json TEXT NOT NULL DEFAULT '{}',
                    detections_json TEXT NOT NULL DEFAULT '[]',
                    rag_document_json TEXT NOT NULL DEFAULT '{}',
                    invoice_id INTEGER,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'draft',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_ai_documents_document_type ON ai_documents(document_type)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_ai_documents_is_invoice ON ai_documents(is_invoice)"
            )
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
            cursor = connection.execute(
                """
                INSERT INTO ai_documents (
                    filename, file_type, file_path, document_type, is_invoice,
                    classification_confidence, extraction_confidence, extracted_text,
                    fields_json, detections_json, rag_document_json, invoice_id,
                    metadata_json, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    file_type,
                    stored_path,
                    document_type,
                    1 if is_invoice else 0,
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
            connection.commit()
            document_id = int(cursor.lastrowid)

        return self.get_document(document_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM ai_documents ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_document(self, document_id: int) -> Dict[str, Any]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM ai_documents WHERE id = ?",
                (document_id,),
            ).fetchone()

        if row is None:
            raise ValueError(f"Document {document_id} not found")

        return self._row_to_document(row)

    def _persist_file(self, source_file_path: str, filename: str) -> str:
        safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in filename)
        base_name, extension = os.path.splitext(safe_name)
        candidate = os.path.join(self.documents_dir, safe_name)
        counter = 1

        while os.path.exists(candidate):
            candidate = os.path.join(self.documents_dir, f"{base_name}_{counter}{extension}")
            counter += 1

        shutil.copy2(source_file_path, candidate)
        return candidate

    def _row_to_document(self, row) -> Dict[str, Any]:
        payload = dict(row)
        payload["is_invoice"] = bool(payload["is_invoice"])
        payload["fields"] = json.loads(payload.pop("fields_json") or "{}")
        payload["detections"] = json.loads(payload.pop("detections_json") or "[]")
        payload["rag_document"] = json.loads(payload.pop("rag_document_json") or "{}")
        payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
        return payload
