import csv
import io
import json
import os
import sqlite3
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional


class FaturaDatasetStore:
    """SQLite index for the FATURA invoice dataset zip.

    The zip can contain thousands of images and annotations. To keep the local
    project light, SQLite stores metadata, split rows and JSON annotations while
    image binaries stay inside the source zip.
    """

    def __init__(self, db_path: str, zip_path: str):
        self.db_path = db_path
        self.zip_path = zip_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_files (
                    path TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    image_name TEXT,
                    annotation_format TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_splits (
                    split TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    annotation_name TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    annotation_path TEXT NOT NULL,
                    PRIMARY KEY (split, image_name)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_annotations (
                    annotation_path TEXT PRIMARY KEY,
                    annotation_format TEXT NOT NULL,
                    image_name TEXT,
                    words_count INTEGER NOT NULL DEFAULT 0,
                    boxes_count INTEGER NOT NULL DEFAULT 0,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_dataset_files_kind ON dataset_files(kind)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_dataset_annotations_image ON dataset_annotations(image_name)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_dataset_splits_split ON dataset_splits(split)")
            connection.commit()

    def needs_import(self) -> bool:
        if not os.path.exists(self.zip_path):
            return False
        zip_stat = os.stat(self.zip_path)
        expected_signature = f"{zip_stat.st_size}:{int(zip_stat.st_mtime)}"
        with self._connect() as connection:
            current = connection.execute(
                "SELECT value FROM dataset_meta WHERE key = 'zip_signature'"
            ).fetchone()
            file_count = connection.execute("SELECT COUNT(*) FROM dataset_files").fetchone()[0]
        return not current or current[0] != expected_signature or file_count == 0

    def import_from_zip(self) -> Dict[str, Any]:
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"FATURA zip introuvable: {self.zip_path}")

        started = time.time()
        zip_stat = os.stat(self.zip_path)
        zip_signature = f"{zip_stat.st_size}:{int(zip_stat.st_mtime)}"
        file_rows = []
        split_rows = []
        annotation_rows = []

        with zipfile.ZipFile(self.zip_path) as archive:
            names = archive.namelist()
            for info in archive.infolist():
                if info.is_dir():
                    continue
                extension = Path(info.filename).suffix.lower()
                kind = self._kind_from_path(info.filename)
                image_name = self._image_name_from_path(info.filename)
                annotation_format = self._annotation_format(info.filename)
                file_rows.append((info.filename, kind, extension, info.file_size, image_name, annotation_format))

            for split_name in ("train", "dev", "test"):
                csv_path = f"invoices_dataset_final/strat1_{split_name}.csv"
                if csv_path not in names:
                    continue
                with archive.open(csv_path) as raw_file:
                    reader = csv.DictReader(io.TextIOWrapper(raw_file, encoding="utf-8-sig", newline=""))
                    for row in reader:
                        image_name = (row.get("img_path") or "").strip()
                        annotation_name = (row.get("annot_path") or "").strip()
                        if not image_name or not annotation_name:
                            continue
                        split_rows.append(
                            (
                                split_name,
                                image_name,
                                annotation_name,
                                f"invoices_dataset_final/images/{image_name}",
                                f"invoices_dataset_final/Annotations/Original_Format/{annotation_name}",
                            )
                        )

            for info in archive.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".json"):
                    continue
                annotation_format = self._annotation_format(info.filename)
                with archive.open(info.filename) as raw_file:
                    payload = json.load(io.TextIOWrapper(raw_file, encoding="utf-8"))
                image_name = payload.get("path") if isinstance(payload, dict) else self._image_name_from_path(info.filename)
                words = payload.get("words", []) if isinstance(payload, dict) else []
                boxes = payload.get("bboxes", []) if isinstance(payload, dict) else []
                tags = payload.get("ner_tags", []) if isinstance(payload, dict) else []
                annotation_rows.append(
                    (
                        info.filename,
                        annotation_format,
                        str(image_name or ""),
                        len(words) if isinstance(words, list) else 0,
                        len(boxes) if isinstance(boxes, list) else 0,
                        json.dumps(tags, ensure_ascii=False),
                        json.dumps(payload, ensure_ascii=False),
                    )
                )

        with self._connect() as connection:
            connection.execute("DELETE FROM dataset_files")
            connection.execute("DELETE FROM dataset_splits")
            connection.execute("DELETE FROM dataset_annotations")
            connection.executemany(
                """
                INSERT OR REPLACE INTO dataset_files (
                    path, kind, extension, size_bytes, image_name, annotation_format
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                file_rows,
            )
            connection.executemany(
                """
                INSERT OR REPLACE INTO dataset_splits (
                    split, image_name, annotation_name, image_path, annotation_path
                ) VALUES (?, ?, ?, ?, ?)
                """,
                split_rows,
            )
            connection.executemany(
                """
                INSERT OR REPLACE INTO dataset_annotations (
                    annotation_path, annotation_format, image_name, words_count,
                    boxes_count, tags_json, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                annotation_rows,
            )
            connection.execute(
                "INSERT OR REPLACE INTO dataset_meta (key, value) VALUES ('zip_path', ?)",
                (self.zip_path,),
            )
            connection.execute(
                "INSERT OR REPLACE INTO dataset_meta (key, value) VALUES ('zip_signature', ?)",
                (zip_signature,),
            )
            connection.execute(
                "INSERT OR REPLACE INTO dataset_meta (key, value) VALUES ('indexed_at', datetime('now'))"
            )
            connection.commit()

        return {
            "db_path": self.db_path,
            "zip_path": self.zip_path,
            "files": len(file_rows),
            "splits": len(split_rows),
            "annotations": len(annotation_rows),
            "processing_time": round(time.time() - started, 3),
        }

    def stats(self) -> Dict[str, Any]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            meta = {
                row["key"]: row["value"]
                for row in connection.execute("SELECT key, value FROM dataset_meta").fetchall()
            }
            by_kind = {
                row["kind"]: row["count"]
                for row in connection.execute("SELECT kind, COUNT(*) AS count FROM dataset_files GROUP BY kind").fetchall()
            }
            by_split = {
                row["split"]: row["count"]
                for row in connection.execute("SELECT split, COUNT(*) AS count FROM dataset_splits GROUP BY split").fetchall()
            }
            by_format = {
                row["annotation_format"]: row["count"]
                for row in connection.execute("SELECT annotation_format, COUNT(*) AS count FROM dataset_annotations GROUP BY annotation_format").fetchall()
            }
            totals = connection.execute(
                """
                SELECT
                    COUNT(*) AS annotations,
                    COALESCE(SUM(words_count), 0) AS words,
                    COALESCE(SUM(boxes_count), 0) AS boxes
                FROM dataset_annotations
                """
            ).fetchone()

        return {
            "db_path": self.db_path,
            "zip_path": self.zip_path,
            "zip_available": os.path.exists(self.zip_path),
            "meta": meta,
            "files_by_kind": by_kind,
            "splits": by_split,
            "annotations_by_format": by_format,
            "annotations": totals["annotations"],
            "words": totals["words"],
            "boxes": totals["boxes"],
        }

    def samples(self, split: Optional[str] = None, limit: int = 12) -> List[Dict[str, Any]]:
        limit = max(1, min(limit, 100))
        query = "SELECT * FROM dataset_splits"
        params: List[Any] = []
        if split:
            query += " WHERE split = ?"
            params.append(split)
        query += " ORDER BY image_name LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def invoice_catalog(
        self,
        search: str = "",
        invoice_name: str = "",
        supplier: str = "",
        company: str = "",
        tax_id: str = "",
        page: int = 1,
        page_size: int = 10,
    ) -> Dict[str, Any]:
        where_parts: List[str] = []
        params: List[Any] = []

        def add_payload_filter(value: str, fields_sql: str = "a.payload_json LIKE ?"):
            normalized = (value or "").strip()
            if not normalized:
                return
            where_parts.append(fields_sql)
            params.append(f"%{normalized}%")

        normalized_search = (search or "").strip()
        if normalized_search:
            like_value = f"%{normalized_search}%"
            where_parts.append("(s.image_name LIKE ? OR s.annotation_name LIKE ? OR a.payload_json LIKE ?)")
            params.extend([like_value, like_value, like_value])

        normalized_invoice_name = (invoice_name or "").strip()
        if normalized_invoice_name:
            like_value = f"%{normalized_invoice_name}%"
            where_parts.append("(s.image_name LIKE ? OR s.annotation_name LIKE ? OR a.payload_json LIKE ?)")
            params.extend([like_value, like_value, like_value])

        add_payload_filter(supplier)
        add_payload_filter(company)
        add_payload_filter(tax_id)

        where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        safe_page = max(1, int(page or 1))
        safe_page_size = min(100, max(1, int(page_size or 10)))
        offset = (safe_page - 1) * safe_page_size

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            total = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM dataset_splits s
                    LEFT JOIN dataset_annotations a ON a.annotation_path = s.annotation_path
                    {where_sql}
                    """,
                    tuple(params),
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT
                    s.split,
                    s.image_name,
                    s.annotation_name,
                    s.image_path,
                    s.annotation_path,
                    a.words_count,
                    a.boxes_count,
                    a.payload_json
                FROM dataset_splits s
                LEFT JOIN dataset_annotations a ON a.annotation_path = s.annotation_path
                {where_sql}
                ORDER BY s.image_name
                LIMIT ? OFFSET ?
                """,
                tuple(params + [safe_page_size, offset]),
            ).fetchall()

        return {
            "items": [self._catalog_row_to_invoice(row) for row in rows],
            "total": total,
            "page": safe_page,
            "page_size": safe_page_size,
            "total_pages": max(1, (total + safe_page_size - 1) // safe_page_size),
        }

    def _catalog_row_to_invoice(self, row) -> Dict[str, Any]:
        payload = {}
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            payload = {}
        words = payload.get("words", []) if isinstance(payload, dict) else []
        text = " ".join(str(word) for word in words)
        invoice_number = self._find_after_labels(words, {"invoice", "facture", "#", "n°", "numéro", "number"})
        invoice_date = self._find_after_labels(words, {"date"})
        total_amount = self._find_amount_near_total(words)
        supplier_name = self._guess_supplier(words)
        customer_name = self._find_after_labels(words, {"buyer", "client", "customer"})
        currency = self._find_currency(words)

        return {
            "id": f"dataset:{row['image_name']}",
            "filename": row["image_name"],
            "file_type": Path(row["image_name"]).suffix.lower().lstrip(".") or "jpg",
            "file_path": row["image_path"],
            "model_used": "dataset_annotation",
            "confidence_score": 1,
            "status": "dataset",
            "created_at": "",
            "updated_at": "",
            "validated_at": None,
            "invoice_number": invoice_number or row["annotation_name"],
            "supplier_name": supplier_name or "Facture dataset",
            "customer_name": customer_name,
            "invoice_date": invoice_date,
            "due_date": "",
            "subtotal": 0,
            "tax_amount": 0,
            "stamp_duty": 0,
            "total_amount": total_amount,
            "currency": currency,
            "normalized_data": {
                "invoice_number": invoice_number,
                "vendor_name": supplier_name,
                "customer_name": customer_name,
                "date": invoice_date,
                "total_amount": total_amount,
                "currency": currency,
            },
            "raw_data": {
                "source": "fatura_dataset",
                "split": row["split"],
                "annotation_path": row["annotation_path"],
                "words_count": row["words_count"],
                "boxes_count": row["boxes_count"],
                "text_preview": text[:500],
            },
            "structured_data": {
                "invoice_number": invoice_number,
                "supplier_name": supplier_name,
                "customer_name": customer_name,
                "invoice_date": invoice_date,
                "due_date": "",
                "total_amount": total_amount,
                "currency": currency,
            },
        }

    def get_catalog_invoice(self, invoice_id: str) -> Dict[str, Any]:
        image_name = invoice_id.replace("dataset:", "", 1)
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT s.*, a.annotation_format, a.words_count, a.boxes_count, a.tags_json, a.payload_json
                FROM dataset_splits s
                LEFT JOIN dataset_annotations a ON a.annotation_path = s.annotation_path
                WHERE s.image_name = ?
                ORDER BY s.split ASC
                LIMIT 1
                """,
                (image_name,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Facture dataset introuvable: {invoice_id}")
        return self._catalog_row_to_invoice(row)

    def read_image_bytes(self, invoice_id: str) -> tuple[str, bytes]:
        image_name = invoice_id.replace("dataset:", "", 1)
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT image_name, image_path FROM dataset_splits WHERE image_name = ? LIMIT 1",
                (image_name,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Image dataset introuvable: {invoice_id}")
        with zipfile.ZipFile(self.zip_path) as archive:
            return row["image_name"], archive.read(row["image_path"])

    @staticmethod
    def _find_after_labels(words: List[Any], labels: set[str]) -> str:
        clean_words = [str(word).strip() for word in words if str(word).strip()]
        lower_words = [word.lower().strip(":#") for word in clean_words]
        for index, word in enumerate(lower_words[:-1]):
            if word in labels:
                return " ".join(clean_words[index + 1:index + 4]).strip(" :#")
        return ""

    @staticmethod
    def _find_amount_near_total(words: List[Any]) -> float:
        clean_words = [str(word).strip() for word in words if str(word).strip()]
        for index, word in enumerate(clean_words):
            if word.lower() in {"total", "balance_due", "ttc"}:
                for candidate in clean_words[index + 1:index + 5]:
                    normalized = candidate.replace(",", ".")
                    try:
                        return float("".join(char for char in normalized if char.isdigit() or char in ".-"))
                    except ValueError:
                        continue
        return 0.0

    @staticmethod
    def _find_currency(words: List[Any]) -> str:
        currencies = {"TND", "EUR", "USD", "GBP"}
        for word in words:
            normalized = str(word).upper().strip(".,:;")
            if normalized in currencies:
                return normalized
        return "TND"

    @staticmethod
    def _guess_supplier(words: List[Any]) -> str:
        clean_words = [str(word).strip() for word in words if str(word).strip()]
        for index, word in enumerate(clean_words):
            if word.lower() in {"seller", "supplier", "fournisseur", "vendor"}:
                return " ".join(clean_words[index + 1:index + 4]).strip(" :")
        return " ".join(clean_words[:3]).strip(" :")

    def annotation(self, annotation_path: str) -> Dict[str, Any]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM dataset_annotations WHERE annotation_path = ?",
                (annotation_path,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Annotation introuvable: {annotation_path}")
        payload = dict(row)
        payload["tags"] = json.loads(payload.pop("tags_json") or "[]")
        payload["payload"] = json.loads(payload.pop("payload_json") or "{}")
        return payload

    @staticmethod
    def _kind_from_path(path: str) -> str:
        lower_path = path.lower()
        if lower_path.endswith(".jpg") or lower_path.endswith(".jpeg") or lower_path.endswith(".png"):
            return "image"
        if lower_path.endswith(".json"):
            return "annotation"
        if lower_path.endswith(".csv"):
            return "split_csv"
        if lower_path.endswith(".txt"):
            return "metadata"
        return "other"

    @staticmethod
    def _annotation_format(path: str) -> Optional[str]:
        parts = path.split("/")
        if "Annotations" in parts:
            index = parts.index("Annotations")
            if index + 1 < len(parts):
                return parts[index + 1]
        return None

    @staticmethod
    def _image_name_from_path(path: str) -> str:
        name = Path(path).name
        return str(Path(name).with_suffix(".jpg"))
