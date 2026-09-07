import json
import os
from decimal import Decimal
from postgres_file_store import PostgresFileStore
from typing import Any, Dict, List, Optional


def json_dumps(value: Any) -> str:
    """Serialize PostgreSQL values safely before writing JSONB audit payloads."""
    return json.dumps(
        value,
        ensure_ascii=False,
        default=lambda item: float(item) if isinstance(item, Decimal) else str(item),
    )


class PostgresInvoiceStore:
    """PostgreSQL store for structured invoice automation data.

    Raw OCR/model data stays in JSONB while operational fields are
    materialized in relational columns for dashboards, filters, validation
    and reporting.
    """

    def __init__(self, database_url: str, documents_dir: str = "invoice_storage/documents"):
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
                    CREATE TABLE IF NOT EXISTS suppliers (
                        id BIGSERIAL PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        tax_id TEXT,
                        address TEXT,
                        phone TEXT,
                        invoices_count INTEGER NOT NULL DEFAULT 0,
                        total_amount NUMERIC(18, 3) NOT NULL DEFAULT 0,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS invoices (
                        id BIGSERIAL PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        extracted_text TEXT,
                        normalized_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        raw_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        model_used TEXT,
                        confidence_score DOUBLE PRECISION DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'draft',
                        invoice_number TEXT,
                        supplier_id BIGINT REFERENCES suppliers(id),
                        supplier_name TEXT,
                        customer_name TEXT,
                        invoice_date TEXT,
                        due_date TEXT,
                        subtotal NUMERIC(18, 3) DEFAULT 0,
                        tax_amount NUMERIC(18, 3) DEFAULT 0,
                        stamp_duty NUMERIC(18, 3) DEFAULT 0,
                        total_amount NUMERIC(18, 3) DEFAULT 0,
                        currency TEXT DEFAULT 'TND',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        validated_at TIMESTAMPTZ
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS invoice_items (
                        id BIGSERIAL PRIMARY KEY,
                        invoice_id BIGINT NOT NULL REFERENCES invoices(id) ON DELETE CASCADE,
                        description TEXT,
                        quantity NUMERIC(18, 3) DEFAULT 1,
                        unit_price NUMERIC(18, 3) DEFAULT 0,
                        total NUMERIC(18, 3) DEFAULT 0,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS validation_events (
                        id BIGSERIAL PRIMARY KEY,
                        invoice_id BIGINT NOT NULL REFERENCES invoices(id) ON DELETE CASCADE,
                        user_email TEXT,
                        action TEXT NOT NULL,
                        before_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        after_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_pg_invoices_status ON invoices(status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_pg_invoices_supplier_id ON invoices(supplier_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_pg_invoices_due_date ON invoices(due_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_pg_invoice_items_invoice_id ON invoice_items(invoice_id)")
            connection.commit()

    def create_invoice(
        self,
        filename: str,
        file_type: str,
        source_file_path: str,
        extracted_text: str,
        normalized_data: Dict[str, Any],
        raw_data: Dict[str, Any],
        model_used: str,
        confidence_score: float,
    ) -> Dict[str, Any]:
        stored_path = self._persist_file(source_file_path, filename)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                duplicate_candidates = self._find_duplicate_candidates(cursor, normalized_data)
                enriched_raw_data = {
                    **raw_data,
                    "duplicate_detection": {
                        "is_potential_duplicate": bool(duplicate_candidates),
                        "candidates": duplicate_candidates,
                    },
                }
                cursor.execute(
                    """
                    INSERT INTO invoices (
                        filename, file_type, file_path, extracted_text, normalized_json,
                        raw_json, model_used, confidence_score, status
                    )
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, 'draft')
                    RETURNING id
                    """,
                    (
                        filename,
                        file_type,
                        stored_path,
                        extracted_text,
                        json_dumps(normalized_data),
                        json_dumps(enriched_raw_data),
                        model_used,
                        confidence_score,
                    ),
                )
                invoice_id = int(cursor.fetchone()["id"])
                self._sync_structured_invoice(cursor, invoice_id, normalized_data)
                self._add_validation_event(cursor, invoice_id, "created_draft", {}, normalized_data)
            connection.commit()
        return self.get_invoice(invoice_id)

    def update_invoice_data(self, invoice_id: int, normalized_data: Dict[str, Any], status: str = "draft") -> Dict[str, Any]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                before_data = self._get_normalized_data(cursor, invoice_id)
                cursor.execute(
                    """
                    UPDATE invoices
                    SET normalized_json = %s::jsonb, status = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (json_dumps(normalized_data), status, invoice_id),
                )
                self._sync_structured_invoice(cursor, invoice_id, normalized_data)
                self._add_validation_event(cursor, invoice_id, f"updated_{status}", before_data, normalized_data)
            connection.commit()
        return self.get_invoice(invoice_id)

    def validate_invoice(
        self,
        invoice_id: int,
        normalized_data: Optional[Dict[str, Any]] = None,
        user_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                before_data = self._get_normalized_data(cursor, invoice_id)
                after_data = normalized_data if normalized_data is not None else before_data

                if normalized_data is None:
                    cursor.execute(
                        """
                        UPDATE invoices
                        SET status = 'validated', updated_at = NOW(), validated_at = NOW()
                        WHERE id = %s
                        """,
                        (invoice_id,),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE invoices
                        SET normalized_json = %s::jsonb, status = 'validated',
                            updated_at = NOW(), validated_at = NOW()
                        WHERE id = %s
                        """,
                        (json_dumps(normalized_data), invoice_id),
                    )
                    self._sync_structured_invoice(cursor, invoice_id, normalized_data)

                self._add_validation_event(cursor, invoice_id, "validated", before_data, after_data, user_email)
            connection.commit()
        return self.get_invoice(invoice_id)

    def list_invoices(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM invoices ORDER BY created_at DESC")
                return [self._row_to_invoice(row) for row in cursor.fetchall()]

    def get_invoice(self, invoice_id: int) -> Dict[str, Any]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM invoices WHERE id = %s", (invoice_id,))
                row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Invoice {invoice_id} not found")
        return self._row_to_invoice(row)

    def search_invoices(self, search="", invoice_name="", supplier="", company="",
                        tax_id="", status="", page=1, page_size=10):
        conditions, params = [], []
        groups = [
            (search, ["filename", "supplier_name", "customer_name", "invoice_number", "normalized_json::text", "extracted_text"]),
            (invoice_name, ["filename", "invoice_number"]),
            (supplier, ["supplier_name"]),
            (company, ["customer_name"]),
            (tax_id, ["normalized_json->>'vendor_tax_id'", "normalized_json->>'customer_tax_id'", "normalized_json->>'tax_id'"]),
        ]
        for value, columns in groups:
            if value.strip():
                conditions.append("(" + " OR ".join(f"{column} ILIKE %s" for column in columns) + ")")
                params.extend(["%" + value.strip() + "%"] * len(columns))
        if status.strip():
            conditions.append("status = %s")
            params.append(status.strip())
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        page, page_size = max(1, int(page or 1)), min(100, max(1, int(page_size or 10)))
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) AS count FROM invoices" + where, params)
                total = int(cursor.fetchone()["count"])
                cursor.execute("SELECT * FROM invoices" + where + " ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s",
                               params + [page_size, (page - 1) * page_size])
                rows = cursor.fetchall()
        return {"items": [self._row_to_invoice(row) for row in rows], "total": total,
                "page": page, "page_size": page_size, "total_pages": max(1, (total + page_size - 1) // page_size)}

    def list_suppliers(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        suppliers.*,
                        COALESCE(last_invoice.invoice_number, '') AS last_invoice_number,
                        COALESCE(last_invoice.invoice_date, '') AS last_invoice_date,
                        COALESCE(last_invoice.status, '') AS last_invoice_status,
                        COALESCE((
                            SELECT AVG(total_amount)
                            FROM invoices
                            WHERE invoices.supplier_id = suppliers.id
                        ), 0) AS average_amount
                    FROM suppliers
                    LEFT JOIN LATERAL (
                        SELECT invoice_number, invoice_date, status
                        FROM invoices
                        WHERE invoices.supplier_id = suppliers.id
                        ORDER BY COALESCE(NULLIF(invoice_date, ''), created_at::text) DESC, id DESC
                        LIMIT 1
                    ) AS last_invoice ON TRUE
                    ORDER BY suppliers.invoices_count DESC, suppliers.total_amount DESC, suppliers.name ASC
                    """
                )
                return [dict(row) for row in cursor.fetchall()]

    def get_invoice_items(self, invoice_id: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                if invoice_id is None:
                    cursor.execute("SELECT * FROM invoice_items ORDER BY invoice_id DESC, id ASC")
                else:
                    cursor.execute(
                        "SELECT * FROM invoice_items WHERE invoice_id = %s ORDER BY invoice_id DESC, id ASC",
                        (invoice_id,),
                    )
                return [dict(row) for row in cursor.fetchall()]

    def get_validation_events(self, invoice_id: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                if invoice_id is None:
                    cursor.execute("SELECT * FROM validation_events ORDER BY created_at DESC")
                else:
                    cursor.execute(
                        "SELECT * FROM validation_events WHERE invoice_id = %s ORDER BY created_at DESC",
                        (invoice_id,),
                    )
                rows = cursor.fetchall()

        events = []
        for row in rows:
            payload = dict(row)
            payload["before_data"] = payload.pop("before_json") or {}
            payload["after_data"] = payload.pop("after_json") or {}
            events.append(payload)
        return events

    def get_dashboard_stats(self) -> Dict[str, Any]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) AS total_invoices,
                        SUM(CASE WHEN status = 'validated' THEN 1 ELSE 0 END) AS validated_invoices,
                        SUM(CASE WHEN status <> 'validated' THEN 1 ELSE 0 END) AS draft_invoices,
                        COALESCE(SUM(total_amount), 0) AS total_amount,
                        COALESCE(AVG(confidence_score), 0) AS avg_confidence
                    FROM invoices
                    """
                )
                totals = cursor.fetchone() or {}
                cursor.execute(
                    """
                    SELECT id, name, invoices_count, total_amount
                    FROM suppliers
                    ORDER BY invoices_count DESC, total_amount DESC, name ASC
                    LIMIT 5
                    """
                )
                top_suppliers = cursor.fetchall()
                cursor.execute(
                    """
                    SELECT id, invoice_number, supplier_name, customer_name, due_date, invoice_date, total_amount, currency, status
                    FROM invoices
                    WHERE COALESCE(due_date, '') <> ''
                    ORDER BY COALESCE(due_date, invoice_date) ASC
                    LIMIT 8
                    """
                )
                due_soon = cursor.fetchall()
                cursor.execute(
                    """
                    SELECT TO_CHAR(created_at, 'YYYY-MM') AS month, COALESCE(SUM(total_amount), 0) AS amount, COUNT(*) AS count
                    FROM invoices
                    GROUP BY TO_CHAR(created_at, 'YYYY-MM')
                    ORDER BY month ASC
                    """
                )
                monthly_amounts = cursor.fetchall()

        return {
            "totals": dict(totals),
            "top_suppliers": [dict(row) for row in top_suppliers],
            "due_soon": [dict(row) for row in due_soon],
            "monthly_amounts": [dict(row) for row in monthly_amounts],
        }

    def _sync_structured_invoice(self, cursor, invoice_id: int, normalized_data: Dict[str, Any]):
        supplier_name = (
            self._as_text(normalized_data.get("vendor_name"))
            or self._as_text(normalized_data.get("supplier_name"))
            or "Inconnu"
        )
        supplier_id = self._upsert_supplier(
            cursor,
            name=supplier_name,
            tax_id=self._as_text(normalized_data.get("tax_id") or normalized_data.get("vendor_tax_id")),
            address=self._as_text(normalized_data.get("vendor_address")),
            phone=self._as_text(normalized_data.get("vendor_phone") or normalized_data.get("supplier_phone")),
        )

        cursor.execute(
            """
            UPDATE invoices
            SET invoice_number = %s, supplier_id = %s, supplier_name = %s, customer_name = %s,
                invoice_date = %s, due_date = %s, subtotal = %s, tax_amount = %s,
                stamp_duty = %s, total_amount = %s, currency = %s, updated_at = NOW()
            WHERE id = %s
            """,
            (
                self._as_text(normalized_data.get("invoice_number")),
                supplier_id,
                supplier_name,
                self._as_text(normalized_data.get("customer_name")),
                self._as_text(normalized_data.get("date") or normalized_data.get("invoice_date")),
                self._as_text(normalized_data.get("due_date")),
                self._as_float(normalized_data.get("subtotal")),
                self._as_float(normalized_data.get("tax_amount")),
                self._as_float(normalized_data.get("stamp_duty")),
                self._as_float(normalized_data.get("total_amount")),
                self._as_text(normalized_data.get("currency")) or "TND",
                invoice_id,
            ),
        )

        cursor.execute("DELETE FROM invoice_items WHERE invoice_id = %s", (invoice_id,))
        for item in normalized_data.get("items") or []:
            if not isinstance(item, dict):
                continue
            cursor.execute(
                """
                INSERT INTO invoice_items (invoice_id, description, quantity, unit_price, total)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    invoice_id,
                    self._as_text(item.get("description") or item.get("designation")),
                    self._as_float(item.get("quantity")) or 1,
                    self._as_float(item.get("unit_price")),
                    self._as_float(item.get("total") or item.get("total_price")),
                ),
            )
        self._refresh_supplier_totals(cursor)

    def _find_duplicate_candidates(self, cursor, normalized_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        invoice_number = self._as_text(normalized_data.get("invoice_number"))
        supplier_name = (
            self._as_text(normalized_data.get("vendor_name"))
            or self._as_text(normalized_data.get("supplier_name"))
            or self._as_text(normalized_data.get("customer_name"))
        )
        total_amount = self._as_float(normalized_data.get("total_amount"))

        if not invoice_number and not supplier_name and not total_amount:
            return []

        cursor.execute(
            """
            SELECT
                id,
                invoice_number,
                supplier_name,
                customer_name,
                invoice_date,
                total_amount,
                currency,
                status,
                CASE
                    WHEN %s <> '' AND invoice_number = %s THEN 'same_invoice_number'
                    WHEN %s <> '' AND LOWER(COALESCE(supplier_name, '')) = LOWER(%s)
                         AND ABS(COALESCE(total_amount, 0) - %s) < 0.01 THEN 'same_supplier_and_total'
                    WHEN %s > 0 AND ABS(COALESCE(total_amount, 0) - %s) < 0.01 THEN 'same_total'
                    ELSE 'possible_match'
                END AS match_reason
            FROM invoices
            WHERE
                (%s <> '' AND invoice_number = %s)
                OR (
                    %s <> ''
                    AND LOWER(COALESCE(supplier_name, '')) = LOWER(%s)
                    AND ABS(COALESCE(total_amount, 0) - %s) < 0.01
                )
                OR (%s > 0 AND ABS(COALESCE(total_amount, 0) - %s) < 0.01)
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (
                invoice_number,
                invoice_number,
                supplier_name,
                supplier_name,
                total_amount,
                total_amount,
                total_amount,
                invoice_number,
                invoice_number,
                supplier_name,
                supplier_name,
                total_amount,
                total_amount,
                total_amount,
            ),
        )
        return [dict(row) for row in cursor.fetchall()]

    def _upsert_supplier(self, cursor, name: str, tax_id: str = "", address: str = "", phone: str = "") -> int:
        cursor.execute(
            """
            INSERT INTO suppliers (name, tax_id, address, phone, invoices_count, total_amount)
            VALUES (%s, %s, %s, %s, 0, 0)
            ON CONFLICT(name) DO UPDATE SET
                tax_id = COALESCE(NULLIF(EXCLUDED.tax_id, ''), suppliers.tax_id),
                address = COALESCE(NULLIF(EXCLUDED.address, ''), suppliers.address),
                phone = COALESCE(NULLIF(EXCLUDED.phone, ''), suppliers.phone),
                updated_at = NOW()
            RETURNING id
            """,
            (name, tax_id, address, phone),
        )
        return int(cursor.fetchone()["id"])

    def _refresh_supplier_totals(self, cursor):
        cursor.execute(
            """
            UPDATE suppliers
            SET invoices_count = source.invoices_count,
                total_amount = source.total_amount,
                updated_at = NOW()
            FROM (
                SELECT supplier_id, COUNT(*) AS invoices_count, COALESCE(SUM(total_amount), 0) AS total_amount
                FROM invoices
                WHERE supplier_id IS NOT NULL
                GROUP BY supplier_id
            ) AS source
            WHERE suppliers.id = source.supplier_id
            """
        )

    def _get_normalized_data(self, cursor, invoice_id: int) -> Dict[str, Any]:
        cursor.execute("SELECT normalized_json FROM invoices WHERE id = %s", (invoice_id,))
        row = cursor.fetchone()
        if not row:
            return {}
        value = row.get("normalized_json") or {}
        if isinstance(value, dict):
            return value
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {}

    def _add_validation_event(
        self,
        cursor,
        invoice_id: int,
        action: str,
        before_data: Dict[str, Any],
        after_data: Dict[str, Any],
        user_email: Optional[str] = None,
    ):
        cursor.execute(
            """
            INSERT INTO validation_events (invoice_id, user_email, action, before_json, after_json)
            VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
            """,
            (
                invoice_id,
                user_email,
                action,
                json_dumps(before_data),
                json_dumps(after_data),
            ),
        )

    def _persist_file(self, source_file_path: str, filename: str) -> str:
        return self.files.put(source_file_path)

    def _row_to_invoice(self, row) -> Dict[str, Any]:
        payload = dict(row)
        payload["normalized_data"] = payload.pop("normalized_json") or {}
        payload["raw_data"] = payload.pop("raw_json") or {}
        payload["duplicate_detection"] = payload["raw_data"].get(
            "duplicate_detection",
            {"is_potential_duplicate": False, "candidates": []},
        )
        payload["structured_data"] = {
            "invoice_number": payload.get("invoice_number"),
            "supplier_id": payload.get("supplier_id"),
            "supplier_name": payload.get("supplier_name"),
            "customer_name": payload.get("customer_name"),
            "invoice_date": payload.get("invoice_date"),
            "due_date": payload.get("due_date"),
            "subtotal": payload.get("subtotal"),
            "tax_amount": payload.get("tax_amount"),
            "stamp_duty": payload.get("stamp_duty"),
            "total_amount": payload.get("total_amount"),
            "currency": payload.get("currency"),
        }
        return payload

    def _as_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _as_float(self, value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.replace(" ", "").replace(",", ".")
            try:
                return float("".join(char for char in cleaned if char.isdigit() or char in ".-"))
            except ValueError:
                return 0.0
        return 0.0
