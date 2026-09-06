import json
import os
import shutil
import sqlite3
from typing import Any, Dict, List, Optional


class InvoiceStore:
    """Local SQLite store for invoice automation records."""

    def __init__(self, db_path: str = "invoice_storage/invoices.sqlite3", documents_dir: str = "invoice_storage/documents"):
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
                CREATE TABLE IF NOT EXISTS invoices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    extracted_text TEXT,
                    normalized_json TEXT NOT NULL DEFAULT '{}',
                    raw_json TEXT NOT NULL DEFAULT '{}',
                    model_used TEXT,
                    confidence_score REAL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'draft',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    validated_at TEXT
                )
                """
            )
            self._ensure_invoice_columns(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS suppliers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    tax_id TEXT,
                    address TEXT,
                    phone TEXT,
                    invoices_count INTEGER NOT NULL DEFAULT 0,
                    total_amount REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS invoice_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    invoice_id INTEGER NOT NULL,
                    description TEXT,
                    quantity REAL DEFAULT 1,
                    unit_price REAL DEFAULT 0,
                    total REAL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(invoice_id) REFERENCES invoices(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    invoice_id INTEGER NOT NULL,
                    user_email TEXT,
                    action TEXT NOT NULL,
                    before_json TEXT NOT NULL DEFAULT '{}',
                    after_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(invoice_id) REFERENCES invoices(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_invoices_status ON invoices(status)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_invoices_file_type ON invoices(file_type)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_invoices_supplier_id ON invoices(supplier_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_invoices_supplier_name ON invoices(supplier_name)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_invoices_invoice_number ON invoices(invoice_number)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_invoices_invoice_date ON invoices(invoice_date)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_invoices_due_date ON invoices(due_date)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_invoice_items_invoice_id ON invoice_items(invoice_id)")
            connection.execute("PRAGMA optimize")
            self._sync_existing_invoices(connection)
            connection.commit()

    def _ensure_invoice_columns(self, connection):
        existing_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(invoices)").fetchall()
        }
        required_columns = {
            "invoice_number": "TEXT",
            "supplier_id": "INTEGER",
            "supplier_name": "TEXT",
            "customer_name": "TEXT",
            "invoice_date": "TEXT",
            "due_date": "TEXT",
            "subtotal": "REAL DEFAULT 0",
            "tax_amount": "REAL DEFAULT 0",
            "stamp_duty": "REAL DEFAULT 0",
            "total_amount": "REAL DEFAULT 0",
            "currency": "TEXT DEFAULT 'TND'",
        }

        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                connection.execute(f"ALTER TABLE invoices ADD COLUMN {column_name} {column_type}")

    def _sync_existing_invoices(self, connection):
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id, normalized_json FROM invoices WHERE invoice_number IS NULL OR total_amount IS NULL OR total_amount = 0"
        ).fetchall()
        for row in rows:
            try:
                normalized_data = json.loads(row["normalized_json"] or "{}")
            except json.JSONDecodeError:
                normalized_data = {}
            self._sync_structured_invoice(connection, int(row["id"]), normalized_data)
        connection.row_factory = None

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
            cursor = connection.execute(
                """
                INSERT INTO invoices (
                    filename, file_type, file_path, extracted_text, normalized_json,
                    raw_json, model_used, confidence_score, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'draft')
                """,
                (
                    filename,
                    file_type,
                    stored_path,
                    extracted_text,
                    json.dumps(normalized_data, ensure_ascii=False),
                    json.dumps(raw_data, ensure_ascii=False),
                    model_used,
                    confidence_score,
                ),
            )
            invoice_id = int(cursor.lastrowid)
            self._sync_structured_invoice(connection, invoice_id, normalized_data)
            self._add_validation_event(
                connection=connection,
                invoice_id=invoice_id,
                action="created_draft",
                before_data={},
                after_data=normalized_data,
            )
            connection.commit()

        return self.get_invoice(invoice_id)

    def update_invoice_data(self, invoice_id: int, normalized_data: Dict[str, Any], status: str = "draft") -> Dict[str, Any]:
        with self._connect() as connection:
            before_data = self._get_normalized_data(connection, invoice_id)
            connection.execute(
                """
                UPDATE invoices
                SET normalized_json = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (json.dumps(normalized_data, ensure_ascii=False), status, invoice_id),
            )
            self._sync_structured_invoice(connection, invoice_id, normalized_data)
            self._add_validation_event(
                connection=connection,
                invoice_id=invoice_id,
                action=f"updated_{status}",
                before_data=before_data,
                after_data=normalized_data,
            )
            connection.commit()

        return self.get_invoice(invoice_id)

    def validate_invoice(
        self,
        invoice_id: int,
        normalized_data: Optional[Dict[str, Any]] = None,
        user_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._connect() as connection:
            before_data = self._get_normalized_data(connection, invoice_id)
            if normalized_data is None:
                connection.execute(
                    """
                    UPDATE invoices
                    SET status = 'validated', updated_at = CURRENT_TIMESTAMP, validated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (invoice_id,),
                )
                normalized_data = before_data
            else:
                connection.execute(
                    """
                    UPDATE invoices
                    SET normalized_json = ?, status = 'validated',
                        updated_at = CURRENT_TIMESTAMP, validated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (json.dumps(normalized_data, ensure_ascii=False), invoice_id),
                )
                self._sync_structured_invoice(connection, invoice_id, normalized_data)
            self._add_validation_event(
                connection=connection,
                invoice_id=invoice_id,
                user_email=user_email,
                action="validated",
                before_data=before_data,
                after_data=normalized_data,
            )
            connection.commit()

        return self.get_invoice(invoice_id)

    def list_invoices(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM invoices ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_invoice(row) for row in rows]

    def search_invoices(
        self,
        search: str = "",
        invoice_name: str = "",
        supplier: str = "",
        company: str = "",
        tax_id: str = "",
        status: str = "",
        page: int = 1,
        page_size: int = 10,
    ) -> Dict[str, Any]:
        where_parts: List[str] = []
        params: List[Any] = []

        normalized_search = (search or "").strip()
        if normalized_search:
            like_value = f"%{normalized_search}%"
            where_parts.append(
                """
                (
                    filename LIKE ?
                    OR supplier_name LIKE ?
                    OR customer_name LIKE ?
                    OR invoice_number LIKE ?
                    OR normalized_json LIKE ?
                    OR extracted_text LIKE ?
                )
                """
            )
            params.extend([like_value] * 6)

        normalized_invoice_name = (invoice_name or "").strip()
        if normalized_invoice_name:
            like_value = f"%{normalized_invoice_name}%"
            where_parts.append("(filename LIKE ? OR invoice_number LIKE ?)")
            params.extend([like_value, like_value])

        normalized_supplier = (supplier or "").strip()
        if normalized_supplier:
            where_parts.append("supplier_name LIKE ?")
            params.append(f"%{normalized_supplier}%")

        normalized_company = (company or "").strip()
        if normalized_company:
            where_parts.append("(customer_name LIKE ? OR normalized_json LIKE ?)")
            like_value = f"%{normalized_company}%"
            params.extend([like_value, like_value])

        normalized_tax_id = (tax_id or "").strip()
        if normalized_tax_id:
            where_parts.append("(normalized_json LIKE ? OR extracted_text LIKE ?)")
            like_value = f"%{normalized_tax_id}%"
            params.extend([like_value, like_value])

        normalized_status = (status or "").strip()
        if normalized_status:
            where_parts.append("status = ?")
            params.append(normalized_status)

        where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        safe_page = max(1, int(page or 1))
        safe_page_size = min(100, max(1, int(page_size or 10)))
        offset = (safe_page - 1) * safe_page_size

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM invoices {where_sql}",
                    tuple(params),
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT *
                FROM invoices
                {where_sql}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                tuple(params + [safe_page_size, offset]),
            ).fetchall()

        total_pages = max(1, (total + safe_page_size - 1) // safe_page_size)
        return {
            "items": [self._row_to_invoice(row) for row in rows],
            "total": total,
            "page": safe_page,
            "page_size": safe_page_size,
            "total_pages": total_pages,
        }

    def get_invoice(self, invoice_id: int) -> Dict[str, Any]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM invoices WHERE id = ?",
                (invoice_id,),
            ).fetchone()

        if row is None:
            raise ValueError(f"Invoice {invoice_id} not found")

        return self._row_to_invoice(row)

    def list_suppliers(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    suppliers.*,
                    COALESCE((
                        SELECT invoice_number
                        FROM invoices
                        WHERE invoices.supplier_id = suppliers.id
                        ORDER BY COALESCE(invoice_date, created_at) DESC, id DESC
                        LIMIT 1
                    ), '') AS last_invoice_number,
                    COALESCE((
                        SELECT COALESCE(invoice_date, created_at)
                        FROM invoices
                        WHERE invoices.supplier_id = suppliers.id
                        ORDER BY COALESCE(invoice_date, created_at) DESC, id DESC
                        LIMIT 1
                    ), '') AS last_invoice_date,
                    COALESCE((
                        SELECT status
                        FROM invoices
                        WHERE invoices.supplier_id = suppliers.id
                        ORDER BY COALESCE(invoice_date, created_at) DESC, id DESC
                        LIMIT 1
                    ), '') AS last_invoice_status,
                    COALESCE((
                        SELECT AVG(total_amount)
                        FROM invoices
                        WHERE invoices.supplier_id = suppliers.id
                    ), 0) AS average_amount
                FROM suppliers
                ORDER BY suppliers.invoices_count DESC, suppliers.total_amount DESC, suppliers.name ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_validation_events(self, invoice_id: Optional[int] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM validation_events"
        params: tuple[Any, ...] = ()
        if invoice_id is not None:
            query += " WHERE invoice_id = ?"
            params = (invoice_id,)
        query += " ORDER BY created_at DESC"

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, params).fetchall()

        events = []
        for row in rows:
            payload = dict(row)
            payload["before_data"] = json.loads(payload.pop("before_json") or "{}")
            payload["after_data"] = json.loads(payload.pop("after_json") or "{}")
            events.append(payload)
        return events

    def get_invoice_items(self, invoice_id: Optional[int] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM invoice_items"
        params: tuple[Any, ...] = ()
        if invoice_id is not None:
            query += " WHERE invoice_id = ?"
            params = (invoice_id,)
        query += " ORDER BY invoice_id DESC, id ASC"

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_dashboard_stats(self) -> Dict[str, Any]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            totals = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_invoices,
                    SUM(CASE WHEN status = 'validated' THEN 1 ELSE 0 END) AS validated_invoices,
                    SUM(CASE WHEN status <> 'validated' THEN 1 ELSE 0 END) AS draft_invoices,
                    COALESCE(SUM(total_amount), 0) AS total_amount,
                    COALESCE(AVG(confidence_score), 0) AS avg_confidence
                FROM invoices
                """
            ).fetchone()
            top_suppliers = connection.execute(
                """
                SELECT id, name, invoices_count, total_amount
                FROM suppliers
                ORDER BY invoices_count DESC, total_amount DESC, name ASC
                LIMIT 5
                """
            ).fetchall()
            due_soon = connection.execute(
                """
                SELECT id, invoice_number, supplier_name, customer_name, due_date, invoice_date, total_amount, currency, status
                FROM invoices
                WHERE COALESCE(due_date, invoice_date, '') <> ''
                ORDER BY COALESCE(due_date, invoice_date) ASC
                LIMIT 8
                """
            ).fetchall()
            monthly_amounts = connection.execute(
                """
                SELECT substr(created_at, 1, 7) AS month, COALESCE(SUM(total_amount), 0) AS amount, COUNT(*) AS count
                FROM invoices
                GROUP BY substr(created_at, 1, 7)
                ORDER BY month ASC
                """
            ).fetchall()

        return {
            "totals": dict(totals) if totals else {},
            "top_suppliers": [dict(row) for row in top_suppliers],
            "due_soon": [dict(row) for row in due_soon],
            "monthly_amounts": [dict(row) for row in monthly_amounts],
        }

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

    def _sync_structured_invoice(self, connection, invoice_id: int, normalized_data: Dict[str, Any]):
        supplier_name = (
            self._as_text(normalized_data.get("vendor_name"))
            or self._as_text(normalized_data.get("supplier_name"))
            or self._as_text(normalized_data.get("customer_name"))
            or "Inconnu"
        )
        customer_name = self._as_text(normalized_data.get("customer_name"))
        supplier_id = self._upsert_supplier(
            connection=connection,
            name=supplier_name,
            tax_id=self._as_text(normalized_data.get("tax_id") or normalized_data.get("vendor_tax_id")),
            address=self._as_text(normalized_data.get("vendor_address")),
            phone=self._as_text(normalized_data.get("vendor_phone") or normalized_data.get("supplier_phone")),
            total_amount=self._as_float(normalized_data.get("total_amount")),
        )

        connection.execute(
            """
            UPDATE invoices
            SET invoice_number = ?, supplier_id = ?, supplier_name = ?, customer_name = ?,
                invoice_date = ?, due_date = ?, subtotal = ?, tax_amount = ?,
                stamp_duty = ?, total_amount = ?, currency = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                self._as_text(normalized_data.get("invoice_number")),
                supplier_id,
                supplier_name,
                customer_name,
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

        connection.execute("DELETE FROM invoice_items WHERE invoice_id = ?", (invoice_id,))
        for item in normalized_data.get("items") or []:
            if not isinstance(item, dict):
                continue
            connection.execute(
                """
                INSERT INTO invoice_items (invoice_id, description, quantity, unit_price, total)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    invoice_id,
                    self._as_text(item.get("description") or item.get("designation")),
                    self._as_float(item.get("quantity")) or 1,
                    self._as_float(item.get("unit_price")),
                    self._as_float(item.get("total") or item.get("total_price")),
                ),
            )

        self._refresh_supplier_totals(connection)

    def _upsert_supplier(self, connection, name: str, tax_id: str = "", address: str = "", phone: str = "", total_amount: float = 0.0) -> int:
        connection.execute(
            """
            INSERT INTO suppliers (name, tax_id, address, phone, total_amount, invoices_count)
            VALUES (?, ?, ?, ?, ?, 0)
            ON CONFLICT(name) DO UPDATE SET
                tax_id = COALESCE(NULLIF(excluded.tax_id, ''), suppliers.tax_id),
                address = COALESCE(NULLIF(excluded.address, ''), suppliers.address),
                phone = COALESCE(NULLIF(excluded.phone, ''), suppliers.phone),
                updated_at = CURRENT_TIMESTAMP
            """,
            (name, tax_id, address, phone, total_amount),
        )
        row = connection.execute("SELECT id FROM suppliers WHERE name = ?", (name,)).fetchone()
        return int(row[0])

    def _refresh_supplier_totals(self, connection):
        connection.execute(
            """
            UPDATE suppliers
            SET invoices_count = (
                    SELECT COUNT(*) FROM invoices WHERE invoices.supplier_id = suppliers.id
                ),
                total_amount = (
                    SELECT COALESCE(SUM(total_amount), 0) FROM invoices WHERE invoices.supplier_id = suppliers.id
                ),
                updated_at = CURRENT_TIMESTAMP
            """
        )

    def _get_normalized_data(self, connection, invoice_id: int) -> Dict[str, Any]:
        row = connection.execute(
            "SELECT normalized_json FROM invoices WHERE id = ?",
            (invoice_id,),
        ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row[0] or "{}")
        except json.JSONDecodeError:
            return {}

    def _add_validation_event(
        self,
        connection,
        invoice_id: int,
        action: str,
        before_data: Dict[str, Any],
        after_data: Dict[str, Any],
        user_email: Optional[str] = None,
    ):
        connection.execute(
            """
            INSERT INTO validation_events (invoice_id, user_email, action, before_json, after_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                invoice_id,
                user_email,
                action,
                json.dumps(before_data, ensure_ascii=False),
                json.dumps(after_data, ensure_ascii=False),
            ),
        )

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

    def _row_to_invoice(self, row) -> Dict[str, Any]:
        payload = dict(row)
        payload["normalized_data"] = json.loads(payload.pop("normalized_json") or "{}")
        payload["raw_data"] = json.loads(payload.pop("raw_json") or "{}")
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
