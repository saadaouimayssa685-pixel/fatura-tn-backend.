import tempfile
import unittest
from pathlib import Path

from invoice_store import InvoiceStore


class InvoiceStoreEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.store = InvoiceStore(str(root / "test.sqlite3"), str(root / "documents"))
        self.source = root / "invoice.txt"
        self.source.write_text("Invoice test", encoding="utf-8")

    def create(self, fields):
        return self.store.create_invoice(
            filename="invoice.txt", file_type="text", source_file_path=str(self.source),
            extracted_text="Invoice test", normalized_data=fields, raw_data={},
            model_used="test", confidence_score=0,
        )

    def test_missing_values_survive_create_update_validate_and_reopen(self):
        fields = {
            "invoice_number": "TEST-NULL", "vendor_name": "Supplier",
            "subtotal": None, "tax_amount": None, "total_amount": None,
            "items": [{"description": "Product", "quantity": None,
                       "unit_price": "6,000", "total": None, "total_price": 999}],
        }
        invoice = self.create(fields)
        self.store.update_invoice_data(invoice["id"], fields)
        self.store.validate_invoice(invoice["id"], fields)
        reopened = InvoiceStore(self.store.db_path, self.store.documents_dir)
        saved = reopened.get_invoice(invoice["id"])
        for key in ("subtotal", "tax_amount", "total_amount"):
            self.assertIsNone(saved[key])
            self.assertIsNone(saved["normalized_data"][key])
        item = reopened.get_invoice_items(invoice["id"])[0]
        self.assertIsNone(item["quantity"])
        self.assertIsNone(item["total"])
        self.assertEqual(item["unit_price"], 6)
        self.assertEqual(saved["status"], "validated")

    def test_real_zero_is_preserved_instead_of_using_alias_or_default(self):
        invoice = self.create({
            "vendor_name": "Supplier", "total_amount": 0, "tax_amount": 0,
            "items": [{"description": "Free sample", "quantity": 0,
                       "unit_price": 0, "total": 0, "total_price": 999}],
        })
        item = self.store.get_invoice_items(invoice["id"])[0]
        self.assertEqual([item[key] for key in ("quantity", "unit_price", "total")], [0, 0, 0])
        self.assertEqual(invoice["total_amount"], 0)

    def test_client_is_not_used_as_supplier(self):
        invoice = self.create({"customer_name": "Customer only"})
        self.assertEqual(invoice["customer_name"], "Customer only")
        self.assertEqual(invoice["supplier_name"], "Inconnu")

    def test_partial_form_save_and_validation_keep_product_rows(self):
        fields = {
            "vendor_name": "Supplier", "vendor_email": "supplier@example.test",
            "total_amount": 12.6,
            "items": [
                {"description": "Roller", "quantity": 1, "unit_price": 6, "total": 6, "price_basis": "TTC"},
                {"description": "Brush", "quantity": 1, "unit_price": 6, "total": 6, "price_basis": "TTC"},
            ],
        }
        invoice = self.create(fields)
        saved = self.store.update_invoice_data(invoice["id"], {"vendor_name": "Corrected"})
        self.assertEqual(saved["normalized_data"]["items"], fields["items"])
        validated = self.store.validate_invoice(invoice["id"], {"invoice_number": "00633"})
        self.assertEqual(validated["normalized_data"]["items"], fields["items"])
        self.assertEqual(validated["normalized_data"]["vendor_email"], fields["vendor_email"])
        self.assertEqual(len(self.store.get_invoice_items(invoice["id"])), 2)
        cleared = self.store.update_invoice_data(invoice["id"], {"items": [], "vendor_email": None})
        self.assertEqual(cleared["normalized_data"]["items"], [])
        self.assertIsNone(cleared["normalized_data"]["vendor_email"])
        self.assertEqual(self.store.get_invoice_items(invoice["id"]), [])

    def test_optional_number_rejects_invalid_or_nonfinite_input(self):
        for value in (None, "", "unknown", "19%", True, "NaN", "inf"):
            with self.subTest(value=value):
                self.assertIsNone(self.store._optional_number(value))
        self.assertEqual(self.store._optional_number("12 345,600"), 12345.6)


if __name__ == "__main__":
    unittest.main()
