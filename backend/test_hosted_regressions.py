import ast
import json
import re
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
from models import InvoiceData, PaymentInfo
from unittest.mock import MagicMock
from postgres_invoice_store import PostgresInvoiceStore
from postgres_invoice_store import json_dumps


class RegressionTests(unittest.TestCase):
    def test_columnar_quantity_is_not_defaulted_to_one(self):
        tree = ast.parse(Path("main.py").read_text(encoding="utf-8-sig"))
        names = {"safe_float", "extract_columnar_invoice_items"}
        scope = {"re": re, "List": List, "Dict": Dict, "Any": Any}
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
                                type_ignores=[]), "main.py", "exec"), scope)
        rows = scope["extract_columnar_invoice_items"](
            "Designation Qte PU TTC Montant\nRouleau Cristal 2 6.000 12.000\nPinceau rond N6/5 SAP 3 6.000 18.000")
        self.assertEqual([row["quantity"] for row in rows], [2, 3])
        self.assertEqual(rows[1]["description"], "Pinceau rond N6/5 SAP")

    def test_grounded_local_values_survive_rule_normalization(self):
        tree = ast.parse(Path("main.py").read_text(encoding="utf-8-sig"))
        names = {"safe_float", "normalize_money_amount", "normalize_invoice_data"}
        module = ast.Module(body=[node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
        scope = {"re": re, "Any": Any, "Dict": Dict, "Optional": Optional, "InvoiceData": InvoiceData}
        exec(compile(module, "main.py", "exec"), scope)
        grounded = {
            "invoice_number": "00633", "supplier_name": "BEN ABDELKADER",
            "total_amount": 1260.6, "stamp_duty": 0.6,
            "payment_method": "Especes", "amount_in_words": "Montant imprime",
            "items": [{"description": "Article verifie", "quantity": 1, "total": 6}],
        }
        data = InvoiceData(additional_data={"ollama_result": grounded},
            payment_info=PaymentInfo(method="Especes"))
        rules = {"fields": {"invoice_number": "WRONG", "vendor_name": "Adresse",
            "total_amount": 6, "stamp_duty": 1, "payment_method": "Cheque",
            "vendor_phone": "22 629 288"}, "items": []}
        result = scope["normalize_invoice_data"](data, "", rules)
        self.assertEqual(result["invoice_number"], "00633")
        self.assertEqual(result["vendor_name"], "BEN ABDELKADER")
        self.assertEqual(result["total_amount"], 1260.6)
        self.assertEqual(result["stamp_duty"], 0.6)
        self.assertEqual(result["payment_method"], "Especes")
        self.assertEqual(result["amount_in_words"], "Montant imprime")
        self.assertEqual(result["vendor_phone"], "22 629 288")
        self.assertEqual(result["items"], grounded["items"])

    def test_money_labels_accept_blank_lines_and_montant(self):
        tree = ast.parse(Path("main.py").read_text(encoding="utf-8-sig"))
        module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name in {"safe_float", "extract_labeled_money"}], type_ignores=[])
        scope = {"re": re, "List": List}
        exec(compile(module, "main.py", "exec"), scope)
        extract = scope["extract_labeled_money"]
        self.assertEqual(extract("Montant TTC\n\n12,600", [r"montant\s+ttc"]), 12.6)
        self.assertEqual(extract("TVA (19%)\n\n5,503", [r"\btva\b"]), 5.503)
        self.assertEqual(extract("TVA (19%)", [r"\btva\b"]), 0.0)
        self.assertEqual(extract("Désignation | Quantité | Total HT\n6,000", [r"total\s+ht"]), 0.0)

    def test_total_parser_skips_a_table_header(self):
        tree = ast.parse(Path("main.py").read_text(encoding="utf-8-sig"))
        names = {"safe_float", "extract_labeled_money"}
        module = ast.Module(
            body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
            type_ignores=[],
        )
        scope = {"re": re, "List": List}
        exec(compile(module, "main.py", "exec"), scope)

        text = """Désignation | Qte | Prix unitaire | Total HT
Régie Son 1 450.000 450.000
Total HT 4650.000
TVA 19% 883.500
TOTAL TTC 5534.100"""
        subtotal = scope["extract_labeled_money"](
            text, [r"\btotal\s+(?:net\s+)?h\.?\s*t\.?\b"], prefer="max"
        )
        self.assertEqual(subtotal, 4650.0)

    def test_invoice_header_identity_is_enough_when_totals_are_missed(self):
        tree = ast.parse(Path("main.py").read_text(encoding="utf-8-sig"))
        target = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "classify_invoice_text"
        )
        scope = {"re": re}
        exec(compile(ast.Module(body=[target], type_ignores=[]), "main.py", "exec"), scope)

        is_invoice, confidence, details = scope["classify_invoice_text"](
            "Factu re : K22/33\nDate : 08.09.2022\nMF : 1085144/P/A/M/000"
        )

        self.assertTrue(is_invoice)
        self.assertGreaterEqual(confidence, 0.82)
        self.assertTrue(details["header_identity_evidence"])

    def test_field_helpers_do_not_confuse_numbers(self):
        tree = ast.parse(Path("main.py").read_text(encoding="utf-8-sig"))
        names = {"extract_invoice_number", "clean_phone_number", "extract_phone_near_labels"}
        module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
        scope = {"re": re, "List": List}
        exec(compile(module, "main.py", "exec"), scope)
        text = "FACTURE DE TEST\nFOURNISSEUR: TEST\nMF: 1234567 A/M/000\nCLIENT: TEST\nFACTURE N: QA-2026-1001"
        self.assertEqual(scope["extract_invoice_number"](text), "QA-2026-1001")
        self.assertEqual(scope["extract_phone_near_labels"](text, ["client"]), "")
        self.assertEqual(scope["extract_phone_near_labels"]("Tel: 22 629 288", ["fournisseur"]), "22 629 288")

    def test_invalid_ocr_date_is_rejected_before_a_valid_date(self):
        tree = ast.parse(Path("main.py").read_text(encoding="utf-8-sig"))
        target = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "extract_invoice_date"
        )
        scope = {"re": re, "date": date}
        exec(compile(ast.Module(body=[target], type_ignores=[]), "main.py", "exec"), scope)

        text = "Date : 20-14-2019\nDate facture : 28-11-2019"
        self.assertEqual(scope["extract_invoice_date"](text), "28-11-2019")

    def test_client_and_vendor_tax_ids_remain_distinct(self):
        tree = ast.parse(Path("main.py").read_text(encoding="utf-8-sig"))
        names = {"clean_tax_identifier", "extract_party_hints"}
        module = ast.Module(
            body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
            type_ignores=[],
        )
        scope = {"re": re, "Dict": dict}
        exec(compile(module, "main.py", "exec"), scope)

        text = """BEN ABDELKADER
Nom du Client : BRIDGE IMMOBILERE
MF : 1362434/S
Adresse fournisseur
MF : 1547376 Z/N/C/000"""
        hints = scope["extract_party_hints"](text)

        self.assertEqual(hints["customer_tax_id"], "1362434/S")
        self.assertEqual(hints["vendor_tax_id"], "1547376Z/N/C/000")

    def test_header_address_belongs_to_vendor(self):
        tree = ast.parse(Path("main.py").read_text(encoding="utf-8-sig"))
        names = {"clean_tax_identifier", "extract_party_hints"}
        module = ast.Module(
            body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
            type_ignores=[],
        )
        scope = {"re": re, "Dict": dict}
        exec(compile(module, "main.py", "exec"), scope)

        text = """BEN ABDELKADER
Adresse : Rue Jawdet Al-Hayet Choutrana 1 Sidi Salah
Téléphone : 22 629 288
Nom du Client : BRIDGE IMMOBILIERE
MF : 1362434/S
ADRESSE:
MF : 1547376 Z/N/C/000"""
        hints = scope["extract_party_hints"](text)

        self.assertEqual(hints["vendor_address"], "Rue Jawdet Al-Hayet Choutrana 1 Sidi Salah")
        self.assertEqual(hints["customer_address"], "")

    def test_ocr_reference_artifact_does_not_hide_a_line_item(self):
        tree = ast.parse(Path("main.py").read_text(encoding="utf-8-sig"))
        names = {"safe_float", "extract_columnar_invoice_items"}
        module = ast.Module(
            body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
            type_ignores=[],
        )
        scope = {"re": re, "List": List, "Dict": dict, "Any": object}
        exec(compile(module, "main.py", "exec"), scope)

        items = scope["extract_columnar_invoice_items"](
            "Référence Désignation Qte PU TTC Montant\n"
            "[A-002018-00076a {Rouleau Antigoutte Cristal 1 6.000 6.000\n"
            ")A-002016-002754 | Pinceau rond N6/5 SAP 1 6.000 6.000"
        )

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["description"], "Rouleau Antigoutte Cristal")

        dotted_reference_items = scope["extract_columnar_invoice_items"](
            "Référence Désignation Qte PU TTC Montant\n"
            "A.002018-00076a {Rouleau Antigoutte Cristal 1 6.000 6.000\n"
            "A-002016-002754 Pinceau rond N6/5 SAP 1 6.000 6.000"
        )
        self.assertEqual(dotted_reference_items[0]["description"], "Rouleau Antigoutte Cristal")

    def test_search_is_parameterized_and_bounded(self):
        store = PostgresInvoiceStore.__new__(PostgresInvoiceStore)
        connection, cursor = MagicMock(), MagicMock()
        connection.__enter__.return_value = connection
        connection.cursor.return_value.__enter__.return_value = cursor
        store._connect = lambda: connection
        cursor.fetchone.return_value = {"count": 0}
        cursor.fetchall.return_value = []
        result = store.search_invoices(supplier="' OR 1=1 --", page=-1, page_size=999)
        self.assertEqual(result["page"], 1)
        self.assertEqual(result["page_size"], 100)
        sql, params = cursor.execute.call_args.args
        self.assertNotIn("' OR 1=1", sql)
        self.assertEqual(params[0], "%' OR 1=1 --%")

    def test_postgres_decimal_audit_payload_is_json_safe(self):
        payload = json.loads(json_dumps({"total": Decimal("12.600")}))
        self.assertEqual(payload["total"], 12.6)


if __name__ == "__main__":
    unittest.main()
