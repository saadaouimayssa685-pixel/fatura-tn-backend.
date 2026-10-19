import ast
import re
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock

from cloud_invoice import parse_visual_invoice, value_in_quote
from invoice_rules import extract_invoice_table_text, filter_invoice_line_items
from models import InvoiceData


def pipeline_functions():
    names = {"safe_float", "normalize_money_amount", "normalize_invoice_data", "field_has_value",
             "_canonical_identifier", "_canonical_date", "_money_values_in_text", "value_has_ocr_support",
             "enforce_evidence_guard", "validate_normalized_invoice_fields", "analyze_invoice_fields"}
    tree = ast.parse(Path("main.py").read_text(encoding="utf-8-sig"))
    nodes = [node for node in tree.body if (isinstance(node, ast.FunctionDef) and node.name in names)
             or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                 and t.id in {"EVIDENCE_GUARDED_FIELDS", "AGENT_POLICY"} for t in node.targets))]
    scope = dict(re=re, Any=Any, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple, InvoiceData=InvoiceData,
                 extract_invoice_table_text=extract_invoice_table_text, filter_invoice_line_items=filter_invoice_line_items)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "main.py", "exec"), scope)
    return scope


class HybridExtractionTests(unittest.TestCase):
    def test_groq_abstention_is_not_replaced_with_regex_amount(self):
        rules = {"fields": {"subtotal": 12.6, "total_amount": 12.6, "currency": "TND"}, "items": []}
        data = InvoiceData(additional_data={"ollama_result": {"subtotal": None, "total_amount": 12.6}})
        text = "TOTAL TTC 12.600 TND"
        normalized = self.scope["normalize_invoice_data"](data, text, rules)
        result, _ = self.scope["enforce_evidence_guard"](normalized, text, {}, rules)
        self.assertIsNone(result["subtotal"])
        self.assertEqual(result["total_amount"], 12.6)
        self.assertNotIn("_llm_abstained_fields", result)

    def setUp(self):
        self.scope = pipeline_functions()
        self.rules = {"fields": {"invoice_number": "WRONG", "subtotal": 6, "tax_amount": 19,
                                 "tax_rate": "19%", "total_amount": 12600, "currency": "TND"},
                      "items": [{"description": "Telephone", "quantity": 1, "unit_price": 2000, "total": 2000}]}

    def test_cloud_abstention_survives_rules_and_guard(self):
        data = InvoiceData(additional_data={"cloud_result": {"invoice_number": "00633", "total_amount": 12.6, "items": []}})
        text = "Facture 00633\nTotal HT 6.000\nTVA 19%\nMontant TTC 12.600"
        normalized = self.scope["normalize_invoice_data"](data, text, self.rules)
        result, _ = self.scope["enforce_evidence_guard"](normalized, text, {}, self.rules, {})
        self.assertEqual(result["invoice_number"], "00633")
        self.assertEqual(result["total_amount"], 12.6)
        self.assertIsNone(result["subtotal"])
        self.assertIsNone(result["tax_amount"])
        self.assertEqual(result["tax_rate"], "")
        self.assertEqual(result["items"], [])

    def test_cloud_rows_are_completed_by_verified_ocr_rows(self):
        rules = {"fields": {"total_amount": 12.6, "currency": "TND"}, "items": [
            {"reference": "A-1", "description": "Rouleau Cristal", "quantity": 1,
             "unit_price": 6, "total": 6, "source": "ocr_table_rules"},
            {"reference": "A-2", "description": "Pinceau rond", "quantity": 1,
             "unit_price": 6, "total": 6, "source": "ocr_table_rules"},
        ]}
        cloud_row = {**rules["items"][0], "source": "gemini_visual",
                     "quote": "A-1 Rouleau Cristal 1 6.000 6.000", "page": 1}
        data = InvoiceData(additional_data={"cloud_result": {
            "total_amount": 12.6, "items": [cloud_row]}})
        text = ("A-1 Rouleau Cristal 1 6.000 6.000\n"
                "A-2 Pinceau rond 1 6.000 6.000\nMontant TTC 12.600")
        normalized = self.scope["normalize_invoice_data"](data, text, rules)
        result, _ = self.scope["enforce_evidence_guard"](normalized, text, {}, rules, {})
        self.assertEqual([row["description"] for row in result["items"]],
                         ["Rouleau Cristal", "Pinceau rond"])

    def test_visual_payment_is_retained_but_requires_review(self):
        proof = {"payment_method": {"value": "Especes", "text": "Paye en Especes",
                 "page_number": 1, "ocr_corroborated": False, "requires_review": True}}
        result, evidence = self.scope["enforce_evidence_guard"](
            {"payment_method": "Especes", "vendor_tax_id": "", "tax_id": "WRONG"}, "", {}, self.rules, proof)
        self.assertEqual(result["payment_method"], "Especes")
        self.assertEqual(result["tax_id"], "")
        self.assertFalse(evidence["payment_method"]["supported"])
        self.assertEqual(evidence["payment_method"]["source"]["type"], "visual_model")
        issues = self.scope["validate_normalized_invoice_fields"](result, evidence)
        self.assertIn("visual_review_payment_method", [issue["code"] for issue in issues])

    def test_legacy_gemini_choice_is_a_local_alias(self):
        cloud, local = Mock(), Mock()
        local.analyze_invoice_text.return_value = None
        self.scope.update(ai_analyzer=cloud, ollama_analyzer=local)
        data, model, raw = self.scope["analyze_invoice_fields"]("OCR", "gemini", {}, self.rules, "invoice.png")
        self.assertIsNone(data)
        self.assertEqual(model, "fallback_regex")
        self.assertEqual(raw, {})
        local.analyze_invoice_text.assert_called_once()
        cloud.analyze_invoice_text.assert_not_called()

    def test_explicit_local_never_calls_cloud(self):
        cloud, local = Mock(), Mock()
        local.model_name = "local-test"
        local.analyze_invoice_text.return_value = InvoiceData()
        self.scope.update(ai_analyzer=cloud, ollama_analyzer=local)
        _, model, _ = self.scope["analyze_invoice_fields"]("OCR", "local", {}, self.rules, "invoice.png")
        self.assertEqual(model, "ollama:local-test")
        cloud.analyze_invoice_text.assert_not_called()
        self.assertNotIn("document_path", local.analyze_invoice_text.call_args.kwargs)

    def test_ttc_lines_keep_their_basis_and_missing_ht_is_not_inferred(self):
        rows = [dict(description="Rouleau Cristal", quantity=1, unit_price=6, total=6, price_basis="TTC",
                     source="gemini_visual", quote="Rouleau Cristal 1 6.000 6.000", page=1, requires_review=False)]
        result, proof = self.scope["enforce_evidence_guard"](
            {"items": rows, "total_amount": 6.6}, "Montant TTC 6.600", {}, self.rules, {})
        self.assertEqual(result["items"][0]["price_basis"], "TTC")
        issues = self.scope["validate_normalized_invoice_fields"](result, proof)
        issue = next(i for i in issues if i["code"] == "missing_subtotal_for_total")
        self.assertEqual(issue["severity"], "warning")

    def test_arabic_names_and_products_are_not_discarded(self):
        name = "\u0634\u0631\u0643\u0629 \u0627\u0644\u0646\u0648\u0631"
        self.assertTrue(self.scope["value_has_ocr_support"]("vendor_name", name, name))
        product = "\u0648\u0631\u0642 \u0623\u0628\u064a\u0636"
        accepted, _ = filter_invoice_line_items(
            [dict(description=product, quantity=1, unit_price=6, total=6)], product + " 1 6.000 6.000", 6)
        self.assertEqual(len(accepted), 1)

    def test_printed_dinar_is_not_used_as_iso_currency(self):
        result = parse_visual_invoice({"currency": "DINAR", "evidence": [dict(field="currency", quote="DINAR", page=1)]},
                                      "DINAR", 1, True, "test")
        self.assertIsNone(result.currency)

    def test_pdf_service_row_symbols_and_placeholders(self):
        self.assertTrue(self.scope["value_has_ocr_support"]("currency", "EUR", "480,00 \u20ac"))
        self.assertFalse(value_in_quote("total_amount", 12, "Total TTC 12 000,000"))
        self.assertTrue(value_in_quote("total_amount", 12000, "Total TTC 12 000,000"))
        text = "mon service 3 100,00 300,00\nTotal TTC: 480,00 \u20ac\n[Adresse]"
        result = parse_visual_invoice({
            "currency": "EUR", "vendor_address": "[Adresse]",
            "items": [dict(description="mon service", quantity=3, unit_price=100, total=300,
                           price_basis="unknown", quote="mon service 3 100,00 300,00", page=1)],
            "evidence": [dict(field="currency", quote="Total TTC: 480,00 \u20ac", page=1),
                         dict(field="vendor_address", quote="[Adresse]", page=1)],
        }, text, 1, True, "test")
        self.assertEqual(result.currency, "EUR")
        self.assertIsNone(result.supplier.address)
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].quantity, 3)
        self.assertEqual(result.items[0].unit_price, 100)
        self.assertEqual(result.items[0].total_price, 300)


if __name__ == "__main__":
    unittest.main()
