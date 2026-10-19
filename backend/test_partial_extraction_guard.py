"""Final guards for interrupted extraction; frozen OCR only, no external calls."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from models import InvoiceData
from test_hybrid_extraction import pipeline_functions


QA = Path(__file__).parent / "qa-results/application-smoke-20261012T083011Z"


class PartialExtractionGuardTests(unittest.TestCase):
    def setUp(self):
        self.scope = pipeline_functions()
        self.network = patch("requests.sessions.Session.request", side_effect=AssertionError("No network"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def check(self, text, rules, model=None, status=None):
        fields = self.scope["normalize_invoice_data"](model, text, rules, analysis_status=status)
        fields, proofs = self.scope["enforce_evidence_guard"](fields, text, {}, rules)
        issues = self.scope["validate_normalized_invoice_fields"](fields, proofs)
        return fields, proofs, issues

    def test_recorded_00791_wrong_ht_is_removed_without_replacing_it(self):
        invoice = json.loads((QA / "facture_00791-detail.json").read_text(encoding="utf-8"))["invoice"]
        original = copy.deepcopy(invoice)
        raw = invoice["raw_data"]
        rules = {"fields": raw["rule_candidates"]["field_values"],
                 "items": raw["rule_candidates"]["article_rows"]}
        self.assertEqual(invoice["normalized_data"]["subtotal"], 0.565)
        self.assertEqual(raw["llm"]["batches"][-1]["status"], "rate_limited")
        fields, proofs = self.scope["enforce_evidence_guard"](
            copy.deepcopy(invoice["normalized_data"]), invoice["extracted_text"], raw["field_evidence"], rules)
        self.assertIsNone(fields["subtotal"])
        self.assertFalse(proofs["subtotal"]["supported"])
        self.assertTrue(proofs["subtotal"]["requires_review"])
        self.assertEqual(proofs["subtotal"]["reason"], "ambiguous_adjacent_monetary_cells")
        self.assertEqual(proofs["subtotal"]["rejected_value"], 0.565)
        self.assertEqual(proofs["subtotal"]["source"]["line_ids"], [21, 22])
        self.assertIn("28,961", proofs["subtotal"]["source"]["text"])
        self.assertEqual(fields["total_amount"], 37.02)
        self.assertEqual(fields["stamp_duty"], 0.831)
        self.assertIsNone(fields["tax_amount"])
        self.assertEqual(fields["taxes"], [])
        self.assertEqual(fields["items"], [])
        self.assertFalse(fields["vendor_name"])
        self.assertEqual(fields["customer_name"], "STE DE PROMOTION")
        issues = self.scope["validate_normalized_invoice_fields"](fields, proofs)
        self.assertIn("evidence_review_subtotal", [issue["code"] for issue in issues])
        self.assertIn("invoice_financial_insufficient_evidence", [issue["code"] for issue in issues])
        self.assertEqual(invoice, original)

    def test_missing_amount_key_after_interruption_does_not_bypass_guard(self):
        text = "Facture F-42\nTotal HT | 12.111\n49.987 | cellule OCR\nTotal TTC 52.987"
        rules = {"fields": {"subtotal": 12.111, "total_amount": 52.987}, "items": []}
        model = InvoiceData(additional_data={"ollama_result": {"invoice_number": "F-42", "items": []}})
        fields, proofs, issues = self.check(text, rules, model, "completed_partial")
        self.assertIsNone(fields["subtotal"])
        self.assertTrue(proofs["subtotal"]["requires_review"])
        self.assertEqual(fields["total_amount"], 52.987)
        self.assertIn("llm_extraction_incomplete", [issue["code"] for issue in issues])

    def test_regex_only_fallback_obeys_the_same_ambiguity_check(self):
        text = "Total HT | 12.111\n49.987"
        rules = {"fields": {"subtotal": 12.111}, "items": []}
        fields, proofs, _ = self.check(text, rules)
        self.assertIsNone(fields["subtotal"])
        self.assertEqual(proofs["subtotal"]["source"]["line_ids"], [1, 2])

    def test_ambiguous_model_amount_is_not_replaced_with_another_regex_amount(self):
        text = "Total HT | 12.111\n49.987\nTotal HT 80.000"
        rules = {"fields": {"subtotal": 80}, "items": []}
        model = InvoiceData(additional_data={"ollama_result": {"subtotal": 12.111}})
        fields, _, _ = self.check(text, rules, model, "completed")
        self.assertIsNone(fields["subtotal"])

    def test_ambiguity_check_also_runs_after_guard_fallback_replacement(self):
        text = "Total HT | 12.111\n49.987"
        rules = {"fields": {"subtotal": 12.111}, "items": []}
        fields, proofs = self.scope["enforce_evidence_guard"]({"subtotal": 999}, text, {}, rules)
        self.assertIsNone(fields["subtotal"])
        self.assertEqual(proofs["subtotal"]["rejected_value"], 12.111)

    def test_duplicate_number_labelled_next_row_and_row_id_are_not_ambiguous(self):
        rules = {"fields": {"subtotal": 12.111}, "items": []}
        for following in ("12.111", "TVA 2.000", "0001 | Article", "14.111 | Total TTC"):
            with self.subTest(following=following):
                fields, proofs, _ = self.check("Total HT | 12.111\n" + following, rules)
                self.assertEqual(fields["subtotal"], 12.111)
                self.assertTrue(proofs["subtotal"]["supported"])

    def test_incomplete_status_blocks_validation_even_with_proven_required_fields(self):
        text = "Facture F-42\nDate 01/10/2026\nACME\nTotal TTC 12.600 TND"
        rules = {"fields": {"invoice_number": "F-42", "date": "01/10/2026", "vendor_name": "ACME",
                            "total_amount": 12.6, "currency": "TND"}, "items": []}
        for status in ("completed_partial", "timeout", "invalid_json", "rate_limited", "error"):
            with self.subTest(status=status):
                fields, proofs, issues = self.check(text, rules, status=status)
                self.assertEqual(fields["total_amount"], 12.6)
                self.assertTrue(proofs["total_amount"]["supported"])
                self.assertEqual(fields["additional_data"]["llm_extraction_status"], status)
                issue = next(i for i in issues if i["code"] == "llm_extraction_incomplete")
                self.assertEqual(issue["severity"], "blocking")

    def test_successful_resume_and_explicit_regex_mode_do_not_keep_incomplete_status(self):
        text = "Total TTC 12.600 TND"
        rules = {"fields": {"total_amount": 12.6, "currency": "TND"}, "items": []}
        for status in (None, "completed"):
            with self.subTest(status=status):
                fields, _, issues = self.check(text, rules, status=status)
                self.assertNotIn("llm_extraction_incomplete", [i["code"] for i in issues])
                self.assertNotIn("llm_extraction_status", fields["additional_data"])

    def test_explicit_model_abstention_is_not_filled_by_proven_fallback(self):
        text = "Total HT 12.111\nTotal TTC 14.111"
        rules = {"fields": {"subtotal": 12.111, "total_amount": 14.111}, "items": []}
        model = InvoiceData(additional_data={"ollama_result": {"subtotal": None}})
        fields, _, _ = self.check(text, rules, model, "completed_partial")
        self.assertIsNone(fields["subtotal"])
        self.assertEqual(fields["total_amount"], 14.111)

    def test_printed_zero_is_preserved(self):
        text = "Total HT 0.000\nTVA 0.000\nTimbre 0.000\nTotal TTC 0.000"
        values = {key: 0 for key in ("subtotal", "tax_amount", "stamp_duty", "total_amount")}
        model = InvoiceData(additional_data={"ollama_result": values})
        fields, proofs, _ = self.check(text, {"fields": values, "items": []}, model)
        for key in values:
            self.assertEqual(fields[key], 0)
            self.assertTrue(proofs[key]["supported"])

    def test_recorded_00008_fields_and_item_survive_final_guard(self):
        trace = json.loads((QA / "facture_00008.json").read_text(encoding="utf-8"))
        before = copy.deepcopy(trace["fields"])
        fields, _ = self.scope["enforce_evidence_guard"](before, trace["ocr_text"], trace["evidence"], trace["rule_candidates"])
        for key in ("vendor_tax_id", "customer_tax_id", "subtotal", "tax_amount", "stamp_duty",
                    "total_amount", "discount_amount", "items", "taxes"):
            self.assertEqual(fields[key], trace["fields"][key], key)

    def test_recorded_00158_keeps_two_ttc_articles_and_complete_customer_identifier(self):
        invoice = json.loads((QA / "facture_00158-api.json").read_text(encoding="utf-8"))["invoice"]
        raw = invoice["raw_data"]
        rules = {"fields": raw["rule_candidates"]["field_values"], "items": raw["rule_candidates"]["article_rows"]}
        fields, proofs = self.scope["enforce_evidence_guard"](
            copy.deepcopy(invoice["normalized_data"]), invoice["extracted_text"], raw["field_evidence"], rules)
        self.assertEqual(fields["customer_tax_id"], "1362434/S")
        self.assertTrue(proofs["customer_tax_id"]["supported"])
        self.assertIsNone(fields["subtotal"])
        self.assertIsNone(fields["tax_amount"])
        self.assertEqual(fields["total_amount"], 12.6)
        self.assertEqual(len(fields["items"]), 2)


if __name__ == "__main__":
    unittest.main()
