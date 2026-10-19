"""Last targeted guards, using recorded contexts/JSON only; no model or OCR."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from identifier_evidence import identifier_role_supported, identifier_spans
from invoice_value_evidence import explicit_customer_name
from ollama_analyzer import OllamaInvoiceAnalyzer
from test_hybrid_extraction import pipeline_functions


QA = Path(__file__).parent / "qa-results/generalization-audit-20260921T121302Z/final"


class FinalTargetedGuardTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = OllamaInvoiceAnalyzer()

    def ground(self, data, text, visible=None, metadata=None):
        if visible is None:
            visible = set(range(1, len(text.splitlines()) + 1))
        return self.analyzer._ground_fields(copy.deepcopy(data), text, visible, metadata or {})

    def replay(self, name):
        previous = json.loads((QA / (name + ".json")).read_text(encoding="utf-8"))
        batches = {frozenset(t["fields"]): t for t in previous["trace"]}
        attempts = {}

        def generation(prompt, schema):
            key = frozenset(schema["properties"]) - {"evidence"}
            index = attempts.get(key, 0)
            attempts[key] = index + 1
            return copy.deepcopy(batches[key]["generations"][index]["payload"])

        with patch.object(self.analyzer, "_generate", side_effect=generation), \
                patch.object(self.analyzer.retriever, "retrieve", side_effect=AssertionError("No retrieval")), \
                patch("requests.sessions.Session.request", side_effect=AssertionError("No network")):
            model = self.analyzer.analyze_invoice_text(previous["ocr_text"], previous["evidence"])
        self.assertIsNotNone(model)
        self.assertTrue(all(count <= len(batches[key]["generations"]) for key, count in attempts.items()))
        scope = pipeline_functions()
        fields = scope["normalize_invoice_data"](model, previous["ocr_text"], previous["rule_candidates"])
        fields, proofs = scope["enforce_evidence_guard"](fields, previous["ocr_text"], previous["evidence"], previous["rule_candidates"])
        issues = scope["validate_normalized_invoice_fields"](fields, proofs)
        return fields, proofs, issues

    def test_00158_full_replay_accepts_client_tax_id_without_recheck(self):
        fields, proofs, _ = self.replay("facture_00158")
        self.assertEqual(fields["customer_tax_id"], "1362434/S")
        self.assertTrue(proofs["customer_tax_id"]["supported"])
        self.assertFalse(proofs["customer_tax_id"].get("requires_review"))
        self.assertEqual(fields["vendor_tax_id"], "1547376Z/N/C/000")

    def test_collapsed_tax_row_with_unambiguous_customer_and_invoice_header(self):
        text = "FACTURE N:2026/17 | Nom du Client : ACME\nMF : 7654321 /K/P/012"
        value = "7654321/K/P/012"
        result = self.ground({"customer_tax_id": value, "evidence": {"customer_tax_id": "L2"}}, text)
        self.assertEqual(result["customer_tax_id"], value)
        self.assertEqual(result["evidence"]["customer_tax_id"], text.splitlines()[1])

    def test_collapsed_tax_row_never_accepts_truncation_absence_or_changed_separator(self):
        text = "Invoice 2026/17 | Customer: ACME\nMF : 7654321/K/P/012"
        for value in ["7654321", "7654321/K", "7654321-K-P-012", "1234567/Z"]:
            with self.subTest(value=value):
                self.assertEqual(identifier_spans("customer_tax_id", value, text.splitlines()[1]), [])
                result = self.ground({"customer_tax_id": value, "evidence": {"customer_tax_id": "L2"}}, text)
                self.assertIsNone(result["customer_tax_id"])

    def test_collapsed_customer_identifier_never_belongs_to_supplier(self):
        text = "Invoice 2026/17 | Customer: ACME\nMF : 7654321/K/P/012"
        self.assertFalse(identifier_role_supported("supplier_tax_id", "7654321/K/P/012", text, "L2"))

    def test_collapsed_owner_requires_visible_immediately_preceding_label(self):
        text = "Invoice 2026/17 | Customer: ACME\nMF : 7654321/K/P/012"
        self.assertFalse(identifier_role_supported("customer_tax_id", "7654321/K/P/012", text, "L2", {2}))

    def test_collapsed_row_with_two_owners_is_ambiguous(self):
        text = "Supplier: ONE | Customer: TWO\nMF : 7654321/K/P/012"
        for field in ("supplier_tax_id", "customer_tax_id"):
            self.assertFalse(identifier_role_supported(field, "7654321/K/P/012", text, "L2"))

    def test_ship_to_is_not_customer(self):
        text = "Invoice 2026/17 | Ship to: ACME\nMF : 7654321/K/P/012"
        self.assertFalse(identifier_role_supported("customer_tax_id", "7654321/K/P/012", text, "L2"))

    def test_collapsed_row_does_not_override_conflicting_role_metadata(self):
        text = "Invoice 2026/17 | Customer: ACME\nMF : 7654321/K/P/012"
        metadata = {"block": {"section": "supplier", "line_ids": [2]}}
        self.assertFalse(identifier_role_supported("customer_tax_id", "7654321/K/P/012", text, "L2", {1, 2}, metadata))

    def test_client_word_in_stamp_is_not_a_name_owner_anchor(self):
        text = "Invoice 2026/17 | Cachet et signature client\nMF : 7654321/K/P/012"
        self.assertFalse(identifier_role_supported("customer_tax_id", "7654321/K/P/012", text, "L2"))

    def test_regular_two_column_fiscal_rows_keep_their_owners(self):
        text = "Supplier: ONE | Customer: TWO\nMF : 7654321/K/P/012 | MF : 1234567/A/M/000"
        self.assertTrue(identifier_role_supported("supplier_tax_id", "7654321/K/P/012", text, "L2"))
        self.assertTrue(identifier_role_supported("customer_tax_id", "1234567/A/M/000", text, "L2"))

    def test_00791_replay_rejects_ambiguous_ht_and_keeps_only_proven_repairs(self):
        fields, proofs, issues = self.replay("facture_00791")
        self.assertIsNone(fields["subtotal"])
        self.assertTrue(proofs["subtotal"]["requires_review"])
        self.assertIsNone(fields["tax_amount"])
        self.assertEqual(fields["taxes"], [])
        self.assertEqual(fields["stamp_duty"], 0.831)
        self.assertTrue(proofs["stamp_duty"]["supported"])
        self.assertEqual(fields["customer_name"], "STE DE PROMOTION")
        self.assertFalse(fields["vendor_name"])
        self.assertEqual(fields["total_amount"], 37.02)
        self.assertEqual(fields["items"], [])
        self.assertIn("invoice_financial_insufficient_evidence", [i["code"] for i in issues])

    def test_ambiguous_ht_never_chooses_the_amount_that_matches_ttc(self):
        text = "Total HT | 12,111\n49,987 | texte OCR\nTVA 3.000\nTotal TTC 52.987"
        data = {"subtotal": 12.111, "total_amount": 52.987, "evidence": {"subtotal": "L1", "total_amount": "L4"}}
        result = self.ground(data, text)
        self.assertIsNone(result["subtotal"])
        self.assertIn("subtotal", result["grounding_rejected_fields"])
        self.assertEqual(result["total_amount"], 52.987)
        self.assertEqual(result["evidence_warnings"]["subtotal"]["line_ids"], [1, 2])

    def test_unseen_neighbor_does_not_change_context_grounding(self):
        text = "Total HT | 12.111\n49.987"
        result = self.ground({"subtotal": 12.111, "evidence": {"subtotal": "L1"}}, text, {1})
        self.assertEqual(result["subtotal"], 12.111)

    def test_labelled_next_amount_and_duplicate_value_do_not_invalidate_ht(self):
        for following in ["TVA | 2.000", "12.111", "0001 | Article", "14.111 | Total TTC"]:
            text = "Total HT | 12.111\n" + following
            result = self.ground({"subtotal": 12.111, "evidence": {"subtotal": "L1"}}, text)
            self.assertEqual(result["subtotal"], 12.111, following)

    def test_stamp_citation_is_extended_only_to_visible_adjacent_label(self):
        text = "Timbre fiscal | (%\n0,725"
        result = self.ground({"stamp_duty": 0.725, "evidence": {"stamp_duty": "L2"}}, text)
        self.assertEqual(result["stamp_duty"], 0.725)
        self.assertEqual(result["evidence"]["stamp_duty"], text)
        self.assertEqual(result["evidence_adjustments"][0]["reference"], "L1-L2")

    def test_stamp_repair_preserves_printed_zero(self):
        result = self.ground({"stamp_duty": 0, "evidence": {"stamp_duty": "L2"}}, "Stamp duty\n0.000")
        self.assertEqual(result["stamp_duty"], 0)

    def test_stamp_citation_repair_cannot_see_hidden_label(self):
        text = "Timbre fiscal\n0.725"
        result = self.ground({"stamp_duty": 0.725, "evidence": {"stamp_duty": "L2"}}, text, {2})
        self.assertIsNone(result["stamp_duty"])

    def test_stamp_pair_never_moves_vat_or_fills_abstention(self):
        for data in [{"stamp_duty": 0.725, "evidence": {"stamp_duty": "L2"}},
                     {"stamp_duty": None, "evidence": {"stamp_duty": None}}]:
            self.assertIsNone(self.ground(data, "TVA 19%\n0.725")["stamp_duty"])
        data = {"stamp_duty": None, "evidence": {"stamp_duty": None}}
        self.assertIsNone(self.ground(data, "Timbre fiscal\n0.725")["stamp_duty"])

    def test_stamp_citation_repair_rejects_gap_multiple_values_or_absent_value(self):
        for text, ref, amount in [("Timbre fiscal\n\n0.725", "L3", 0.725),
                                   ("Timbre fiscal\n0.725 | 1.000", "L2", 0.725),
                                   ("Timbre fiscal\n0.725", "L2", 0.999)]:
            self.assertIsNone(self.ground({"stamp_duty": amount, "evidence": {"stamp_duty": ref}}, text)["stamp_duty"])

    def test_customer_company_cell_requires_explicit_code_block_and_visible_code(self):
        text = "Code client | SOCIETE EXEMPLE\n5.12345.00 | code\nFacture 123"
        result = self.ground({"customer_name": None, "evidence": {}}, text)
        self.assertEqual(result["customer_name"], "SOCIETE EXEMPLE")
        self.assertEqual(result["evidence_adjustments"][0]["reference"], "L1-L2")
        self.assertIsNone(self.ground({"customer_name": None, "evidence": {}}, text, {1})["customer_name"])

    def test_company_name_without_owner_is_not_inferred(self):
        text = "CODE-0123\nACME IMMOBILIERE -SUARL-\nFACTURE 2026/1"
        self.assertIsNone(explicit_customer_name(text))
        self.assertIsNone(self.ground({"customer_name": None, "evidence": {}}, text)["customer_name"])

    def test_customer_copy_rejects_conflicting_role_or_multiple_names(self):
        text = "Code client | SOCIETE EXEMPLE\n5.12345.00"
        metadata = {"block": {"section": "ship_to", "line_ids": [1, 2]}}
        self.assertIsNone(self.ground({"customer_name": None, "evidence": {}}, text, {1, 2}, metadata)["customer_name"])
        text += "\nCode client | SOCIETE SECONDE\n6.12345.00"
        self.assertIsNone(self.ground({"customer_name": None, "evidence": {}}, text)["customer_name"])

    def test_customer_copy_never_creates_supplier_or_missing_products(self):
        text = "Code client | SOCIETE EXEMPLE\n5.12345.00\nwww.operateur.tn"
        result = self.ground({"supplier_name": None, "items": [], "evidence": {}}, text)
        self.assertIsNone(result["supplier_name"])
        self.assertEqual(result["items"], [])

    def test_00079_replay_keeps_proven_aggregate_and_abstains_on_unprovided_identity(self):
        fields, proofs, issues = self.replay("facture_00079")
        self.assertFalse(fields["customer_name"])
        self.assertFalse(fields["customer_tax_id"])
        self.assertEqual(fields["tax_amount"], 10730.222)
        self.assertEqual(fields["taxes"], [])
        self.assertTrue(proofs["taxes"]["requires_review"])
        self.assertEqual(fields["total_amount"], 68350.131)
        self.assertEqual(len(fields["items"]), 5)
        self.assertIn("invoice_financial_insufficient_evidence", [i["code"] for i in issues])


if __name__ == "__main__":
    unittest.main()
