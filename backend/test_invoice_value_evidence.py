"""Functional regressions using frozen evidence, without OCR/RAG/LLM execution."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from extraction_guards import monetary_in_source
from extraction_retrieval import selected_context
from invoice_rules import filter_invoice_line_items
from invoice_value_evidence import (additional_tax_total, discount_evidence, invoice_discount,
                                    party_value_supported, supported_tax, party_cell_hints, item_discount)
from models import InvoiceItem, TaxInfo
from ollama_analyzer import EXTRACTION_GROUPS, OllamaInvoiceAnalyzer, extraction_schema
from test_hybrid_extraction import pipeline_functions


ROOT = Path(__file__).resolve().parent
QA = ROOT / "qa-results/end-to-end-00008-20260921/pipeline.json"


class InvoiceValueEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qa = json.loads(QA.read_text(encoding="utf-8"))
        cls.text = cls.qa["ocr_text"]
        cls.scope = pipeline_functions()

    def setUp(self):
        self.analyzer = OllamaInvoiceAnalyzer()

    def ground(self, field, value, citation, text=None, allowed=None):
        text = self.text if text is None else text
        allowed = set(range(1, len(text.splitlines()) + 1)) if allowed is None else allowed
        return self.analyzer._ground_fields({field: value, "evidence": {field: citation}}, text, allowed)

    def test_discount_rate_and_amount_are_distinct(self):
        proof = invoice_discount(self.text)
        self.assertEqual(proof["amount"], 50.960)
        self.assertEqual(proof["rate"], 1.0)
        self.assertEqual(proof["line_ids"], [16, 17])

    def test_row_number_never_supports_discount(self):
        self.assertFalse(monetary_in_source("discount", 1, self.text))
        self.assertTrue(monetary_in_source("discount", 50.96, self.text))
        self.assertFalse(monetary_in_source("discount", 5096, self.text))

    def test_percentage_only_is_not_an_amount(self):
        for source in ("Remise 1%", "Remise 19%\n0001 Produit", "Discount 5%"):
            self.assertEqual(discount_evidence(source), [])

    def test_standalone_discount_keeps_rate_and_printed_amount(self):
        self.assertEqual(invoice_discount("Remise 5% : 25,000")["amount"], 25.0)
        self.assertEqual(invoice_discount("Remise : 25,000")["rate"], None)
        self.assertEqual(invoice_discount("Remise | 5% | 25,000")["amount"], 25.0)

    def test_other_table_columns_and_different_values(self):
        text = "Designation | Remise 2% | Quantite | PU | MT HT\n0002 Produit | 4,000 | 2 | 100,000 | 200,000"
        self.assertEqual(invoice_discount(text)["amount"], 4.0)
        self.assertEqual(invoice_discount(text)["rate"], 2.0)
        self.assertFalse(monetary_in_source("discount", 2, text))

    def test_multiple_row_discounts_do_not_become_arbitrary_invoice_total(self):
        text = "Designation | Remise 2% | Quantite\nProduit A | 4,000 | 1\nProduit B | 2,000 | 1"
        self.assertEqual(len(discount_evidence(text)), 2)
        self.assertIsNone(invoice_discount(text))

    def test_missing_discount_cell_is_not_completed_with_math(self):
        self.assertIsNone(invoice_discount("Designation | Remise 2% | MT HT\nProduit | | 200,000"))

    def test_complete_supplier_address_is_grounded(self):
        value = "46 A AV DE TUNIS ZIND EL MGHIRA III"
        self.assertEqual(self.ground("supplier_address", value, "L31")["supplier_address"], value)

    def test_supplier_address_truncation_is_rejected(self):
        self.assertIsNone(self.ground("supplier_address", "46 A AV DE TUNIS", "L31")["supplier_address"])

    def test_null_supplier_address_can_be_copied_from_its_validated_fiscal_block(self):
        result = self.analyzer._ground_fields({"supplier_address": None, "supplier_tax_id": "816153BAM 000",
            "evidence": {"supplier_address": None, "supplier_tax_id": "L33"}}, self.text, {31, 32, 33},
            {"legal_block": {"section": "supplier", "line_ids": [31, 32, 33]}})
        self.assertEqual(result["supplier_address"], "46 A AV DE TUNIS ZIND EL MGHIRA III")
        self.assertEqual(result["evidence_adjustments"][0]["reference"], "L31-L31")
        self.assertEqual(result["evidence"]["supplier_address"], self.text.splitlines()[30])

    def test_party_address_copy_never_uses_unseen_lines_or_client_fiscal_id(self):
        for value, reference, allowed in (("816153BAM 000", "L33", {33}),
                                          ("1362434 /S/P/M/000", "L13", set(range(1, 35)))):
            result = self.analyzer._ground_fields({"supplier_address": None, "supplier_tax_id": value,
                "evidence": {"supplier_address": None, "supplier_tax_id": reference}}, self.text, allowed)
            self.assertIsNone(result["supplier_address"])

    def test_conflicting_addresses_in_same_legal_block_require_review(self):
        source = "Fournisseur: ALPHA\nAdresse: RUE A\nAdresse: RUE B\nMF: 1111111/A/M/000"
        result = self.analyzer._ground_fields({"supplier_address": None, "supplier_tax_id": "1111111/A/M/000",
            "evidence": {"supplier_address": None, "supplier_tax_id": "L4"}}, source, {1, 2, 3, 4})
        self.assertIsNone(result["supplier_address"])

    def test_customer_address_continuation_is_grounded(self):
        value = "2 BS IBRAHIM AL MAZEJ CITE OLYMPIQUE"
        self.assertEqual(self.ground("customer_address", value, "L10-L11")["customer_address"], value)

    def test_truncated_address_rejected_even_with_single_line_citation(self):
        for citation in ("L10", "L10-L11"):
            self.assertIsNone(self.ground("customer_address", "2 BS IBRAHIM AL MAZEJ CITE", citation)["customer_address"])

    def test_address_does_not_include_neighbouring_chantier(self):
        self.assertIsNone(self.ground("customer_address", "2 BS IBRAHIM AL MAZEJ CITE OLYMPIQUE CHANTIER", "L10-L11")["customer_address"])

    def test_complete_name_with_spacing_and_multiline_cells(self):
        value = "SOCIETE BRIDGE IMMOBILIERE"
        self.assertEqual(self.ground("customer_name", value, "L8-L9")["customer_name"], value)

    def test_name_cannot_absorb_date_or_bcn_cells(self):
        for value in ("SOCIETE BRIDGE IMMOBILIERE BCN", "SOCIETE BRIDGE 15/07/2024 IMMOBILIERE", "DATE SOCIETE BRIDGE IMMOBILIERE"):
            self.assertIsNone(self.ground("customer_name", value, "L8-L9")["customer_name"])

    def test_partial_name_is_rejected(self):
        self.assertIsNone(self.ground("customer_name", "SOCIETEBRIDGE", "L8")["customer_name"])

    def test_name_missing_letters_are_not_invented(self):
        self.assertFalse(party_value_supported("customer_name", "BRIDGE IMMOBILIERE", "Client: BRIDGE IMMOBILERE"))

    def test_citations_cannot_expand_into_unseen_lines(self):
        self.assertIsNone(self.ground("customer_address", "2 BS IBRAHIM AL MAZEJ CITE OLYMPIQUE", "L10-L11", allowed={10})["customer_address"])

    def test_unrelated_names_on_two_blocks_are_not_merged(self):
        source = "Client: BETA\nAdresse: RUE A\nClient: GAMMA"
        self.assertIsNone(self.ground("customer_name", "BETA GAMMA", "L1-L3", source)["customer_name"])

    def test_existing_registration_and_customer_id_models_are_mapped(self):
        data = self.analyzer._ground_fields({"supplier_registration_number": "B156302002", "customer_id": "411025",
             "evidence": {"supplier_registration_number": "L34", "customer_id": "L12-L14"}}, self.text,
             set(range(1, 35)))
        model = self.analyzer._to_invoice_data(data, "{}")
        self.assertEqual(model.supplier.registration_number, "B156302002")
        self.assertEqual(model.customer.customer_id, "411025")

    def test_registration_truncated_or_wrong_role_is_rejected(self):
        self.assertIsNone(self.ground("supplier_registration_number", "B1563", "L34")["supplier_registration_number"])
        self.assertIsNone(self.ground("customer_id", "B156302002", "L34")["customer_id"])

    def test_customer_id_value_line_is_valid_when_its_heading_is_visible(self):
        self.assertEqual(self.ground("customer_id", "411025", "L14", allowed={12, 13, 14})["customer_id"], "411025")

    def test_customer_id_cannot_use_an_unseen_heading_or_tax_value(self):
        self.assertIsNone(self.ground("customer_id", "411025", "L14", allowed={14})["customer_id"])
        self.assertIsNone(self.ground("customer_id", "1362434 /S/P/M/000", "L13")["customer_id"])
        self.assertIsNone(self.ground("customer_id", "4110", "L14")["customer_id"])

    def test_customer_name_spacing_preserves_all_letters(self):
        result = self.ground("customer_name", "SOCIETEBRIDGE IMMOBILIERE", "L8-L9")
        self.assertEqual(result["customer_name"], "SOCIETE BRIDGE IMMOBILIERE")

    def test_absent_values_remain_absent(self):
        self.assertIsNone(self.ground("supplier_address", "RUE INVENTEE", "L31")["supplier_address"])
        self.assertIsNone(self.ground("customer_id", "123456", "L12-L14")["customer_id"])

    def test_taxes_have_separate_rates_amounts_and_sources(self):
        rows = [{"rate": 1, "amount": 50.450, "base_amount": None, "source": "L23"},
                {"rate": 19, "amount": 968.143, "base_amount": None, "source": "L24"}]
        data = self.analyzer._ground_fields({"taxes": rows}, self.text, {23, 24})
        self.assertEqual([r["label"] for r in data["taxes"]], ["FODEC", "TVA"])
        model = self.analyzer._to_invoice_data(data, "{}")
        self.assertEqual([tax.amount for tax in model.taxes], [50.45, 968.143])

    def test_tax_rate_cannot_be_used_as_amount_or_fabricated_base(self):
        self.assertIsNone(supported_tax({"rate": 1, "amount": 1}, "FODEC | 1% | 50,450"))
        self.assertIsNone(supported_tax({"rate": 1, "amount": 50.45, "base_amount": 5045.04}, "FODEC | 1% | 50,450"))

    def test_tax_citations_are_restricted_to_context(self):
        data = self.analyzer._ground_fields({"taxes": [{"rate": 1, "amount": 50.45, "source": "L23"}]}, self.text, {24})
        self.assertEqual(data["taxes"], [])

    def test_tva_is_not_counted_twice_in_financial_check(self):
        taxes = [supported_tax({"rate": 1, "amount": 50.45}, "FODEC | 1% | 50,450"),
                 supported_tax({"rate": 19, "amount": 968.143}, "TVA | 19% | 968,143")]
        self.assertEqual(float(additional_tax_total(taxes + taxes, {"supported": True})), 50.45)

    def test_unproven_taxes_do_not_reconcile_total(self):
        row = supported_tax({"rate": 1, "amount": 50.45}, "FODEC | 1% | 50,450")
        self.assertEqual(additional_tax_total([row]), 0)

    def test_fodec_and_rounding_preserve_printed_ttc(self):
        row = supported_tax({"rate": 1, "amount": 50.45}, "FODEC | 1% | 50,450")
        fields = {"subtotal": 5045.04, "tax_amount": 968.143, "stamp_duty": 1, "total_amount": 6064.634, "taxes": [row], "currency": "TND"}
        original = copy.deepcopy(fields)
        proofs = {key: copy.deepcopy(self.qa["proofs"][key]) for key in ("subtotal", "tax_amount", "stamp_duty", "total_amount")}
        proofs["taxes"] = {"supported": True, "source": row["source"]}
        issues = self.scope["validate_normalized_invoice_fields"](fields, proofs)
        self.assertNotIn("invoice_total_math_mismatch", [i["code"] for i in issues])
        self.assertIn("invoice_total_rounding_difference", [i["code"] for i in issues])
        self.assertEqual(fields, original)

    def test_real_financial_mismatch_is_still_blocking(self):
        fields = {"subtotal": 100, "tax_amount": 19, "stamp_duty": 1, "total_amount": 160, "currency": "TND"}
        proofs = {key: {"value": value, "supported": True, "source": f"{key}: {value}"} for key, value in fields.items()}
        issues = self.scope["validate_normalized_invoice_fields"](fields, proofs)
        self.assertIn("invoice_total_math_mismatch", [i["code"] for i in issues])

    def test_tolerance_is_not_automatically_called_rounding(self):
        fields = {"subtotal": 100, "tax_amount": 19, "stamp_duty": 1, "total_amount": 120.04, "currency": "TND"}
        proofs = {key: {"value": value, "supported": True, "source": f"{key}: {value}"} for key, value in fields.items()}
        issues = self.scope["validate_normalized_invoice_fields"](fields, proofs)
        self.assertIn("invoice_total_math_mismatch", [i["code"] for i in issues])
        self.assertNotIn("invoice_total_rounding_difference", [i["code"] for i in issues])

    def test_duplicate_tax_with_overlapping_citations_is_counted_once(self):
        first = supported_tax({"rate": 1, "amount": 50.45}, "FODEC | 1% | 50,450")
        second = supported_tax({"rate": 1, "amount": 50.45}, "Total Taxes\nFODEC | 1% | 50,450\nTVA | 19% | 968,143")
        self.assertEqual(float(additional_tax_total([first, second], {"supported": True})), 50.45)

    def test_unit_survives_existing_item_model_and_final_filter(self):
        row = copy.deepcopy(self.qa["fields"]["items"][0])
        row["unit"] = "m\u00b3"
        model = self.analyzer._to_invoice_data({"items": [row]}, "{}")
        self.assertEqual(model.items[0].unit, "m\u00b3")
        rows, _ = filter_invoice_line_items([row], self.text, 6064.634)
        self.assertEqual(rows[0]["unit"], "m\u00b3")

    def test_unproven_unit_is_rejected(self):
        row = {**self.qa["fields"]["items"][0], "source": "L16-L19", "unit": "tonne"}
        result = self.analyzer._ground_fields({"items": [row]}, self.text, {16, 17, 18, 19})
        self.assertIsNone(result["items"][0]["unit"])

    def test_no_new_item_discount_or_tax_label_model_properties(self):
        self.assertNotIn("discount_rate", InvoiceItem.model_fields)
        self.assertNotIn("discount_amount", InvoiceItem.model_fields)
        self.assertNotIn("label", TaxInfo.model_fields)
        self.assertIn("unit", extraction_schema()["properties"]["items"]["items"]["properties"])

    def test_currency_default_is_not_model_or_ocr_value(self):
        model = self.analyzer._to_invoice_data({"currency": None}, "{}")
        self.assertIsNone(model.currency)
        rules = {"fields": {"currency": "TND"}, "items": []}
        normalized = self.scope["normalize_invoice_data"](model, "FACTURE", rules)
        fields, proofs = self.scope["enforce_evidence_guard"](normalized, "FACTURE", {}, rules)
        self.assertEqual(fields["currency"], "TND")
        self.assertIsNone(fields["currency_extracted"])
        self.assertFalse(proofs["currency"]["supported"])
        self.assertEqual(proofs["currency"]["source"], "workflow_default_tnd")
        codes = [i["code"] for i in self.scope["validate_normalized_invoice_fields"](fields, proofs)]
        self.assertIn("currency_workflow_default", codes)
        self.assertNotIn("unproven_currency", codes)

    def test_printed_currency_remains_proven(self):
        fields, proofs = self.scope["enforce_evidence_guard"]({"currency": "EUR"}, "TOTAL 10 EUR", {},
                                                             {"fields": {"currency": "TND"}})
        self.assertEqual(fields["currency_extracted"], "EUR")
        self.assertTrue(proofs["currency"]["supported"])
        self.assertFalse(proofs["currency"]["default_applied"])

    def test_rejected_party_values_request_review_after_mapping(self):
        response = {"customer_name": "SOCIETEBRIDGE", "evidence": {"customer_name": "L8"}}
        with patch.object(self.analyzer, "_generate", return_value={"response": json.dumps(response)}):
            model = self.analyzer.analyze_invoice_text(self.text, {"customer_name": self.qa["evidence"]["customer_name"]})
        fields = self.scope["normalize_invoice_data"](model, self.text, self.qa["rule_candidates"])
        _, proofs = self.scope["enforce_evidence_guard"](fields, self.text, self.qa["evidence"], self.qa["rule_candidates"])
        self.assertTrue(proofs["customer_name"]["requires_review"])

    def test_context_lines_and_passages_unchanged_for_all_batches(self):
        for group, previous in zip(EXTRACTION_GROUPS, self.qa["trace"]):
            context = selected_context(self.text, {k: self.qa["evidence"].get(k, []) for k in group})
            self.assertEqual(set(context["allowed_lines"]), set(previous["allowed_lines"]))
            self.assertEqual(context["sources"], previous["sources"])
            self.assertEqual(context["text"], previous["prompt"].split("Passages OCR retrouves (lignes globales du document):\n")[1])

    def test_one_recheck_can_fix_truncation_without_changing_grounded_fields(self):
        fields = ["customer_name", "supplier_tax_id"]
        initial = self.analyzer._ground_fields({"customer_name": "SOCIETEBRIDGE", "supplier_tax_id": "816153BAM 000",
            "evidence": {"customer_name": "L8", "supplier_tax_id": "L33"}}, self.text, set(range(1, 35)))
        response = {"customer_name": "SOCIETE BRIDGE IMMOBILIERE", "supplier_tax_id": None,
                    "evidence": {"customer_name": "L8-L9", "supplier_tax_id": None}}
        batch = {}
        with patch.object(self.analyzer, "_generate", return_value={"response": json.dumps(response)}) as generate:
            result, citations, _ = self.analyzer._recheck_rejected(initial, {"supplier_tax_id": "L33"}, [], "same context", {},
                self.text, {"allowed_lines": set(range(1, 35)), "source_metadata": {}}, fields, batch, "original raw")
        generate.assert_called_once()
        self.assertEqual(result["customer_name"], "SOCIETE BRIDGE IMMOBILIERE")
        self.assertEqual(result["supplier_tax_id"], "816153BAM 000")
        self.assertEqual(citations["customer_name"], "L8-L9")
        self.assertEqual(batch["evidence_recheck"]["initial_response"], "original raw")
        self.assertEqual(result["grounding_rejected_fields"], [])

    def test_recheck_failure_preserves_first_grounded_answer(self):
        initial = {"customer_name": None, "supplier_name": "S.T.FOOM", "evidence": {},
                   "grounding_rejected_fields": ["customer_name"]}
        with patch.object(self.analyzer, "_generate", side_effect=TimeoutError) as generate:
            result, _, _ = self.analyzer._recheck_rejected(copy.deepcopy(initial), {}, [], "context", {}, self.text,
                {"allowed_lines": {8}, "source_metadata": {}}, ["customer_name", "supplier_name"], {}, "raw")
        generate.assert_called_once()
        self.assertEqual(result, initial)

    def test_recheck_cannot_accept_unseen_evidence(self):
        initial = {"customer_name": None, "evidence": {}, "grounding_rejected_fields": ["customer_name"]}
        response = {"customer_name": "SOCIETE BRIDGE IMMOBILIERE", "evidence": {"customer_name": "L8-L9"}}
        with patch.object(self.analyzer, "_generate", return_value={"response": json.dumps(response)}) as generate:
            result, _, _ = self.analyzer._recheck_rejected(initial, {}, [], "context", {}, self.text,
                {"allowed_lines": {8}, "source_metadata": {}}, ["customer_name"], {}, "raw")
        generate.assert_called_once()
        self.assertIsNone(result["customer_name"])

    def test_valid_batch_never_triggers_recheck(self):
        data = {"invoice_number": "95/2024", "grounding_rejected_fields": []}
        with patch.object(self.analyzer, "_generate") as generate:
            self.analyzer._recheck_rejected(data, {}, [], "", {}, "", {}, ["invoice_number"], {}, "raw")
        generate.assert_not_called()

    def test_missing_supplier_name_with_available_passages_can_be_rechecked_once(self):
        context = {"allowed_lines": {1, 2}, "source_metadata": {"header": {"section": "supplier", "line_ids": [1, 2]}},
                   "field_sources": {"vendor_name": ["P1:L1-2"]}}
        response = {"supplier_name": "S.T.FOOM", "evidence": {"supplier_name": "L2"}}
        with patch.object(self.analyzer, "_generate", return_value={"response": json.dumps(response)}) as generate:
            result, _, _ = self.analyzer._recheck_rejected({"supplier_name": None, "evidence": {},
                "grounding_rejected_fields": []}, {}, [], "context", {}, self.text, context, ["supplier_name"], {}, "raw")
        generate.assert_called_once()
        self.assertEqual(result["supplier_name"], "S.T.FOOM")

    def test_party_hints_use_complete_cells_and_no_hidden_lines(self):
        hints = party_cell_hints(["customer_name", "customer_id", "supplier_address"], self.text,
                                 {8, 9, 10, 11, 12, 13, 14})
        self.assertIn("SOCIETEBRIDGE IMMOBILIERE", hints)
        self.assertIn("2 BS IBRAHIM AL MAZEJ CITE OLYMPIQUE", hints)
        self.assertIn('"value":"411025"', hints)
        self.assertIn('"source":"L12-L14"', hints)
        self.assertNotIn("46 A AV DE TUNIS", hints)

    def test_discount_metadata_can_use_printed_header_outside_row_citation(self):
        row = {"source": "\n".join(self.text.splitlines()[16:19])}
        discount = item_discount(row, self.text)
        self.assertEqual(discount["rate"], 1)
        self.assertEqual(discount["amount"], 50.96)

    def test_missing_tva_is_not_reported_as_a_rounding_difference(self):
        issues = self.scope["validate_normalized_invoice_fields"]({"subtotal": 5045.04,
                    "total_amount": 6064.634, "stamp_duty": 1})
        self.assertNotIn("invoice_total_rounding_difference", [i["code"] for i in issues])

    def test_complete_00008_values_survive_mapping_and_final_validation(self):
        responses = [copy.deepcopy(t["parsed_before_grounding"]) for t in self.qa["trace"]]
        identity = responses[1]
        changes = {
            "supplier_address": ("46 A AV DE TUNIS ZIND EL MGHIRA III", "L31"),
            "customer_address": ("2 BS IBRAHIM AL MAZEJ CITE OLYMPIQUE", "L10-L11"),
            "customer_name": ("SOCIETE BRIDGE IMMOBILIERE", "L8-L9"),
            "supplier_registration_number": ("B156302002", "L34"),
            "customer_id": ("411025", "L12-L14"),
        }
        for key, (value, source) in changes.items():
            identity[key], identity["evidence"][key] = value, source
        responses[2]["taxes"] = [{"rate": 1, "amount": 50.45, "source": "L23", "base_amount": None},
                                   {"rate": 19, "amount": 968.143, "source": "L24", "base_amount": None}]
        responses[4]["items"][0]["unit"] = "m\u00b3"
        with patch.object(self.analyzer, "_generate", side_effect=[{"response": json.dumps(r)} for r in responses]) as generate:
            invoice = self.analyzer.analyze_invoice_text(self.text, self.qa["evidence"])
        self.assertEqual(generate.call_count, 5)
        normalized = self.scope["normalize_invoice_data"](invoice, self.text, self.qa["rule_candidates"])
        fields, proofs = self.scope["enforce_evidence_guard"](normalized, self.text, self.qa["evidence"], self.qa["rule_candidates"])
        self.assertEqual(fields["customer_name"], "SOCIETE BRIDGE IMMOBILIERE")
        self.assertEqual(fields["vendor_address"], changes["supplier_address"][0])
        self.assertEqual(fields["customer_address"], changes["customer_address"][0])
        self.assertEqual(fields["customer_id"], "411025")
        self.assertEqual(fields["vendor_registration_number"], "B156302002")
        self.assertEqual(fields["discount_amount"], 50.96)
        self.assertEqual(fields["additional_data"]["item_discounts"][0]["discount_rate"], 1)
        self.assertEqual(fields["items"][0]["unit"], "m\u00b3")
        self.assertEqual([r["line_ids"] for r in fields["taxes"]], [[23], [24]])
        self.assertEqual(fields["total_amount"], 6064.634)
        issues = self.scope["validate_normalized_invoice_fields"](fields, proofs)
        self.assertFalse([i for i in issues if i["severity"] == "blocking"], issues)


if __name__ == "__main__":
    unittest.main()
