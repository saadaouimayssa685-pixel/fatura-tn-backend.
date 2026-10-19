"""Frozen audit responses only: no OCR, retrieval, model or network execution."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from extraction_guards import monetary_in_source, party_conflicts
from identifier_evidence import identifier_spans
from invoice_rules import extract_unit_table_rows, filter_invoice_line_items
from invoice_value_evidence import invoice_discount, supported_tax, party_citation_complete
from ollama_analyzer import OllamaInvoiceAnalyzer
from test_hybrid_extraction import pipeline_functions


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "qa-results/generalization-audit-20260921T121302Z/final"


class EvidenceGuardGeneralizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = {name: json.loads((AUDIT / (name + ".json")).read_text(encoding="utf-8")) for name in (
            "facture_00040", "facture_00079", "facture_00158", "Template1_Instance175", "Template26_Instance175")}
        cls.scope = pipeline_functions()

    def setUp(self):
        self.analyzer = OllamaInvoiceAnalyzer()

    def batch(self, name, field):
        doc = self.docs[name]
        trace = next(t for t in doc["trace"] if field in t["fields"])
        return doc, trace

    def ground_batch(self, name, field):
        doc, trace = self.batch(name, field)
        raw = copy.deepcopy(trace["grounding_attempts"][0]["before"])
        return self.analyzer._ground_fields(raw, doc["ocr_text"], set(trace["allowed_lines"]), trace["source_metadata"])

    def field(self, name, field, value, reference, allowed=None):
        doc, trace = self.batch(name, "supplier_name")
        return self.analyzer._ground_fields({field: value, "evidence": {field: reference}}, doc["ocr_text"],
            set(trace["allowed_lines"]) if allowed is None else allowed, trace["source_metadata"])[field]

    def test_template26_correct_address_stops_before_invoice_id(self):
        data = self.ground_batch("Template26_Instance175", "supplier_name")
        self.assertEqual(data["supplier_address"], "496 Buck Rest Suite 598 Jenningsburgh, UT 95460 US")
        self.assertNotIn("INVOICE ID", data["evidence"]["supplier_address"])

    def test_template26_contaminated_address_is_rejected(self):
        value = "496 Buck Rest Suite 598 Jenningsburgh, UT 95460 US INVOICE ID INV/21-53/920"
        self.assertIsNone(self.field("Template26_Instance175", "supplier_address", value, "L3-L5"))

    def test_template26_incomplete_citation_remains_rejected(self):
        self.assertIsNone(self.field("Template26_Instance175", "supplier_address", "496 Buck Rest Suite 598", "L3"))

    def test_address_stops_at_new_header_or_role(self):
        for boundary in ("INVOICE ID ABC-1", "Description | Quantity | Price", "Bill To: BETA", "SHIP TO:",
                         "Terms and Conditions", "Bank Account 123456", "www.example.com"):
            text = "Fournisseur ALPHA\nAdresse: 46 Avenue Centrale\n" + boundary
            data = self.analyzer._ground_fields({"supplier_address": "46 Avenue Centrale", "evidence": {"supplier_address": "L2"}}, text, {1, 2, 3})
            self.assertEqual(data["supplier_address"], "46 Avenue Centrale", boundary)

    def test_template1_supplier_address_excludes_other_column(self):
        value = "16424 Timothy Mission Markville, AK 58294 US"
        self.assertEqual(self.field("Template1_Instance175", "supplier_address", value, "L5-L6"), value)

    def test_template1_customer_postal_tail_is_not_supplier_continuation(self):
        value = "16424 Timothy Mission Markville, AK 58294 US Halltown, OR 08608 US"
        self.assertIsNone(self.field("Template1_Instance175", "supplier_address", value, "L5-L7"))

    def test_template1_customer_address_remains_complete(self):
        data = self.ground_batch("Template1_Instance175", "supplier_name")
        self.assertEqual(data["customer_address"], "1006 Ali Burg Apt. 051 Halltown, OR 08608 US")

    def test_template1_customer_phone_cannot_belong_to_supplier(self):
        self.assertIsNone(self.field("Template1_Instance175", "supplier_phone", "+(445)550-8759", "L8"))
        self.assertEqual(self.field("Template1_Instance175", "customer_phone", "+(445)550-8759", "L8"), "+(445)550-8759")

    def test_00040_complete_identifiers_keep_their_roles(self):
        data = self.ground_batch("facture_00040", "supplier_name")
        self.assertEqual(data["supplier_tax_id"], "1144983D/A/M/000")
        self.assertEqual(data["customer_tax_id"], "1085144/P/A/M/000")

    def test_00040_swapped_tax_ids_are_rejected(self):
        self.assertIsNone(self.field("facture_00040", "supplier_tax_id", "1085144/P/A/M/000", "L5"))
        self.assertIsNone(self.field("facture_00040", "customer_tax_id", "1144983D/A/M/000", "L32"))

    def test_00040_truncation_and_modified_separators_are_still_rejected(self):
        for value in ("1144983", "1144983D/A", "1144983D-A-M-000", "1144983D/A/M/001"):
            with self.subTest(value=value):
                self.assertIsNone(self.field("facture_00040", "supplier_tax_id", value, "L32"))

    def test_template1_gstin_is_not_owned_by_neighbouring_bill_to(self):
        data = self.ground_batch("Template1_Instance175", "supplier_name")
        self.assertEqual(data["supplier_tax_id"], "12345670 00070007")
        self.assertIsNone(self.field("Template1_Instance175", "customer_tax_id", "12345670 00070007", "L10"))

    def test_template1_gstin_survives_final_conflict_guard(self):
        doc = self.docs["Template1_Instance175"]
        fields = {"vendor_tax_id": "12345670 00070007", "customer_name": "Andrea Mills"}
        self.assertNotIn("vendor_tax_id", party_conflicts(fields, doc["ocr_text"]))
        result, _ = self.scope["enforce_evidence_guard"](fields, doc["ocr_text"], doc["evidence"], doc["rule_candidates"])
        self.assertEqual(result["vendor_tax_id"], "12345670 00070007")

    def test_ship_to_is_neither_supplier_nor_billing_customer(self):
        source = "SHIP TO:\nReceiver\nMF: 1234567/A/M/000\nPhone: 12345678"
        for field in ("supplier_tax_id", "customer_tax_id"):
            result = self.analyzer._ground_fields({field: "1234567/A/M/000", "evidence": {field: "L3"}}, source, {1, 2, 3, 4})
            self.assertIsNone(result[field])

    def test_ship_to_name_cannot_become_a_billing_customer(self):
        source = "SHIP TO:\nReceiver"
        result = self.analyzer._ground_fields({"customer_name": "Receiver", "evidence": {"customer_name": "L2"}}, source, {1, 2})
        self.assertIsNone(result["customer_name"])

    def test_unknown_single_tax_id_needs_review(self):
        for field in ("supplier_tax_id", "customer_tax_id"):
            data = self.analyzer._ground_fields({field: "1234567/A/M/000", "evidence": {field: "L1"}}, "MF: 1234567/A/M/000", {1})
            self.assertIsNone(data[field])
            self.assertIn(field, data["grounding_rejected_fields"])

    def test_hidden_bill_to_and_ship_to_cannot_establish_ownership(self):
        for field, value, reference in (("supplier_phone", "+(940)204-2472", "L12"),
                                        ("customer_address", "East Annettetown, OR 08010 US", "L11")):
            self.assertIsNone(self.field("Template26_Instance175", field, value, reference))

    def test_city_only_citation_is_rejected_even_in_known_customer_block(self):
        source = "Client: BETA\n209 Brianna Lane Apt. 079\nEast Annettetown, OR 08010 US"
        value = "East Annettetown, OR 08010 US"
        self.assertFalse(party_citation_complete("customer_address", value, source, "L3", {1, 2, 3}))
        data = self.analyzer._ground_fields({"customer_address": value, "evidence": {"customer_address": "L3"}}, source, {1, 2, 3})
        self.assertIsNone(data["customer_address"])

    def test_full_unlabelled_address_in_customer_block_is_accepted(self):
        source = "Client: BETA\n209 Brianna Lane Apt. 079\nEast Annettetown, OR 08010 US"
        value = "209 Brianna Lane Apt. 079 East Annettetown, OR 08010 US"
        data = self.analyzer._ground_fields({"customer_address": value, "evidence": {"customer_address": "L2-L3"}}, source, {1, 2, 3})
        self.assertEqual(data["customer_address"], value)

    def test_parenthesized_discount_keeps_rate_and_amount_separate(self):
        result = invoice_discount("DISCOUNT(3.84%): (-) 26.92")
        self.assertEqual((result["rate"], result["amount"]), (3.84, 26.92))
        self.assertFalse(monetary_in_source("discount", 3.84, result["source"]))

    def test_discount_cannot_consume_next_line_or_cell_id(self):
        for source in ("DISCOUNT(3.84%):\n0001", "DISCOUNT(3.84%): | Line 0001", "DISCOUNT(3.84%)"):
            self.assertIsNone(invoice_discount(source))

    def test_parenthesized_vat_is_supported_without_calculating_base(self):
        source = "TAX:VAT (3.57%): 25.04 USD"
        row = supported_tax({"rate": 3.57, "amount": 25.04}, source)
        self.assertEqual((row["label"], row["rate"], row["amount"], row["base_amount"]), ("VAT", 3.57, 25.04, None))
        self.assertIsNone(supported_tax({"rate": 3.57, "amount": 3.57}, source))
        self.assertIsNone(supported_tax({"rate": 3.57, "amount": 25.04, "base_amount": 700.94}, source))

    def test_template1_subtotal_tax_and_discount_survive_mapping(self):
        doc = self.docs["Template1_Instance175"]
        data = self.ground_batch("Template1_Instance175", "taxes")
        model = self.analyzer._to_invoice_data(data, "{}")
        normalized = self.scope["normalize_invoice_data"](model, doc["ocr_text"], doc["rule_candidates"])
        normalized["tax_rate"] = "3,57%"
        final, _ = self.scope["enforce_evidence_guard"](normalized, doc["ocr_text"], doc["evidence"], doc["rule_candidates"])
        self.assertEqual(final["subtotal"], 700.94)
        self.assertEqual(final["taxes"][0]["amount"], 25.04)
        self.assertEqual(final["tax_rate"], "3,57%")
        self.assertEqual((final["discount_rate"], final["discount_amount"]), (3.84, 26.92))
        self.assertEqual(final["total_amount"], 694.09)  # Printed total, not recomputed.

    def test_rc_prefix_and_mf_boundary_preserve_complete_registration(self):
        source = "Fournisseur ALPHA\nR.C:B 183611999-MF:385191J/A/M/000"
        data = self.analyzer._ground_fields({"supplier_registration_number": "B183611999", "supplier_tax_id": "385191J/A/M/000",
            "evidence": {"supplier_registration_number": "L2", "supplier_tax_id": "L2"}}, source, {1, 2})
        self.assertEqual(data["supplier_registration_number"], "B183611999")
        self.assertEqual(data["supplier_tax_id"], "385191J/A/M/000")

    def test_mf_boundary_exception_does_not_accept_arbitrary_suffix_truncation(self):
        for source in ("R.C:B183611999-A", "R.C:B183611999-MF2", "R.C:B183611999/MF", "R.C:B183611999-MF"):
            self.assertFalse(identifier_spans("supplier_registration_number", "B183611999", source))
        self.assertFalse(identifier_spans("supplier_registration_number", "B1836", "R.C:B183611999-MF:1234567/A"))

    def test_long_ttc_label_and_subtotal_are_not_interchanged(self):
        self.assertTrue(monetary_in_source("total_amount", 37.02, "Montant total de la facture (en DT TTC) | 37,020"))
        self.assertTrue(monetary_in_source("subtotal", 700.94, "SUB_TOTAL: 700.94 USD"))
        self.assertFalse(monetary_in_source("total_amount", 700.94, "SUB_TOTAL: 700.94 USD"))
        self.assertFalse(monetary_in_source("subtotal", 37.02, "Montant total de la facture (en DT TTC) | 37,020"))

    def test_00079_thousands_spaces_keep_all_five_grounded_rows(self):
        data = self.ground_batch("facture_00079", "items")
        self.assertEqual(len(data["items"]), 5)
        self.assertEqual(data["items"][0]["unit_price"], 2640.753)
        self.assertEqual([r["unit"] for r in data["items"]], ["TONNE"] * 5)
        self.assertNotIn("items:number_not_in_source", data["grounding_rejected_fields"])

    def test_numeric_guard_rejects_partial_thousands_and_invented_amounts(self):
        for separator in (" ", "\u00a0", "\u202f"):
            source = "Produit | 3 | 2" + separator + "640,753 | 7" + separator + "922,259"
            for wrong in (640.753, 2640.75, 2641.753):
                row = {"description": "Produit", "quantity": 3, "unit_price": wrong, "total": 7922.259, "source": "L1"}
                self.assertEqual(self.analyzer._ground_fields({"items": [row]}, source, {1})["items"], [])

    def test_existing_ocr_currency_letter_does_not_drop_template1_last_item(self):
        self.assertEqual(len(self.ground_batch("Template1_Instance175", "items")["items"]), 4)

    def test_00079_fallback_preserves_units_and_printed_reference(self):
        doc = self.docs["facture_00079"]
        rows = extract_unit_table_rows(doc["ocr_text"])
        kept, _ = filter_invoice_line_items(rows, doc["ocr_text"], 68350.131)
        self.assertEqual(len(kept), 5)
        self.assertTrue(all(row["unit"] == "TONNE" for row in kept))
        for before, after in zip(rows, kept):
            if before.get("reference"):
                self.assertEqual(before["reference"], after["reference"])

    def test_00158_fallback_keeps_exact_reference_cells(self):
        doc = self.docs["facture_00158"]
        rows = copy.deepcopy(doc["fields"]["items"])
        kept, _ = filter_invoice_line_items(rows, doc["ocr_text"], 12.6)
        self.assertEqual([r["reference"] for r in kept], ["A-002018-000769", "A-002018-002754"])

    def test_fallback_cannot_borrow_metadata_from_another_row(self):
        text = "Reference | Description | Unite | Qte | PU | Total\nA-1 | Produit A | kg | 1 | 2 | 2\nB-2 | Produit B | m3 | 1 | 3 | 3"
        rows = [{"description": "Produit A", "quantity": 1, "unit_price": 2, "total": 2,
                 "unit": "m3", "reference": "B-2", "source": text.splitlines()[1]}]
        kept, _ = filter_invoice_line_items(rows, text, 5)
        self.assertEqual((kept[0]["unit"], kept[0]["reference"]), ("kg", "A-1"))

    def test_matching_rule_duplicate_preserves_reference_after_llm_row(self):
        doc = self.docs["facture_00079"]
        llm_rows = self.ground_batch("facture_00079", "items")["items"]
        fallback = extract_unit_table_rows(doc["ocr_text"])
        kept, _ = filter_invoice_line_items(llm_rows + fallback, doc["ocr_text"], 68350.131)
        self.assertEqual(len(kept), 5)
        for row, expected in zip(kept, fallback):
            self.assertEqual(row["unit"], "TONNE")
            if expected.get("reference"):
                self.assertEqual(row["reference"], expected["reference"])

    def test_duplicate_numeric_strings_are_normalized_without_losing_metadata(self):
        text = "Reference | Description | Unite | Qte | PU | Total\nA-1 | Produit A | kg | 1 | 2 | 2"
        row = {"description": "Produit A", "quantity": "1", "unit_price": "2", "total": "2", "source": text.splitlines()[1]}
        kept, _ = filter_invoice_line_items([row, copy.deepcopy(row)], text, 2)
        self.assertEqual(len(kept), 1)
        self.assertEqual((kept[0]["unit"], kept[0]["reference"]), ("kg", "A-1"))

    def test_fallback_metadata_requires_a_real_source_not_just_a_matching_description(self):
        text = "Description | Qte | PU | Total\nProduit A | 1 | 2 | 2"
        row = {"description": "Produit A", "quantity": 1, "unit_price": 2, "total": 2,
               "unit": "kg", "reference": "B-2", "source": "B-2 | Produit A | kg | 1 | 2 | 2"}
        kept, _ = filter_invoice_line_items([row], text, 2)
        self.assertNotIn("unit", kept[0])
        self.assertNotIn("reference", kept[0])

    def test_reread_rejects_city_only_and_unowned_contacts(self):
        doc, trace = self.batch("Template26_Instance175", "supplier_name")
        initial = self.ground_batch("Template26_Instance175", "supplier_name")
        response = trace["generations"][1]["payload"]
        context = {"allowed_lines": set(trace["allowed_lines"]), "source_metadata": trace["source_metadata"]}
        with patch.object(self.analyzer, "_generate", return_value=response):
            result, _, _ = self.analyzer._recheck_rejected(initial, {}, [], "unchanged", {}, doc["ocr_text"], context,
                trace["fields"], {}, "frozen")
        self.assertIsNone(result["customer_address"])
        self.assertIsNone(result["supplier_phone"])
        self.assertEqual(result["supplier_address"], "496 Buck Rest Suite 598 Jenningsburgh, UT 95460 US")

    def test_reread_may_accept_complete_owned_address(self):
        doc, trace = self.batch("Template1_Instance175", "supplier_name")
        response = {"supplier_address": "16424 Timothy Mission Markville, AK 58294 US", "evidence": {"supplier_address": "L5-L6"}}
        initial = {"supplier_address": None, "evidence": {}, "grounding_rejected_fields": ["supplier_address"]}
        with patch.object(self.analyzer, "_generate", return_value={"response": json.dumps(response)}):
            result, _, _ = self.analyzer._recheck_rejected(initial, {}, [], "unchanged", {}, doc["ocr_text"],
                {"allowed_lines": set(trace["allowed_lines"]), "source_metadata": {}}, ["supplier_address"], {}, "frozen")
        self.assertEqual(result["supplier_address"], response["supplier_address"])

    def test_reread_unknown_name_cannot_fill_empty_party(self):
        initial = {"supplier_name": None, "evidence": {}, "grounding_rejected_fields": ["supplier_name"]}
        with patch.object(self.analyzer, "_generate", return_value={"response": json.dumps({"supplier_name": "ALPHA", "evidence": {"supplier_name": "L1"}})}):
            result, _, _ = self.analyzer._recheck_rejected(initial, {}, [], "", {}, "ALPHA", {"allowed_lines": {1}, "source_metadata": {}},
                                                        ["supplier_name"], {}, "frozen")
        self.assertIsNone(result["supplier_name"])


if __name__ == "__main__":
    unittest.main()
