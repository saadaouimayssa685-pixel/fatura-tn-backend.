"""Offline regressions for supplier fiscal IDs; no new OCR, retrieval or LLM calls."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import unittest

from extraction_retrieval import selected_context
from identifier_evidence import identifier_instruction
from ollama_analyzer import EXTRACTION_GROUPS, OllamaInvoiceAnalyzer


ROOT = Path(__file__).resolve().parent


class SupplierIdentifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qa = json.loads((ROOT / "qa-results/paddle-pipeline-groq-20260913-verified/facture_00008.result.json")
                           .read_text(encoding="utf-8"))
        cls.context = selected_context(cls.qa["ocr_text"],
            {k: cls.qa["evidence"].get(k, []) for k in EXTRACTION_GROUPS[1]})
        results = ROOT / "qa-results/supplier-tax-id-20260921"
        cls.before = json.loads((results / "before.json").read_text(encoding="utf-8"))
        cls.after = json.loads((results / "after.json").read_text(encoding="utf-8"))

    def setUp(self):
        self.analyzer = OllamaInvoiceAnalyzer()

    def ground(self, value, source, field="supplier_tax_id", citation="L1", **kwargs):
        return self.analyzer._ground_fields({field: value, "evidence": {field: citation}}, source, **kwargs)[field]

    def test_real_before_after_is_llm_generation_not_parsing_or_validation(self):
        self.assertEqual(self.before["context_sha256"], self.after["context_sha256"])
        self.assertEqual(self.before["sources"], self.after["sources"])
        self.assertEqual(self.before["model"], self.after["model"])
        self.assertEqual(self.after["dropped_passages"], 0)
        self.assertIn(33, self.context["allowed_lines"])
        self.assertIn("P1:L30-34", self.context["field_sources"]["vendor_tax_id"])
        self.assertEqual(self.qa["ocr_text"].splitlines()[32], "*Sous revetement | Code Tva 816153BAM 000")
        before = self.analyzer._parse_json(self.before["raw_response"])
        after = self.analyzer._parse_json(self.after["raw_response"])
        self.assertIsNone(before["supplier_tax_id"])
        self.assertIsNone(before["evidence"]["supplier_tax_id"])
        self.assertEqual(after["supplier_tax_id"], "816153BAM 000")
        self.assertEqual(after["evidence"]["supplier_tax_id"], "L33")
        grounded = self.analyzer._ground_fields(after, self.qa["ocr_text"], self.context["allowed_lines"])
        self.assertEqual(grounded["supplier_tax_id"], "816153BAM 000")
        self.assertEqual(grounded["customer_tax_id"], "1362434 /S/P/M/000")
        self.assertNotIn("supplier_tax_id", grounded["grounding_rejected_fields"])
        self.assertNotIn("customer_tax_id", grounded["grounding_rejected_fields"])
        # The name remains rejected; the address is now copied from its proven fiscal block.
        self.assertEqual(set(grounded["grounding_rejected_fields"]), {"customer_name"})
        self.assertEqual(grounded["customer_address"], "2 BS IBRAHIM AL MAZEJ CITE OLYMPIQUE")

    def test_complete_compact_identifiers_are_not_specific_to_one_invoice(self):
        for value in ("816153BAM 000", "731209CAB 002", "AB-987654/M/02", "8765432A/M/000"):
            with self.subTest(value=value):
                self.assertEqual(self.ground(value, "Fournisseur Code TVA: " + value), value)

    def test_truncated_identifiers_are_rejected_not_completed(self):
        for value in ("816153", "816153BAM", "816153BAM 00"):
            with self.subTest(value=value):
                self.assertIsNone(self.ground(value, "Code Tva 816153BAM 000"))

    def test_invented_digits_or_separators_are_rejected(self):
        for value in ("0816153BAM 000", "816153B/A/M/000", "816153B-AM-000"):
            with self.subTest(value=value):
                self.assertIsNone(self.ground(value, "Code Tva 816153BAM 000"))
        self.assertIsNone(self.ground("AB/987654/M/02", "MF: AB-987654/M/02"))
        self.assertIsNone(self.ground("AB-987654M02", "MF: AB-987654/M/02"))

    def test_supplier_and_customer_cannot_be_swapped(self):
        text = "Fournisseur ALPHA\nCode TVA: 731209CAB 002\nClient BETA\nMF: 1362434/S/P/M/000"
        self.assertEqual(self.ground("731209CAB 002", text, citation="L2"), "731209CAB 002")
        self.assertIsNone(self.ground("1362434/S/P/M/000", text, citation="L4"))
        self.assertIsNone(self.ground("731209CAB 002", text, field="customer_tax_id", citation="L2"))

    def test_absence_cannot_be_filled_from_customer(self):
        text = "Client BETA\nMF: 1362434/S/P/M/000"
        self.assertIsNone(self.ground(None, text, citation="L2"))
        self.assertIsNone(self.ground("731209CAB 002", text, citation="L2"))
        self.assertIsNone(self.ground("1362434/S/P/M/000", text, citation="L2"))
        prompt = identifier_instruction(["supplier_tax_id"])
        self.assertIn("Si la mention fiscale manque", prompt)
        self.assertIn("Ne deduis ni la valeur ni son proprietaire du matricule client", prompt)

    def test_missing_or_unseen_citation_cannot_support_a_complete_id(self):
        text = "Fournisseur ALPHA\nCode TVA: 731209CAB 002"
        self.assertIsNone(self.ground("731209CAB 002", text, citation="L1"))
        self.assertIsNone(self.ground("731209CAB 002", text, citation="L2", allowed_lines={1}))

    def test_full_value_survives_normalization_and_final_guard(self):
        from test_hybrid_extraction import pipeline_functions
        scope = pipeline_functions()
        raw = self.analyzer._parse_json(self.after["raw_response"])
        for value, expected in (("816153BAM 000", "816153BAM 000"), ("816153BAM", "")):
            parsed = copy.deepcopy(raw)
            parsed["supplier_tax_id"] = value
            grounded = self.analyzer._ground_fields(parsed, self.qa["ocr_text"], self.context["allowed_lines"])
            model = self.analyzer._to_invoice_data(grounded, self.after["raw_response"])
            normalized = scope["normalize_invoice_data"](model, self.qa["ocr_text"], self.qa["rule_candidates"])
            result, _ = scope["enforce_evidence_guard"](normalized, self.qa["ocr_text"],
                                                       self.qa["evidence"], self.qa["rule_candidates"])
            self.assertEqual(result["vendor_tax_id"], expected)
            self.assertEqual(result["customer_tax_id"], "1362434 /S/P/M/000")

    def test_instruction_is_supplier_only_with_no_fixture_values(self):
        prompt = identifier_instruction(["supplier_tax_id"])
        self.assertIn("Pour supplier_tax_id uniquement", prompt)
        self.assertIn("Code TVA", prompt)
        self.assertIn("n'est pas une certification de son format administratif", prompt)
        for value in ("816153", "00008", "S.T.FOOM", "L33", "1362434"):
            self.assertNotIn(value, prompt)
        for field, expected in {
            "customer_tax_id": "3d6dbd4fdd0178299c0a406a3bd9185180394ff185e7410d3dd76f4215e1d726",
            "invoice_number": "86d313fd7061eb21b881b31e350e7e9654839932cc8af904637573b8a4920af8",
            "rib": "0df73ab0f84233619bd9a75c073b1323229b171641d05483a3f666d03a2b113e",
        }.items():
            self.assertEqual(hashlib.sha256(identifier_instruction([field]).encode()).hexdigest(), expected)

    def test_server_identifier_guards_still_require_complete_owned_evidence(self):
        # Ownership is deliberately revised; freeze the safety contract, not its AST.
        text = "Fournisseur ALPHA\nMF: 8765432A/M/000\nSHIP TO:\nMF: 1234567/B/M/000"
        self.assertEqual(self.ground("8765432A/M/000", text, citation="L2"), "8765432A/M/000")
        for value, citation in (("8765432", "L2"), ("8765432A/M", "L2"), ("1234567/B/M/000", "L4")):
            self.assertIsNone(self.ground(value, text, citation=citation))
        self.assertIsNone(self.ground("8765432A/M/000", "MF: 8765432A/M/000"))


if __name__ == "__main__":
    unittest.main()
