import ast
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ollama_analyzer import EXTRACTION_GROUPS, OllamaInvoiceAnalyzer
from extraction_retrieval import selected_context


ROOT = Path(__file__).resolve().parent
QA = ROOT / "qa-results/paddle-pipeline-groq-20260913-verified/facture_00008.result.json"


class IdentifierEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = OllamaInvoiceAnalyzer()

    def ground(self, field, value, text, citation="L1", **kwargs):
        return self.analyzer._ground_fields(
            {field: value, "evidence": {field: citation}}, text, **kwargs)[field]

    def test_truncated_tax_id_is_rejected_not_completed(self):
        for value in ("1362434", "1362434/A", "1362434/A/M"):
            with self.subTest(value=value):
                self.assertIsNone(self.ground("customer_tax_id", value, "MF : 1362434/A/M/000"))

    def test_complete_tax_id_is_accepted_unchanged(self):
        value = "1362434/A/M/000"
        self.assertEqual(self.ground("customer_tax_id", value, "Client MF : " + value), value)

    def test_invoice_and_bank_identifiers_keep_letters_separators_and_zeros(self):
        for field, value, source in (
            ("invoice_number", "FA-00008/A", "Facture N: FA-00008/A"),
            ("rib", "TN59 1000 6035 1835 9847 8831", "IBAN: TN59 1000 6035 1835 9847 8831"),
            ("rib", "01 234 0000123456789 10", "RIB: 01 234 0000123456789 10"),
            ("supplier_tax_id", "1547376 Z/N/C/000", "Fournisseur MF : 1547376 Z/N/C/000"),
            ("customer_tax_id", "1362434/S", "Client MF : 1362434/S"),
        ):
            with self.subTest(field=field):
                self.assertEqual(self.ground(field, value, source), value)

    def test_missing_changed_separators_and_substrings_are_rejected(self):
        for field, value, source in (
            ("supplier_tax_id", "1362434AM000", "MF: 1362434/A/M/000"),
            ("supplier_tax_id", "1362434-A-M-000", "MF: 1362434/A/M/000"),
            ("invoice_number", "00008", "Facture: FA-00008/A"),
            ("invoice_number", "FA/00008/A", "Facture: FA-00008/A"),
            ("invoice_number", "FA-00008", "Facture: FA-00008 / A"),
            ("rib", "TN59 1000", "IBAN: TN59 1000 6035 1835 9847 8831"),
            ("rib", "234 0000123456789 10", "RIB: 01 234 0000123456789 10"),
        ):
            with self.subTest(value=value):
                self.assertIsNone(self.ground(field, value, source))

    def test_identifier_absent_or_numeric_json_is_not_invented(self):
        self.assertIsNone(self.ground("supplier_tax_id", "1362434/A/M/000", "MF : absent"))
        self.assertIsNone(self.ground("invoice_number", 158, "Facture 00158"))

    def test_roles_are_not_interchangeable(self):
        text = "Fournisseur : ALPHA\nMF: 1111111/A/M/000\nClient : BETA\nMF: 2222222/B/M/000"
        for field, own, ref, wrong, wrong_ref in (
            ("supplier_tax_id", "1111111/A/M/000", "L2", "2222222/B/M/000", "L4"),
            ("customer_tax_id", "2222222/B/M/000", "L4", "1111111/A/M/000", "L2"),
        ):
            self.assertEqual(self.ground(field, own, text, ref), own)
            self.assertIsNone(self.ground(field, wrong, text, wrong_ref))

    def test_same_line_roles_use_nearest_label(self):
        text = "Fournisseur MF: 1111111/A/M/000 | Client MF: 2222222/B/M/000"
        self.assertEqual(self.ground("customer_tax_id", "2222222/B/M/000", text), "2222222/B/M/000")
        self.assertIsNone(self.ground("supplier_tax_id", "2222222/B/M/000", text))

    def test_existing_section_metadata_is_used_without_guessing(self):
        text = "MF: 1362434/A/M/000"
        metadata = {"chunk-1": {"section": "customer", "line_ids": [1]}}
        self.assertIsNone(self.ground("supplier_tax_id", "1362434/A/M/000", text, source_metadata=metadata))
        self.assertEqual(self.ground("customer_tax_id", "1362434/A/M/000", text,
                                    source_metadata=metadata), "1362434/A/M/000")

    def test_role_from_unseen_line_is_not_used(self):
        text = "Client BETA\nMF: 1362434/A/M/000"
        self.assertIsNone(self.ground("supplier_tax_id", "1362434/A/M/000", text, "L2", allowed_lines={1, 2}))
        self.assertIsNone(self.ground("supplier_tax_id", "1362434/A/M/000", text, "L2", allowed_lines={2}))

    def test_citation_to_unseen_line_or_literal_quote_is_still_rejected(self):
        text = "FACTURE\nMF: 1362434/A/M/000"
        self.assertIsNone(self.ground("customer_tax_id", "1362434/A/M/000", text, "L2", allowed_lines={1}))
        self.assertIsNone(self.ground("customer_tax_id", "1362434/A/M/000", text,
                                     "MF: 1362434/A/M/000", allowed_lines={1, 2}))

    def test_unlabelled_multiple_tax_ids_require_review(self):
        text = "MF: 1111111/A/M/000\nMF: 2222222/B/M/000"
        self.assertIsNone(self.ground("customer_tax_id", "1111111/A/M/000", text))

    def test_qa_00008_raw_response_is_already_truncated_and_full_copy_survives(self):
        qa = json.loads(QA.read_text(encoding="utf-8"))
        group = EXTRACTION_GROUPS[1]
        context = selected_context(qa["ocr_text"], {k: qa["evidence"].get(k, []) for k in group})
        raw = json.loads(qa["llm"]["additional_data"]["ollama_raw_response"])[1]
        parsed = self.analyzer._parse_json(raw)
        self.assertEqual(parsed["customer_tax_id"], "1362434")
        self.assertEqual(parsed["evidence"]["customer_tax_id"], "L13")
        self.assertIn(13, context["allowed_lines"])
        complete = "1362434 /S/P/M/000"
        self.assertIn(complete, qa["ocr_text"].splitlines()[12])
        rejected = self.analyzer._ground_fields(copy.deepcopy(parsed), qa["ocr_text"], context["allowed_lines"])
        self.assertIsNone(rejected["customer_tax_id"])
        self.assertIn("customer_tax_id", rejected["grounding_rejected_fields"])
        parsed["customer_tax_id"] = complete
        accepted = self.analyzer._ground_fields(parsed, qa["ocr_text"], context["allowed_lines"])
        self.assertEqual(accepted["supplier_tax_id"], "816153BAM 000")
        self.assertEqual(accepted["customer_tax_id"], complete)
        model = self.analyzer._to_invoice_data(accepted, raw)
        self.assertEqual(model.customer.vat_number, complete)
        self.assertEqual(model.supplier.vat_number, "816153BAM 000")

    def test_qa_00040_ambiguous_footer_cannot_be_asserted_as_customer(self):
        qa = json.loads((ROOT / "qa-results/groq-multisource-20260912-replay/facture_00040.result.json")
                        .read_text(encoding="utf-8"))
        self.assertIsNone(self.ground("customer_tax_id", "1144983D/A/M/000", qa["ocr_text"], "L24"))

    def test_rejected_value_stays_null_through_batch_merge(self):
        text = "Client : BETA\nMF: 1362434/A/M/000"
        proof = {"text": text, "source_id": "P1:L1-2", "line_start": 1, "line_end": 2}
        payload = {"customer_name": "BETA", "customer_tax_id": "1362434", "evidence": {
            "customer_name": "L1", "customer_tax_id": "L2"}}
        diagnostics = {}
        with patch.object(self.analyzer, "_generate", return_value={"response": json.dumps(payload)}):
            result = self.analyzer.analyze_invoice_text(text, {"customer_tax_id": [proof]}, diagnostics=diagnostics)
        self.assertIsNone(result.customer.vat_number)
        self.assertIsNone(result.additional_data["ollama_result"]["customer_tax_id"])
        self.assertIn("customer_tax_id", diagnostics["batches"][1]["rejected_fields"])

    def test_normalization_and_final_guard_preserve_full_id_and_rejected_abstention(self):
        from test_hybrid_extraction import pipeline_functions
        scope = pipeline_functions()
        text = "Client : BETA\nMF: 1362434/A/M/000"
        rules = {"fields": {"customer_tax_id": "1362434"}, "items": []}
        for value, expected in (("1362434", ""), ("1362434/A/M/000", "1362434/A/M/000")):
            parsed = self.analyzer._ground_fields({"customer_name": "BETA", "customer_tax_id": value,
                "evidence": {"customer_name": "L1", "customer_tax_id": "L2"}}, text, {1, 2})
            model = self.analyzer._to_invoice_data(parsed, "{}")
            normalized = scope["normalize_invoice_data"](model, text, rules)
            result, _ = scope["enforce_evidence_guard"](normalized, text, {}, rules)
            self.assertEqual(result["customer_tax_id"], expected)

    def test_identifier_instruction_does_not_leak_into_amount_or_item_prompts(self):
        prompt = self.analyzer._build_prompt("MF: 1362434/A/M/000", fields=["customer_tax_id"])
        self.assertIn("copie la valeur COMPLETE", prompt)
        self.assertIn("Preserve lettres, zeros initiaux, /, - et TOUS les suffixes", prompt)
        for fields, expected in (
            (["subtotal", "tax_amount", "stamp_duty", "total_amount"],
             "0e78e11b6a47598dad4e85e72613c46c24e61c9a3a1ab4282d700d3d9f68c722"),
            # Approved schema addition: existing InvoiceItem.unit is now requested.
            (["items"], "d75da326dc7340c725a4d2914bd30eb46dc701d90edcdf3d6b4d800a8fb9cc26"),
        ):
            prompt = self.analyzer._build_prompt("TVA 1.000\nArticle 1 2.000", fields=fields)
            self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(), expected)

    def test_chatbot_methods_are_unchanged(self):
        module = ast.parse((ROOT / "ollama_analyzer.py").read_text(encoding="utf-8"))
        methods = {n.name: n for cls in module.body if isinstance(cls, ast.ClassDef)
                   for n in cls.body if isinstance(n, ast.FunctionDef)}
        for name, expected in {
            "answer_sources": "7acf4566370b540694f7c0624981b10d1d034f9edd6ef786ecba9dffdd1febac",
            "_answer_is_grounded": "905c423df572c36004e39850643ad27df44831c01799b9aa24d9081ccb4ca75b",
            "_answer_labeled_amount": "404182ff8ba88a1df52d33bd1163cce72757ab3e14cc7f755c959d8c9f20c7ad",
        }.items():
            self.assertEqual(hashlib.sha256(ast.dump(methods[name]).encode()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
