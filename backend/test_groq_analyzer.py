import json
import os
import tempfile
import unittest
from unittest.mock import patch

from extraction_retrieval import source_passages
from groq_analyzer import GroqInvoiceAnalyzer, strict_schema
from ollama_analyzer import extraction_schema


class GroqExtractionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.env = patch.dict(os.environ, {"GROQ_API_KEY": "unit-test-key", "GROQ_RESUME_DIR": directory.name})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_schema_requires_keys_but_allows_missing_values_and_proofs(self):
        original = extraction_schema()
        schema = strict_schema(original)
        self.assertNotEqual(original["required"], schema["required"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertIn("null", schema["properties"]["total_amount"]["type"])
        proof = schema["properties"]["evidence"]
        self.assertFalse(proof["additionalProperties"])
        self.assertIn("null", proof["properties"]["total_amount"]["type"])
        row = schema["properties"]["items"]["items"]
        self.assertEqual(set(row["required"]), set(row["properties"]))

    def test_prompt_does_not_contradict_required_nullable_schema(self):
        text = "TOTAL TTC: 24.800"
        prompt = GroqInvoiceAnalyzer()._build_prompt(text,
            {"total_amount": source_passages(text)}, fields=["subtotal", "tax_amount", "total_amount"])
        self.assertNotIn("Omet les champs absents", prompt)
        self.assertIn("TOUS les champs demandes", prompt)
        self.assertIn('"subtotal":null', prompt)
        self.assertIn('"evidence":{"subtotal":null', prompt)

    @patch("groq_analyzer.requests.post")
    def test_cloud_receives_only_retrieved_source_and_checks_citation(self, post):
        text = "--- Page 1 ---\nUNRELATED PRIVATE HEADER\n--- Page 2 ---\nTOTAL TTC 24.800"
        post.return_value.status_code = 200
        post.return_value.json.return_value = {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"total_amount": 24.8, "subtotal": None, "tax_amount": None,
                "stamp_duty": None, "evidence": {"total_amount": "L4", "subtotal": None,
                    "tax_amount": None, "stamp_duty": None}})}}], "usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        diagnostics = {}
        result = GroqInvoiceAnalyzer().analyze_invoice_text(text,
            {"total_amount": source_passages(text)[-1:]}, diagnostics=diagnostics)
        self.assertEqual(result.total_amount, 24.8)
        self.assertIsNone(result.subtotal)
        body = post.call_args.kwargs["json"]
        self.assertNotIn("PRIVATE HEADER", json.dumps(body))
        self.assertNotIn("unit-test-key", json.dumps(body))
        self.assertEqual(body["model"], "openai/gpt-oss-120b")
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(diagnostics["provider"], "groq")
        self.assertEqual(result.additional_data["analysis_method"], "groq_rag")

    @patch("groq_analyzer.requests.post")
    def test_product_batch_uses_row_citations_without_empty_evidence_object(self, post):
        text = "Designation Quantite Prix Total\nProduit A 2 3.000 6.000"
        post.return_value.status_code = 200
        post.return_value.json.return_value = {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"items": [{"description": "Produit A", "quantity": 2,
                "unit_price": 3, "total": 6, "source": "L2"}]})}}]}
        result = GroqInvoiceAnalyzer().analyze_invoice_text(text, {"items": source_passages(text)})
        self.assertEqual(len(result.items), 1)
        schema = post.call_args.kwargs["json"]["response_format"]["json_schema"]["schema"]
        self.assertEqual(list(schema["properties"]), ["items"])
        self.assertNotIn("evidence", schema["properties"])

    @patch("groq_analyzer.requests.post")
    def test_no_key_means_no_network_call(self, post):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            diagnostics = {}
            result = GroqInvoiceAnalyzer().analyze_invoice_text("FACTURE 001",
                {"invoice_number": source_passages("FACTURE 001")}, diagnostics=diagnostics)
        self.assertIsNone(result)
        self.assertEqual(diagnostics["status"], "missing_api_key")
        post.assert_not_called()

    @patch("groq_analyzer.requests.post")
    def test_quota_is_reported_without_retry_or_local_model_call(self, post):
        post.return_value.status_code = 429
        diagnostics = {}
        result = GroqInvoiceAnalyzer().analyze_invoice_text("FACTURE 001",
            {"invoice_number": source_passages("FACTURE 001")}, diagnostics=diagnostics)
        self.assertIsNone(result)
        self.assertEqual(diagnostics["status"], "rate_limited")
        post.assert_called_once()

    @patch("groq_analyzer.requests.post")
    def test_truncated_cloud_json_is_rejected(self, post):
        post.return_value.status_code = 200
        post.return_value.json.return_value = {"choices": [{"finish_reason": "length", "message": {
            "content": '{"invoice_number":"001"}'}}]}
        diagnostics = {}
        result = GroqInvoiceAnalyzer().analyze_invoice_text("FACTURE 001",
            {"invoice_number": source_passages("FACTURE 001")}, diagnostics=diagnostics)
        self.assertIsNone(result)
        self.assertEqual(diagnostics["status"], "incomplete_output")


if __name__ == "__main__":
    unittest.main()
