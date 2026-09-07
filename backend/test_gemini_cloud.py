import json
import unittest
from unittest.mock import patch, Mock
from gemini_cloud import GeminiCloud


class GeminiTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.dict("os.environ", {"GEMINI_API_KEY": "test-secret", "GEMINI_MODEL": "gemini-2.5-flash-lite"})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.client = GeminiCloud()

    @patch("gemini_cloud.requests.post")
    def test_quota_is_reported_without_secret(self, post):
        post.return_value = Mock(status_code=429)
        with self.assertRaisesRegex(RuntimeError, "Quota") as error:
            self.client.generate_json("instructions", {})
        self.assertNotIn("test-secret", str(error.exception))
        self.assertNotIn("test-secret", post.call_args.args[0])

    @patch("gemini_cloud.requests.post")
    def test_truncated_response_rejected(self, post):
        post.return_value = Mock(status_code=200, ok=True)
        post.return_value.json.return_value = {"candidates": [{"finishReason": "MAX_TOKENS"}]}
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.client.generate_json("instructions", {})

    def test_no_source_does_not_call_cloud(self):
        with patch.object(self.client, "generate_json") as generate:
            result = self.client.answer_sources("Total?", [])
            generate.assert_not_called()
            self.assertEqual(result["citations"], [])

    def test_invalid_source_reference_rejected(self):
        with patch.object(self.client, "generate_json", return_value={"answer": "Fake [8]", "citations": [8]}):
            with self.assertRaisesRegex(RuntimeError, "Citations"):
                self.client.answer_sources("Total?", [{"text": "TTC 24,800"}])

    def test_missing_evidence_abstains(self):
        with patch.object(self.client, "generate_json", return_value={"answer": "Unproven", "citations": []}):
            result = self.client.answer_sources("Total?", [{"text": "No amounts"}])
            self.assertNotEqual(result["answer"], "Unproven")

    def test_invoice_prompt_includes_field_rag_evidence(self):
        with patch.object(self.client, "generate_json", return_value={"invoice_number": "F-1", "items": []}) as generate:
            result = self.client.analyze_invoice_text(
                "FACTURE F-1",
                field_evidence={"invoice_number": [{"page_number": 1, "text": "FACTURE F-1"}]},
            )
        self.assertEqual(result.invoice_number, "F-1")
        payload = generate.call_args.args[1]
        self.assertEqual(payload["rag_evidence_by_field"]["invoice_number"][0]["passage"], "FACTURE F-1")


if __name__ == "__main__":
    unittest.main()
