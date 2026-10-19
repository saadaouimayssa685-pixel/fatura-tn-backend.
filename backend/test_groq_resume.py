import copy
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from extraction_retrieval import source_passages
from groq_analyzer import GroqInvoiceAnalyzer
from groq_resume import GroqResumeStore, fingerprint, rate_limit_details, retry_after_seconds
from test_hybrid_extraction import pipeline_functions


def response(content=None, status=200, headers=None, error=None):
    result = Mock(status_code=status, headers=headers or {})
    result.json.return_value = ({"error": error or {"type": "tokens", "code": "rate_limit_exceeded", "message": "Quota reached"}}
                               if status != 200 else {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}],
                                                      "usage": {"prompt_tokens": 100, "completion_tokens": 25}})
    return result


class GroqResumeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        env = patch.dict(os.environ, {"GROQ_API_KEY": "unit-test-key", "GROQ_RESUME_DIR": self.directory.name})
        env.start()
        self.addCleanup(env.stop)
        self.analyzer = GroqInvoiceAnalyzer()
        self.store = GroqResumeStore()
        self.text = "Facture 005\nTotal TTC 12.600"
        self.evidence = {"invoice_number": source_passages(self.text), "total_amount": source_passages(self.text)}
        self.identity = {"invoice_number": "005", "evidence": {"invoice_number": "L1"}}
        self.total = {"total_amount": 12.6, "evidence": {"total_amount": "L2"}}

    def run_analysis(self, post, outcomes, analyzer=None, text=None, evidence=None):
        post.side_effect = outcomes
        diagnostics = {}
        result = (analyzer or self.analyzer).analyze_invoice_text(text or self.text,
                    self.evidence if evidence is None else evidence, diagnostics=diagnostics)
        return result, diagnostics

    def test_retry_after_numeric_date_and_invalid_values(self):
        self.assertEqual(retry_after_seconds({"retry-after": "2.5"}, 100), 2.5)
        self.assertEqual(retry_after_seconds({"retry-after": "0"}, 100), 0)
        start = datetime(2026, 10, 12, 10, tzinfo=timezone.utc).timestamp()
        self.assertEqual(retry_after_seconds({"retry-after": "Mon, 12 Oct 2026 10:00:20 GMT"}, start), 20)
        self.assertEqual(retry_after_seconds({"retry-after": "Mon, 12 Oct 2026 10:00:20 GMT",
                         "date": "Mon, 12 Oct 2026 10:00:00 GMT"}, start + 100), 20)
        for value in (None, "", "later", "-1", "NaN", "inf", "1m30s"):
            self.assertIsNone(retry_after_seconds({"retry-after": value}, start), value)

    def test_capture_full_provider_error_and_quota_headers_not_secrets(self):
        data = rate_limit_details(response(status=429, headers={"Retry-After": "20", "X-RateLimit-Limit-Tokens": "8000",
            "X-RateLimit-Reset-Tokens": "1m", "X-Request-Id": "req-123", "Set-Cookie": "secret", "Authorization": "unit-test-key"},
            error={"message": "Quota unit-test-key", "type": "tokens", "code": "rate_limit_exceeded"}), "unit-test-key", 100)
        self.assertEqual(data["retry_not_before"], 120)
        self.assertEqual(data["headers"]["x-ratelimit-reset-tokens"], "1m")
        self.assertEqual(data["error"]["message"], "Quota [REDACTED]")
        self.assertNotIn("unit-test-key", json.dumps(data))
        self.assertNotIn("secret", json.dumps(data))

    @patch("groq_analyzer.time.sleep")
    @patch("groq_analyzer.requests.post")
    def test_unknown_reset_means_no_automatic_retry(self, post, sleep):
        result, diag = self.run_analysis(post, [response(status=429, headers={"X-RateLimit-Reset-Tokens": "1m"})])
        self.assertIsNone(result)
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()
        self.assertTrue(diag["groq_transport"]["resume"]["manual_retry_required"])
        self.assertIsNone(diag["groq_transport"]["last_rate_limit"]["retry_after_seconds"])

    @patch("groq_analyzer.time.sleep")
    @patch("groq_analyzer.requests.post")
    def test_one_retry_same_request_when_explicit_delay_is_short(self, post, sleep):
        result, diag = self.run_analysis(post, [response(status=429, headers={"Retry-After": "2"}), response(self.identity), response(self.total)])
        self.assertEqual(result.total_amount, 12.6)
        self.assertEqual(post.call_args_list[0].kwargs["json"], post.call_args_list[1].kwargs["json"])
        sleep.assert_called_once_with(2)
        self.assertEqual(diag["groq_transport"]["automatic_retries"], 1)
        self.assertFalse(diag["groq_transport"]["resume"]["pending"])
        self.assertEqual(list(Path(self.directory.name).glob("*.json")), [])

    @patch("groq_analyzer.time.sleep")
    @patch("groq_analyzer.requests.post")
    def test_retry_budget_is_global_for_document_not_reset_per_batch(self, post, sleep):
        result, diag = self.run_analysis(post, [response(status=429, headers={"Retry-After": "1"}), response(self.identity),
                                               response(status=429, headers={"Retry-After": "1"})])
        self.assertEqual(result.invoice_number, "005")
        self.assertEqual(post.call_count, 3)
        sleep.assert_called_once_with(1)
        self.assertEqual(diag["status"], "completed_partial")

    @patch("groq_analyzer.time.sleep")
    @patch("groq_analyzer.requests.post")
    def test_repeated_429_is_bounded_and_preserves_reset(self, post, sleep):
        _, diag = self.run_analysis(post, [response(status=429, headers={"Retry-After": "0"}),
                                          response(status=429, headers={"Retry-After": "5"})])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(diag["groq_transport"]["last_rate_limit"]["retry_after_seconds"], 5)
        self.assertTrue(diag["groq_transport"]["resume"]["pending"])

    @patch("groq_analyzer.time.time", return_value=100)
    @patch("groq_analyzer.time.sleep")
    @patch("groq_analyzer.requests.post")
    def test_long_wait_and_restart_never_send_early(self, post, sleep, clock):
        _, first = self.run_analysis(post, [response(self.identity), response(status=429, headers={"Retry-After": "120"})])
        self.assertEqual(first["groq_transport"]["resume"]["retry_not_before"], 220)
        post.reset_mock()
        result, resumed = self.run_analysis(post, [], analyzer=GroqInvoiceAnalyzer())
        self.assertEqual(result.invoice_number, "005")
        post.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual(resumed["groq_transport"]["reused_responses"], 1)
        self.assertEqual(resumed["groq_transport"]["last_rate_limit"], first["groq_transport"]["last_rate_limit"])

    @patch("groq_analyzer.requests.post")
    def test_restart_reuses_success_only_and_checks_grounding_again(self, post):
        self.run_analysis(post, [response(self.identity), response(status=429)])
        post.reset_mock()
        fresh = GroqInvoiceAnalyzer()
        with patch.object(fresh, "_ground_fields", wraps=fresh._ground_fields) as ground:
            result, diag = self.run_analysis(post, [response(self.total)], analyzer=fresh)
        self.assertEqual(result.invoice_number, "005")
        self.assertEqual(result.total_amount, 12.6)
        self.assertEqual(ground.call_count, 2)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(diag["groq_transport"]["reused_responses"], 1)
        self.assertNotIn("invoice_number", post.call_args.kwargs["json"]["response_format"]["json_schema"]["schema"]["properties"])

    @patch("groq_analyzer.requests.post")
    def test_completed_extraction_does_not_cache_a_future_reextract(self, post):
        self.run_analysis(post, [response(self.identity), response(self.total)])
        post.reset_mock()
        self.run_analysis(post, [response(self.identity), response(self.total)])
        self.assertEqual(post.call_count, 2)

    @patch("groq_analyzer.requests.post")
    def test_invalidation_text_evidence_model_and_credential(self, post):
        self.run_analysis(post, [response(self.identity), response(status=429)])
        original = self.analyzer.checkpoint_id(self.text, self.evidence)
        self.assertNotEqual(original, self.analyzer.checkpoint_id(self.text + "\nOther", self.evidence))
        self.assertNotEqual(original, self.analyzer.checkpoint_id(self.text, {}))
        self.analyzer.model_name = "different-model"
        self.assertNotEqual(original, self.analyzer.checkpoint_id(self.text, self.evidence))
        self.analyzer = GroqInvoiceAnalyzer()
        with patch.dict(os.environ, {"GROQ_API_KEY": "different-key"}):
            self.assertNotEqual(original, self.analyzer.checkpoint_id(self.text, self.evidence))
        with patch.dict(os.environ, {"GROQ_MAX_OUTPUT_TOKENS": "1500"}):
            self.assertNotEqual(original, self.analyzer.checkpoint_id(self.text, self.evidence))

    @patch("groq_analyzer.requests.post")
    def test_prompt_change_does_not_reuse_old_response(self, post):
        self.run_analysis(post, [response(self.identity), response(status=429)])
        build = self.analyzer._build_prompt
        with patch.object(self.analyzer, "_build_prompt", side_effect=lambda *a, **k: build(*a, **k) + "\nDifferent test instruction"):
            post.reset_mock()
            _, diag = self.run_analysis(post, [response(self.identity), response(self.total)])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(diag["groq_transport"]["reused_responses"], 0)

    @patch("groq_analyzer.requests.post")
    def test_quota_during_recheck_preserves_first_answer_and_stops_later_calls(self, post):
        invalid = {"invoice_number": "WRONG", "evidence": {"invoice_number": "L1"}}
        _, first = self.run_analysis(post, [response(invalid), response(status=429)])
        self.assertEqual(post.call_count, 2)
        self.assertTrue(first["groq_transport"]["resume"]["pending"])
        post.reset_mock()
        result, diag = self.run_analysis(post, [response(self.identity), response(self.total)], analyzer=GroqInvoiceAnalyzer())
        self.assertEqual(result.invoice_number, "005")
        self.assertEqual(diag["groq_transport"]["reused_responses"], 1)
        self.assertEqual(post.call_count, 2)

    @patch("groq_analyzer.requests.post")
    def test_last_batch_recheck_quota_is_not_reported_completed(self, post):
        text = "Total TTC 12.600"
        bad = {"total_amount": 999, "evidence": {"total_amount": "L1"}}
        _, diag = self.run_analysis(post, [response(bad), response(status=429)], text=text,
                                   evidence={"total_amount": source_passages(text)})
        self.assertIn(diag["status"], {"completed_partial", "rate_limited"})
        self.assertTrue(diag["groq_transport"]["resume"]["pending"])
        self.assertEqual(post.call_count, 2)

    @patch("groq_analyzer.requests.post")
    def test_timeout_preserves_previously_received_batch_without_retry(self, post):
        result, diag = self.run_analysis(post, [response(self.identity), requests.Timeout()])
        self.assertEqual(result.invoice_number, "005")
        self.assertEqual(diag["groq_transport"]["resume"]["reason"], "timeout")
        self.assertEqual(post.call_count, 2)

    @patch("groq_analyzer.requests.post")
    def test_corrupt_checkpoint_fails_closed_without_paid_request(self, post):
        self.run_analysis(post, [response(self.identity), response(status=429)])
        path = self.store.path(self.analyzer.checkpoint_id(self.text, self.evidence))
        state = json.loads(path.read_text(encoding="utf-8"))
        next(iter(state["responses"].values()))["payload"]["response"] = '{"invoice_number":"fake"}'
        path.write_text(json.dumps(state), encoding="utf-8")
        post.reset_mock()
        result, diag = self.run_analysis(post, [])
        self.assertIsNone(result)
        self.assertEqual(diag["status"], "resume_storage_error")
        post.assert_not_called()

    @patch("groq_analyzer.requests.post")
    def test_malformed_checkpoint_shape_or_deadline_stops_without_request(self, post):
        identity = self.analyzer.checkpoint_id(self.text, self.evidence)
        valid = self.store.load(identity)
        for state in ([], {**valid, "blocked": []}, {**valid, "blocked": {"retry_not_before": "invalid"}},
                      {**valid, "blocked": {"retry_not_before": float("nan")}}):
            with self.subTest(state=state):
                self.store.path(identity).write_text(json.dumps(state), encoding="utf-8")
                result, diag = self.run_analysis(post, [])
                self.assertIsNone(result)
                self.assertEqual(diag["status"], "resume_storage_error")
                post.assert_not_called()

    @patch("groq_analyzer.requests.post")
    def test_checkpoint_storage_failure_stops_further_calls(self, post):
        with patch.object(GroqResumeStore, "save", side_effect=OSError("read only")):
            _, diag = self.run_analysis(post, [response(self.identity)])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(diag["groq_transport"]["resume"]["reason"], "resume_storage_error")

    @patch("groq_analyzer.requests.post")
    def test_concurrent_same_document_is_not_sent_twice(self, post):
        identity = self.analyzer.checkpoint_id(self.text, self.evidence)
        with self.store.lock(identity):
            result, diag = self.run_analysis(post, [])
        self.assertIsNone(result)
        self.assertEqual(diag["status"], "resume_in_progress")
        post.assert_not_called()

    @patch("groq_analyzer.requests.post")
    def test_concurrent_different_documents_do_not_share_checkpoint_context(self, post):
        def generate(url, **kwargs):
            prompt = kwargs["json"]["messages"][1]["content"]
            value = "005" if "Facture 005" in prompt else "006"
            return response({"invoice_number": value, "evidence": {"invoice_number": "L1"}})
        post.side_effect = generate
        def run(value):
            text = "Facture " + value
            diag = {}
            result = self.analyzer.analyze_invoice_text(text, {"invoice_number": source_passages(text)}, diagnostics=diag)
            return result.invoice_number, diag["groq_transport"]["http_call_count"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(run, ["005", "006"])), [("005", 1), ("006", 1)])

    @patch("groq_analyzer.requests.post")
    def test_00008_four_recorded_batches_survive_restart_only_items_requested(self, post):
        path = Path("qa-results/fresh-final-smoke-20260922T093925Z/facture_00008.json")
        recorded = json.loads(path.read_text(encoding="utf-8"))
        accepted = [response(json.loads(t["generations"][0]["payload"]["response"])) for t in recorded["trace"][:4]]
        first, diag = self.run_analysis(post, accepted + [response(status=429)],
                                       text=recorded["ocr_text"], evidence=recorded["evidence"])
        self.assertEqual(post.call_count, 5)
        self.assertEqual(first.supplier.vat_number, "816153BAM 000")
        for call, batch in zip(post.call_args_list, recorded["trace"]):
            generated = batch["generations"][0]
            self.assertEqual(call.kwargs["json"]["messages"][1]["content"], generated["prompt"])
            self.assertEqual(call.kwargs["json"], self.analyzer._request_body(generated["prompt"], generated["schema"]))
        post.reset_mock()
        before = copy.deepcopy(first)
        # Only the transport is simulated. Real grounding, mapping and finance run.
        items = {"items": [{"description": "FOURNITURE ET POSE DE BETON CELLULAIRE 3P LUXE dosage 300Kg/m³ densite 400 Kg/m³",
                           "quantity": 35, "unit_price": 145.6, "total": 5096, "unit": "m³", "source": "L16-L19"}]}
        after, resumed = self.run_analysis(post, [response(items)], analyzer=GroqInvoiceAnalyzer(),
                                          text=recorded["ocr_text"], evidence=recorded["evidence"])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(resumed["groq_transport"]["reused_responses"], 4)
        self.assertEqual(after.supplier, before.supplier)
        self.assertEqual(after.customer, before.customer)
        self.assertEqual(after.total_amount, before.total_amount)
        self.assertEqual(len(after.items), 1)
        scope = pipeline_functions()
        fields = scope["normalize_invoice_data"](after, recorded["ocr_text"], recorded["rule_candidates"])
        fields, proofs = scope["enforce_evidence_guard"](fields, recorded["ocr_text"], recorded["evidence"], recorded["rule_candidates"])
        issues = scope["validate_normalized_invoice_fields"](fields, proofs)
        self.assertEqual(fields["customer_tax_id"], "1362434 /S/P/M/000")
        self.assertEqual(fields["discount_amount"], 50.96)
        self.assertEqual(len(fields["items"]), 1)
        self.assertNotIn("no_verified_line_items", [i["code"] for i in issues])


if __name__ == "__main__":
    unittest.main()
