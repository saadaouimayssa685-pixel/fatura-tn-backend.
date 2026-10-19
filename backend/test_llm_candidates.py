import unittest
from unittest.mock import patch
import requests

from ollama_analyzer import OllamaInvoiceAnalyzer
from extraction_retrieval import source_passages, selected_context


class LocalLlmCandidatePromptTests(unittest.TestCase):
    def test_wrapped_description_and_separate_ocr_cells(self):
        text = "FOURNITURE ET POSE DE | m3 | 35 | 145,600 | 5096,000\nBETON CELLULAIRE"
        result = OllamaInvoiceAnalyzer()._ground_fields({"items": [{
            "description": "FOURNITURE ET POSE DE BETON CELLULAIRE", "quantity": 35,
            "unit_price": 145.6, "total": 5096, "source": "L1-L2"}]}, text)
        self.assertEqual(len(result["items"]), 1)

    def test_tax_rate_in_separate_cell_is_not_the_tax_amount(self):
        text = "TVA | 19% | 968,143 | Timbre | 1,000"
        result = OllamaInvoiceAnalyzer()._ground_fields({"tax_amount": 968.143,
            "evidence": {"tax_amount": "L1"}}, text)
        self.assertEqual(result["tax_amount"], 968.143)

    def test_line_citations_resolve_to_original_text(self):
        result = OllamaInvoiceAnalyzer()._ground_fields({"total_amount": 12.6,
            "evidence": {"total_amount": "L3"}, "items": [{"description": "Rouleau Cristal",
                "quantity": 1, "unit_price": 6, "total": 6, "source": "L2"}]},
            "FACTURE 00633\nRouleau Cristal 1 6.000 6.000\nTOTAL TTC 12.600")
        self.assertEqual(result["total_amount"], 12.6)
        self.assertEqual(result["evidence"]["total_amount"], "TOTAL TTC 12.600")
        self.assertEqual(result["items"][0]["source"], "Rouleau Cristal 1 6.000 6.000")

    def test_unknown_line_citation_does_not_ground_value(self):
        result = OllamaInvoiceAnalyzer()._ground_fields({"total_amount": 12.6,
            "evidence": {"total_amount": "L999"}}, "TOTAL TTC 12.600")
        self.assertIsNone(result["total_amount"])

    @patch("ollama_analyzer.requests.post")
    def test_incomplete_response_is_not_accepted_as_success(self, post):
        post.return_value.json.return_value = {"response": '{"invoice_number":"001"}', "done_reason": "length"}
        diagnostics = {}
        self.assertIsNone(OllamaInvoiceAnalyzer().analyze_invoice_text("FACTURE 001", diagnostics=diagnostics))
        self.assertEqual(diagnostics["status"], "incomplete_output")

    @patch("ollama_analyzer.requests.post", side_effect=requests.ReadTimeout())
    def test_timeout_has_explicit_diagnostics(self, post):
        diagnostics = {}
        self.assertIsNone(OllamaInvoiceAnalyzer().analyze_invoice_text("FACTURE 001", diagnostics=diagnostics))
        self.assertEqual(diagnostics["status"], "timeout")
        self.assertIn("seconds", diagnostics)

    def test_matching_number_under_wrong_label_is_rejected(self):
        result = OllamaInvoiceAnalyzer()._ground_fields({"subtotal": 12.6,
            "evidence": {"subtotal": "TOTAL TTC: 12.600"}}, "TOTAL TTC: 12.600")
        self.assertIsNone(result["subtotal"])

    def test_vat_rate_and_amount_remain_distinct(self):
        quote = "TVA (19%) : 3,800"
        for value, expected in ((3.8, 3.8), (19, None)):
            result = OllamaInvoiceAnalyzer()._ground_fields({"tax_amount": value,
                "evidence": {"tax_amount": quote}}, quote)
            self.assertEqual(result["tax_amount"], expected)

    def test_tax_identifier_spacing_does_not_erase_verified_value(self):
        text = "Fournisseur MF : 1547376 Z/N/C/000"
        result = OllamaInvoiceAnalyzer()._ground_fields({"supplier_tax_id": "1547376Z/N/C/000",
            "evidence": {"supplier_tax_id": text}}, text)
        self.assertEqual(result["supplier_tax_id"], "1547376Z/N/C/000")

    def test_retrieval_does_not_duplicate_source_text(self):
        source = "FACTURE UNIQUE N 00158 TOTAL TTC 12.600"
        proof = {"text": source, "source_id": "P1:L1-1", "line_start": 1, "line_end": 1}
        prompt = OllamaInvoiceAnalyzer()._build_prompt(source, {"total_amount": [proof], "invoice_number": [proof]})
        self.assertEqual(prompt.count(source), 1)
        self.assertIn("P1:L1-1", prompt)

    def test_prompt_requires_grounding_without_closed_candidates(self):
        analyzer = OllamaInvoiceAnalyzer()
        prompt = analyzer._build_prompt(
            "FACTURE N: 00633\nTOTAL TTC: 12.600",
            {"invoice_number": source_passages("FACTURE N: 00633\nTOTAL TTC: 12.600")},
            rule_candidates={
                "field_values": {"invoice_number": "00633", "total_amount": 12.6},
                "article_rows": [{"description": "Rouleau", "quantity": 1, "unit_price": 6, "total": 6}],
            },
        )
        self.assertNotIn("liste fermee", prompt)
        self.assertIn("00633", prompt)
        self.assertIn("citation exacte", prompt)
        self.assertIn("TVA 19% est un taux", prompt)

    def test_prompt_keeps_retrieved_end_not_entire_document(self):
        text = "\n".join("UNRELATED_HEADER " * 10 for _ in range(60)) + "\nTOTAL TTC 12,600"
        prompt = OllamaInvoiceAnalyzer()._build_prompt(text, {"total_amount": source_passages(text)[-1:]})
        self.assertIn("TOTAL TTC 12,600", prompt)
        self.assertLess(prompt.count("UNRELATED_HEADER"), text.count("UNRELATED_HEADER"))

    @patch("ollama_analyzer.requests.post")
    def test_empty_retrieval_never_sends_document(self, post):
        diagnostics = {}
        result = OllamaInvoiceAnalyzer().analyze_invoice_text("FACTURE SECRET 001", {}, diagnostics=diagnostics)
        self.assertIsNone(result)
        post.assert_not_called()
        self.assertEqual(diagnostics["status"], "no_evidence")

    def test_candidates_cannot_leak_unretrieved_values(self):
        prompt = OllamaInvoiceAnalyzer()._build_prompt("FACTURE SECRET 001", {},
            {"field_values": {"supplier_name": "SECRET COMPANY"}})
        self.assertNotIn("SECRET", prompt)

    def test_grounding_rejects_unseen_line_even_if_document_contains_it(self):
        result = OllamaInvoiceAnalyzer()._ground_fields({"total_amount": 12.6,
            "evidence": {"total_amount": "L2"}}, "FACTURE\nTOTAL TTC 12.600", allowed_lines={1})
        self.assertIsNone(result["total_amount"])

    @patch("ollama_analyzer.requests.post")
    def test_batches_are_scoped_and_unrequested_fields_rejected(self, post):
        text = "--- Page 1 ---\nFACTURE 00633\n--- Page 2 ---\nTOTAL TTC 12.600"
        passages = source_passages(text)
        post.return_value.json.side_effect = [
            {"response": '{"invoice_number":"00633","evidence":{"invoice_number":"L2"}}', "done": True},
            {"response": '{"total_amount":12.6,"supplier_name":"Invented","evidence":{"total_amount":"L4"}}', "done": True},
        ]
        diagnostics = {}
        result = OllamaInvoiceAnalyzer().analyze_invoice_text(text,
            {"invoice_number": passages[:1], "total_amount": passages[1:]}, diagnostics=diagnostics)
        self.assertEqual(result.invoice_number, "00633")
        self.assertEqual(result.total_amount, 12.6)
        self.assertIsNone(result.supplier)
        self.assertEqual(post.call_count, 2)
        self.assertNotIn("12.600", post.call_args_list[0].kwargs["json"]["prompt"])
        self.assertNotIn("00633", post.call_args_list[1].kwargs["json"]["prompt"])
        self.assertEqual(diagnostics["context_mode"], "retrieved_only")
        proof_schema = post.call_args_list[0].kwargs["json"]["format"]["properties"]["evidence"]
        self.assertFalse(proof_schema["additionalProperties"])
        self.assertNotIn("total_amount", proof_schema["properties"])

    @patch("ollama_analyzer.requests.post")
    def test_timeout_keeps_previously_verified_batch_and_marks_partial(self, post):
        text = "--- Page 1 ---\nFACTURE 00633\n--- Page 2 ---\nTOTAL TTC 12.600"
        passages = source_passages(text)
        from unittest.mock import Mock
        success = Mock()
        success.json.return_value = {"response": '{"invoice_number":"00633","evidence":{"invoice_number":"L2"}}', "done": True}
        post.side_effect = [success, requests.ReadTimeout()]
        diagnostics = {}
        result = OllamaInvoiceAnalyzer().analyze_invoice_text(text,
            {"invoice_number": passages[:1], "total_amount": passages[1:]}, diagnostics=diagnostics)
        self.assertEqual(result.invoice_number, "00633")
        self.assertIsNone(result.total_amount)
        self.assertEqual(diagnostics["status"], "completed_partial")
        self.assertEqual(diagnostics["batches"][-1]["status"], "timeout")

    def test_context_rejects_foreign_passage_and_bounds_size(self):
        text = "\n".join(f"Ligne originale {i}" for i in range(30))
        passages = source_passages(text)
        forged = {**passages[0], "text": "FACTURE AUTRE CLIENT"}
        self.assertFalse(selected_context(text, {"invoice_number": [forged]})["allowed_lines"])
        context = selected_context(text, {"invoice_number": passages}, max_chars=130)
        self.assertLessEqual(len(context["text"]), 130)
        self.assertGreater(context["dropped_passages"], 0)

    def test_prompt_requires_second_check_and_abstention(self):
        prompt = OllamaInvoiceAnalyzer()._build_prompt("FACTURE")
        for rule in ("CONTROLE FINAL OBLIGATOIRE", "relis le passage", "retourne null pour ce champ",
                     "Une absence n'est pas zero", "sans source", "uniquement du texte fourni"):
            self.assertIn(rule, prompt)

    def test_grounding_rejects_invented_amount_and_missing_quote(self):
        analyzer = OllamaInvoiceAnalyzer()
        data = analyzer._ground_fields({
            "total_amount": 12.6, "subtotal": 10, "supplier_name": "Invented",
            "evidence": {"total_amount": "TOTAL TTC: 12,600", "subtotal": "TOTAL TTC: 12,600"},
        }, "TOTAL TTC: 12,600")
        self.assertEqual(data["total_amount"], 12.6)
        self.assertIsNone(data["subtotal"])
        self.assertIsNone(data["supplier_name"])

    def test_conversion_preserves_contacts_and_two_products(self):
        result = OllamaInvoiceAnalyzer()._to_invoice_data({
            "supplier_name": "BEN ABDELKADER", "supplier_phone": "22 629 288",
            "supplier_tax_id": "1547376Z/N/C/000", "items": [
                {"description": "Rouleau Antigoutte Cristal", "quantity": 1, "total": 6},
                {"description": "Pinceau rond N6/5 SAP", "quantity": 1, "total": 6},
            ],
        }, "{}")
        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.supplier.phone, "22 629 288")
        self.assertEqual(result.supplier.vat_number, "1547376Z/N/C/000")

    def test_grounding_rejects_hallucinated_products_and_percentage(self):
        result = OllamaInvoiceAnalyzer()._ground_fields({
            "tax_amount": 19, "evidence": {"tax_amount": "TVA 19%"},
            "items": [{"description": "Produit 1", "quantity": 1, "total": 12.6}],
        }, "TVA 19%\nTOTAL TTC: 12,600")
        self.assertIsNone(result["tax_amount"])
        self.assertEqual(result["items"], [])


if __name__ == "__main__":
    unittest.main()
