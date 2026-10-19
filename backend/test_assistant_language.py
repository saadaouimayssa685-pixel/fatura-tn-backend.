import json
import unittest
from unittest.mock import Mock, patch

from assistant_language import answer_language, grounded_numbers, retrieval_question
from ollama_analyzer import OllamaInvoiceAnalyzer
from rag.service import RAGService


class AssistantLanguageTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = OllamaInvoiceAnalyzer()

    def test_question_language_and_explicit_override(self):
        self.assertEqual(answer_language("Quel est le total TTC ?"), "fr")
        self.assertEqual(answer_language("ما هو المبلغ شامل الضريبة؟"), "ar")
        self.assertEqual(answer_language("ما هو TTC ؟ Réponds en français"), "fr")
        self.assertEqual(answer_language("Quel montant ? بالعربية"), "ar")

    def test_arabic_search_retrieves_french_invoice_concepts(self):
        query = retrieval_question("ما هو المبلغ شامل الضريبة؟")
        self.assertIn("TTC", query)
        service = RAGService.__new__(RAGService)
        self.assertGreater(service._lexical_score(query, "Montant TTC : 12.600"), 0)
        self.assertGreater(service._lexical_score(retrieval_question("ما عنوان المورد؟"), "Adresse fournisseur Tunis"), 0)

    def test_direct_answer_uses_actual_source_index_and_language(self):
        sources = [{"text": "Fournisseur QBA"}, {"text": "Montant TTC : 12.600"}]
        with patch("ollama_analyzer.requests.post") as post:
            for question, language in [("Quel est le montant TTC ?", "fr"), ("ما هو المبلغ شامل الضريبة؟", "ar")]:
                result = self.analyzer.answer_sources(question, sources)
                self.assertEqual(result["citations"], [2])
                self.assertIn("12.600", result["answer"])
                self.assertEqual(answer_language(result["answer"]), language)
            post.assert_not_called()

    def test_conflicting_invoices_do_not_pick_first_total(self):
        sources = [{"text": "Total TTC : 12.600", "document_id": 1}, {"text": "Total TTC : 15.600", "document_id": 2}]
        self.assertIsNone(self.analyzer._answer_labeled_amount("Quel TTC ?", sources))
        self.assertIsNone(self.analyzer._answer_labeled_amount("Quel taux TVA ?", sources))
        self.assertIsNone(self.analyzer._answer_labeled_amount("Différence HT et TTC ?", sources))

    def test_empty_sources_abstain_in_question_language(self):
        result = self.analyzer.answer_sources("ما عنوان العميل؟", [])
        self.assertEqual(result["citations"], [])
        self.assertEqual(answer_language(result["answer"]), "ar")

    @patch("ollama_analyzer.requests.post")
    def test_local_prompt_and_hallucinated_number_guard(self, post):
        post.return_value = Mock()
        post.return_value.json.return_value = {"response": json.dumps({"answer": "الهاتف 99999999 [1]", "citations": [1]})}
        result = self.analyzer.answer_sources("ما رقم الهاتف؟", [{"text": "Telephone 22 629 288"}])
        self.assertEqual(result["citations"], [])
        self.assertIn("reponse est le arabe", post.call_args.kwargs["json"]["prompt"])

    def test_arabic_digits_and_citations_are_not_amounts(self):
        self.assertTrue(grounded_numbers("المبلغ ١٢٫٦٠٠ [1]", [{"text": "TTC 12.600"}]))
        self.assertFalse(grounded_numbers("المبلغ 12600 [1]", [{"text": "TTC 12.600"}]))
        self.assertFalse(grounded_numbers("Le montant est 99 [1]", [{"text": "TTC 199"}]))


if __name__ == "__main__":
    unittest.main()
