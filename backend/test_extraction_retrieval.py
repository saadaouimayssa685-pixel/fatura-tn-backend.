import os
import unittest
from unittest.mock import patch

from extraction_retrieval import ExtractionRetriever, prompt_evidence, source_passages
from paddle_classifier import layout_decision


class RetrievalTests(unittest.TestCase):
    def test_oversized_table_is_reported_as_partial(self):
        text = "Description Qty Price\n" + "\n".join(f"Long product reference {i} 1 6.000" for i in range(100))
        from extraction_retrieval import selected_context
        evidence = ExtractionRetriever().retrieve(text)
        self.assertGreater(selected_context(text, {"items": evidence["items"]})["dropped_passages"], 0)

    def test_all_table_chunks_are_retrieved_without_repeating_header_keywords(self):
        text = "PRIVATE HEADER\nDescription | Qty | Unit Price\n" + "\n".join(
            f"Produit XYZ{i} | 1 | 6.000" for i in range(20)) + "\nTOTAL TTC 120.600\nPRIVATE FOOTER"
        evidence = ExtractionRetriever().retrieve(text)["items"]
        from extraction_retrieval import selected_context
        context = selected_context(text, {"items": evidence})["text"]
        for i in range(20):
            self.assertIn(f"Produit XYZ{i}", context)
        self.assertNotIn("PRIVATE HEADER", context)
        self.assertNotIn("PRIVATE FOOTER", context)
        self.assertNotIn("TOTAL TTC", context)

    def setUp(self):
        self.env = patch.dict(os.environ, {"EXTRACTION_SEMANTIC_RAG": "false"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_totals_at_end_are_not_truncated(self):
        text = "\n".join(f"Produit reference {i} description longue" for i in range(45)) + "\nTOTAL TTC: 12,600 TND"
        evidence = ExtractionRetriever().retrieve(text)
        self.assertTrue(any("TOTAL TTC: 12,600 TND" in p["text"] for p in evidence["total_amount"]))
        self.assertTrue(all(p["text"] in text for p in evidence["total_amount"]))

    def test_pages_and_lines_are_preserved(self):
        passages = source_passages("--- Page 1 ---\nFournisseur ABC\n--- Page 2 ---\nTOTAL TTC: 24,800")
        self.assertEqual([p["page_number"] for p in passages], [1, 2])
        self.assertEqual(passages[1]["source_id"], "P2:L1-1")
        self.assertEqual(passages[1]["global_line_start"], 4)
        self.assertEqual(passages[0]["global_line_start"], 2)

    def test_arabic_retrieval_and_no_other_invoice_leak(self):
        retriever = ExtractionRetriever()
        self.assertTrue(retriever.retrieve("المبلغ الجملي 12.600 دينار")["total_amount"])
        self.assertEqual(retriever.retrieve("Bonjour monde")["total_amount"], [])

    def test_prompt_sources_are_deduplicated(self):
        p = {"source_id": "P1:L1-2", "page_number": 1, "text": "Facture 00633 Total TTC 12,600"}
        result = prompt_evidence({"invoice_number": [p], "total_amount": [p]})
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["field_sources"]["total_amount"], ["P1:L1-2"])


class PaddleDecisionTests(unittest.TestCase):
    boxes = [{"label": "table", "score": .9}]

    def test_table_alone_is_not_an_invoice(self):
        self.assertFalse(layout_decision("Tableau de statistiques", self.boxes, (False, .1, {}))[0])

    def test_quotation_with_totals_is_not_invoice(self):
        self.assertFalse(layout_decision("Devis total TTC 12.600", self.boxes, (True, .9, {}))[0])
        self.assertFalse(layout_decision("Bon de livraison total HT 12", self.boxes, (True, .9, {}))[0])

    def test_invoice_combines_text_and_real_layout(self):
        result = layout_decision("FACTURE 00633 TOTAL TTC 12,600", self.boxes, (False, .4, {"amount_label_matches": 1}))
        self.assertTrue(result[0])
        self.assertEqual(result[2]["table_count"], 1)
        self.assertEqual(result[2]["analysis_method"], "paddleocr_layout_analysis")

    def test_arabic_invoice(self):
        self.assertTrue(layout_decision("فاتورة المبلغ الجملي 12", self.boxes, (False, .4, {"amount_label_matches": 1}))[0])


if __name__ == "__main__":
    unittest.main()
