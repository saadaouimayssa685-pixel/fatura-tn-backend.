import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from extraction_chunks import RetrievedEvidence, structured_passages
from extraction_retrieval import ExtractionRetriever, selected_context, source_passages
from groq_analyzer import GroqInvoiceAnalyzer
from field_locations import locate_paddle_fields


ROOT = Path(__file__).resolve().parent


def fixture(lines, page=1, label="table"):
    text = (f"--- Page {page} ---\n" if page != 1 else "") + "\n".join(lines)
    metadata = {"coordinate_space": "exif_oriented_image", "image_width": 1000,
                "image_height": 1000, "lines": [{"page_number": page, "line_number": i + 1,
                    "text": line, "cells": [{"text": line, "box": [.1, .1 + i * .015, .6, .01], "score": .9}]}
                    for i, line in enumerate(lines)]}
    layout = {**{k: v for k, v in metadata.items() if k != "lines"},
              "layout_boxes": [{"page_number": page, "label": label, "coordinate": [90, 90, 800, 980]}]}
    return text, metadata, layout


class StructuredChunkTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"EXTRACTION_SEMANTIC_RAG": "false", "RAG_CHUNKING_MODE": "legacy"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_totals_remain_together_with_source_geometry(self):
        text, metadata, layout = fixture(["Total HT : 1250,000", "TVA : 237,500", "Timbre : 1,000", "TOTAL TTC : 1488,500"], page=2)
        chunks = structured_passages(text, metadata, layout, "doc-1")
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk["section"], "totals")
        self.assertEqual(chunk["layout_label"], "table")
        self.assertEqual(chunk["page"], 2)
        self.assertEqual(chunk["document_id"], "doc-1")
        self.assertAlmostEqual(chunk["ocr_confidence"], .9)
        self.assertEqual(chunk["line_ids"], [2, 3, 4, 5])
        self.assertIsNotNone(chunk["bbox"])

    def test_large_table_repeats_header_and_preserves_every_row(self):
        rows = ["Designation | Qte | Prix | Montant"] + [f"Article reference ABC{i:02} | 1 | 6.000 | 6.000" for i in range(25)]
        text, metadata, layout = fixture(rows)
        chunks = structured_passages(text, metadata, layout, max_chars=180)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(len({c["parent_block_id"] for c in chunks}), 1)
        self.assertEqual({i for c in chunks for i in c["line_ids"]}, set(range(1, 27)))
        for chunk in chunks:
            self.assertTrue(chunk["text"].startswith(rows[0]))
            self.assertEqual(chunk["header_line_ids"], [1])
        evidence = RetrievedEvidence({"items": chunks}, chunks)
        context = selected_context(text, evidence)
        self.assertEqual(context["text"].count(rows[0]), 1)
        self.assertEqual(context["dropped_passages"], 0)

    def test_payment_inside_small_totals_layout_does_not_split_amounts(self):
        text, metadata, layout = fixture(["Montant HT : 10.000", "Paye en especes",
            "Arretee a la somme de douze dinars", "TVA : 1.400", "Timbre : 0.600", "Montant TTC : 12.000"])
        chunks = structured_passages(text, metadata, layout)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["section"], "totals")
        self.assertEqual(chunks[0]["line_ids"], list(range(1, 7)))

    def test_tax_identifier_is_not_a_totals_section(self):
        text, metadata, layout = fixture(["Code T.V.A : 816153BAM000"])
        self.assertNotEqual(structured_passages(text, metadata, layout)[0]["section"], "totals")

    def test_no_geometry_or_confidence_is_invented(self):
        text, metadata, layout = fixture(["TOTAL TTC : 12.600"])
        metadata["lines"][0]["cells"] = []
        chunk = structured_passages(text, metadata, layout)[0]
        self.assertIsNone(chunk["bbox"])
        self.assertIsNone(chunk["ocr_confidence"])
        self.assertIsNone(chunk["layout_label"])

    def test_mismatched_layout_frame_is_not_joined(self):
        text, metadata, layout = fixture(["TOTAL TTC : 12.600"])
        layout["image_width"] = 2000
        chunk = structured_passages(text, metadata, layout)[0]
        self.assertFalse(chunk["layout_joined"])
        self.assertIsNone(chunk["layout_label"])
        self.assertIsNotNone(chunk["bbox"])

    def test_metadata_survives_retrieval_and_context(self):
        text, metadata, layout = fixture(["TOTAL TTC : 12.600"], page=2)
        evidence = ExtractionRetriever().retrieve(text, ocr_metadata=metadata, layout_metadata=layout,
                                                 document_id="test-doc", chunking_mode="layout")
        passage = evidence["total_amount"][0]
        self.assertIn("lexical_score", passage)
        self.assertIn("semantic_score", passage)
        self.assertIn("score", passage)
        self.assertEqual(passage["page"], 2)
        self.assertIsNotNone(passage["bbox"])
        self.assertEqual(selected_context(text, evidence)["source_metadata"][passage["chunk_id"]]["bbox"], passage["bbox"])

    def test_legacy_is_identical_even_with_metadata_and_fallback(self):
        text, metadata, layout = fixture(["FACTURE 00633", "TOTAL TTC : 12.600"])
        retriever = ExtractionRetriever()
        baseline = retriever.retrieve(text)
        self.assertEqual(baseline, retriever.retrieve(text, ocr_metadata=metadata, layout_metadata=layout, chunking_mode="legacy"))
        self.assertEqual(baseline, retriever.retrieve(text, chunking_mode="layout"))
        self.assertNotIn("chunk_id", source_passages(text)[0])

    def test_invalid_text_and_foreign_catalog_are_rejected(self):
        text, metadata, layout = fixture(["TOTAL TTC : 12.600"])
        evidence = ExtractionRetriever().retrieve(text, ocr_metadata=metadata, layout_metadata=layout, chunking_mode="layout")
        self.assertFalse(selected_context("OTHER DOCUMENT", evidence)["allowed_lines"])
        value = evidence["total_amount"][0]
        forged = {"total_amount": [{**value, "text": "TOTAL TTC : 900.000"}]}
        self.assertFalse(selected_context(text, forged, source_catalog=evidence.passages)["allowed_lines"])
        self.assertFalse(selected_context(text, dict(evidence))["allowed_lines"])

    def test_header_from_adjacent_layout_block_stays_with_table(self):
        text, metadata, layout = fixture(["Designation Qte Prix", "Produit A 2 3.000 6.000"])
        layout["layout_boxes"] = [
            {"page_number": 1, "label": "paragraph_title", "coordinate": [90, 90, 800, 112]},
            {"page_number": 1, "label": "table", "coordinate": [90, 113, 800, 980]}]
        chunks = structured_passages(text, metadata, layout)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["line_ids"], [1, 2])
        self.assertIn("table", chunks[0]["layout_labels"])

    def test_location_does_not_use_gap_between_repeated_header_and_rows(self):
        lines = [{"page_number": 1, "line_number": 2, "text": "TARGET VALUE",
                  "cells": [{"text": "TARGET VALUE", "box": [.1, .2, .2, .1]}]}]
        proof = {"vendor_name": {"source": {"page_number": 1, "line_start": 1, "line_end": 4,
                  "page_line_ids": [1, 4]}}}
        self.assertEqual(locate_paddle_fields(lines, {"vendor_name": "TARGET VALUE"}, proof), {})

    def test_frozen_qa_lines_survive_without_reexecution_of_ocr(self):
        for name in ("facture_00158", "facture_00079"):
            data = json.loads((ROOT / f"qa-results/paddle-small-20260913/{name}.ocr.json").read_text(encoding="utf-8"))
            chunks = structured_passages(data["ocr_text"], data["ocr_metadata"], document_id=data["sha256"])
            self.assertEqual({i for c in chunks for i in c["line_ids"]}, set(range(1, len(data["ocr_text"].splitlines()) + 1)))
            self.assertTrue(all(c["bbox"] is not None for c in chunks))
            self.assertTrue(any(c["section"] == "line_items" for c in chunks))

    @patch("groq_analyzer.requests.post")
    def test_real_sent_chunk_is_linked_and_unseen_citation_rejected(self, post):
        text, metadata, layout = fixture(["TOTAL TTC : 12.600", "Invoice ABC"])
        evidence = ExtractionRetriever().retrieve(text, ocr_metadata=metadata, layout_metadata=layout, chunking_mode="layout")
        chunk = next(c for c in evidence.passages if c["section"] == "totals")
        evidence = RetrievedEvidence({"total_amount": [chunk]}, evidence.passages)
        post.return_value.status_code = 200
        post.return_value.json.return_value = {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"total_amount": 12.6, "subtotal": 12.6, "evidence": {"total_amount": "L1", "subtotal": "L2"}})}}]}
        with patch.dict(os.environ, {"GROQ_API_KEY": "unit-test-key"}):
            diagnostics = {}
            result = GroqInvoiceAnalyzer().analyze_invoice_text(text, evidence, diagnostics=diagnostics)
        self.assertEqual(result.total_amount, 12.6)
        self.assertIsNone(result.subtotal)
        batch = next(b for b in diagnostics["batches"] if b["status"] == "completed")
        self.assertEqual(batch["citation_chunk_ids"]["total_amount"], [chunk["chunk_id"]])
        self.assertNotIn("Invoice ABC", post.call_args.kwargs["json"]["messages"][1]["content"])
