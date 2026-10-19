import ast
import re
import unittest
from pathlib import Path
from unittest.mock import Mock


class OCRRoutingTests(unittest.TestCase):
    def test_paddle_success_does_not_trigger_second_full_page_llm_ocr(self):
        tesseract, glm = Mock(), Mock()
        tesseract.extract_text.return_value = ("text", .8, {"method_used": "paddle_text", "engines_tried": ["paddle_text"]})
        _, _, meta = self.run_ocr(tesseract, glm)
        glm.is_available.assert_not_called()
        glm.extract.assert_not_called()
        self.assertEqual(meta["engines_tried"], ["paddle_text"])
    def run_ocr(self, tesseract, glm):
        tree = ast.parse(Path(__file__).with_name("main.py").read_text(encoding="utf-8-sig"))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"is_pdf_file", "extract_document_text", "enrich_payment_ocr"}]
        scope = {"Path": Path, "re": re, "Dict": dict, "Any": object, "ocr_extractor": tesseract, "glm_ocr": glm}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "main.py", "exec"), scope)
        return scope["extract_document_text"]("scan.png")

    def test_tesseract_failure_does_not_prevent_glm(self):
        tesseract = Mock()
        tesseract.extract_text.side_effect = RuntimeError("OCR failed")
        glm = Mock()
        glm.extract.return_value = "FACTURE 123"
        text, score, meta = self.run_ocr(tesseract, glm)
        self.assertIn("FACTURE 123", text)
        self.assertEqual(score, 0)
        self.assertEqual(meta["method_used"], "glm_ocr")

    def test_glm_does_not_inflate_score(self):
        tesseract, glm = Mock(), Mock()
        tesseract.extract_text.return_value = ("text", .2, {})
        glm.extract.return_value = "readable text"
        self.assertEqual(self.run_ocr(tesseract, glm)[1], .2)

    def test_unavailable_glm_is_not_reported_as_run(self):
        tesseract, glm = Mock(), Mock()
        tesseract.extract_text.return_value = ("text", .5, {})
        glm.is_available.return_value = False
        self.assertEqual(self.run_ocr(tesseract, glm)[2]["engines_tried"], ["tesseract"])
        glm.extract.assert_not_called()
