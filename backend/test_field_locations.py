import ast
import unittest
from pathlib import Path
from field_locations import match_field_lines
from field_locations import value_matches


class FieldLocationTests(unittest.TestCase):
    def test_paddle_cell_coordinates_are_used_without_re_ocr(self):
        from field_locations import locate_paddle_fields
        lines = [{"page_number": 1, "line_number": 8, "text": "TOTAL TTC | 12.600",
                  "cells": [{"text": "TOTAL TTC", "box": [.1, .8, .3, .03]},
                            {"text": "12.600", "box": [.7, .8, .1, .03]}]}]
        result = locate_paddle_fields(lines, {"total_amount": 12.6})
        self.assertEqual(result["total_amount"]["boxes"], [[.7, .8, .1, .03]])
    def line(self, text, page=1):
        return {"text": text, "box": [0, .2, .9, .03], "page_number": page, "line_number": 3,
                "words": [{"text": word, "box": [index * .08, .2, .07, .03]}
                          for index, word in enumerate(text.split())]}

    def test_word_highlight_does_not_include_customer_on_same_row(self):
        result = match_field_lines([self.line("BEN ABDELKADER Client BRIDGE")],
                                   {"vendor_name": "BEN ABDELKADER"}, lambda *args: True)
        self.assertEqual(result["vendor_name"]["matched_text"], "BEN ABDELKADER")
        self.assertEqual(len(result["vendor_name"]["boxes"]), 2)

    def test_number_is_not_substring_or_thousands(self):
        self.assertFalse(value_matches("invoice_number", "00633", "100633"))
        self.assertTrue(value_matches("invoice_number", "00633", "N°00633"))
        self.assertTrue(value_matches("total_amount", 12.6, "12,600"))
        self.assertFalse(value_matches("total_amount", 12.6, "12600"))
        self.assertFalse(value_matches("total_amount", 19, "19%"))

    def test_repeated_amounts_on_same_row_are_ambiguous(self):
        result = match_field_lines([self.line("PU 6.000 Total 6.000")], {"total_amount": 6}, lambda *args: True)
        self.assertEqual(result, {})

    def test_arabic_words_and_page_context(self):
        line = self.line("المورد شركة النور", page=2)
        result = match_field_lines([self.line("Other page"), line], {"vendor_name": "شركة النور"}, lambda *args: True)
        self.assertEqual(result["vendor_name"]["page_number"], 2)
        self.assertEqual(result["vendor_name"]["context"], [{"line_number": 3, "text": line["text"]}])

    def test_unique_match_preserves_real_coordinates(self):
        line = {"text": "Invoice ABC123", "box": [.1, .2, .3, .04], "page_number": 1}
        result = match_field_lines([line], {"invoice_number": "ABC123"}, lambda k, v, t: v in t)
        self.assertEqual(result["invoice_number"]["box"], line["box"])

    def test_repeated_or_missing_values_have_no_guessed_box(self):
        lines = [{"text": "6.000", "box": [.1, .2, .3, .04]}] * 2
        result = match_field_lines(lines, {"total": "6.000", "name": "absent"}, lambda k, v, t: v in t)
        self.assertEqual(result, {})

    def test_auto_and_explicit_local_never_leave_the_host(self):
        import os
        from unittest.mock import patch
        tree = ast.parse(Path("main.py").read_text(encoding="utf-8-sig"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "choose_agent_model")
        scope = {"os": os, "AGENT_POLICY": {"minimum_text_length": 80, "low_ocr_confidence": .45, "uncertain_classification": .70}}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), scope)
        route = scope["choose_agent_model"]
        with patch.dict(os.environ, {"INVOICE_LLM_PROVIDER": "local"}):
            self.assertEqual(route("auto", "x" * 7000, .2, .6)[0], "local")
            self.assertEqual(route("auto", "", 0, 0)[0], "local")
        with patch.dict(os.environ, {"INVOICE_LLM_PROVIDER": "groq"}):
            self.assertEqual(route("auto", "x", 1, 1)[0], "groq")
            self.assertEqual(route("local", "x", 1, 1)[0], "local")
        self.assertEqual(route("local", "x" * 7000, .2, .6)[0], "local")
        self.assertEqual(route("fallback", "x", 0, 0)[0], "fallback")


if __name__ == "__main__":
    unittest.main()
