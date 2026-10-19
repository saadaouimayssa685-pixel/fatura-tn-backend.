import unittest

from paddle_text import assemble_rows, prefer_arabic, needs_arabic


class PaddleGeometryTests(unittest.TestCase):
    def test_cells_stay_on_their_product_row(self):
        def cell(text, x, y):
            return {"text": text, "box": [x, y, .1, .02], "score": .9}
        rows = assemble_rows([cell("6.000", .8, .401), cell("Pinceau", .1, .45),
                              cell("Rouleau", .1, .40), cell("1", .6, .4)])
        self.assertEqual([r["text"] for r in rows], ["Rouleau | 1 | 6.000", "Pinceau"])
        self.assertEqual(len(rows[0]["boxes"]), 3)
        self.assertEqual(rows[1]["line_number"], 2)

    def test_arabic_does_not_replace_confident_french_with_weaker_reading(self):
        self.assertFalse(prefer_arabic("Fournisseur", .98, "\u0627\u0644\u0645\u0648\u0631\u062f", .65))
        self.assertTrue(prefer_arabic("???", .3, "\u0627\u0644\u0645\u0648\u0631\u062f", .9))
        self.assertFalse(prefer_arabic("12.600", .99, "12.600", .99))

    def test_second_recognizer_is_targeted_not_applied_to_every_number(self):
        self.assertFalse(needs_arabic("12.600", .4))
        self.assertFalse(needs_arabic("Rouleau", .99))
        self.assertTrue(needs_arabic("???", .1))
        self.assertTrue(needs_arabic("\u4e2d\u6587", .95))
