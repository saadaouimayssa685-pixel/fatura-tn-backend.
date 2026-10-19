from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import unittest

from paddle_structure import assemble_rows


FIXTURES = Path(__file__).parent / "qa-results/paddle-small-20260913"


def fixture(name):
    data = json.loads((FIXTURES / f"{name}.ocr.json").read_text(encoding="utf-8"))
    old = data["ocr_metadata"]["lines"]
    cells = [c for line in old for c in line["cells"]]
    return old, cells, assemble_rows(cells)


class FrozenPaddleStructureTests(unittest.TestCase):
    def test_seven_skewed_product_rows_reunited_without_changing_values(self):
        old, _, rows = fixture("facture_00040")
        self.assertEqual(len(old), 39)
        body = [r for r in rows if r.get("table_role") == "body"]
        self.assertEqual(len(body), 7)
        amounts = ["450.000", "1400.000", "100.000", "650.000", "350.000", "1400.000", "300.000"]
        for index, (row, amount) in enumerate(zip(body, amounts)):
            self.assertEqual(row["text"], old[9 + 2 * index]["text"] + f" | 1 | {amount} | {amount}")
            self.assertEqual([c["column_index"] for c in row["cells"]], [0, 1, 2, 3])
            self.assertFalse(row["requires_review"])

    def test_other_seven_frozen_documents_keep_exact_text_and_order(self):
        paths = list(FIXTURES.glob("*.ocr.json"))
        self.assertEqual(len(paths), 8)
        for path in paths:
            if path.name == "facture_00040.ocr.json":
                continue
            with self.subTest(document=path.name):
                old, _, rows = fixture(path.name.removesuffix(".ocr.json"))
                self.assertEqual([r["text"] for r in rows], [r["text"] for r in old])

    def test_no_recognized_cell_lost_modified_duplicated_or_input_mutated(self):
        for path in FIXTURES.glob("*.ocr.json"):
            with self.subTest(document=path.name):
                _, cells, rows = fixture(path.name.removesuffix(".ocr.json"))
                before = deepcopy(cells)
                self.assertEqual(rows, assemble_rows(cells))
                self.assertEqual(cells, before)
                key = lambda c: (c["text"], tuple(c["box"]), c.get("score"), c.get("language"),
                                 json.dumps(c.get("alternatives"), sort_keys=True))
                self.assertEqual(Counter(map(key, cells)), Counter(key(c) for r in rows for c in r["cells"]))
                self.assertEqual([r["line_number"] for r in rows], list(range(1, len(rows) + 1)))
                self.assertTrue(all(r["page_number"] == 1 for r in rows))
                self.assertTrue(all(r["boxes"] == [c["box"] for c in r["cells"]] for r in rows))

    def test_character_error_is_not_replaced_with_qa_answer(self):
        _, cells, rows = fixture("facture_00158")
        original = next(c for c in cells if "BRIDGE" in c["text"])
        actual = next(c for r in rows for c in r["cells"] if "BRIDGE" in c["text"])
        self.assertIn("IMMOBILERE", actual["text"])
        self.assertEqual(actual["text"], original["text"])
        self.assertEqual(actual["score"], original["score"])
        self.assertNotIn("low_confidence", actual)

    def test_merged_reference_description_and_empty_discount_are_not_guessed(self):
        _, _, rows = fixture("facture_00079")
        body = [r for r in rows if r.get("table_role") == "body"]
        self.assertEqual(len(body), 5)
        for row in body:
            merged = row["cells"][0]
            self.assertIsNone(merged["column_index"])
            self.assertEqual(merged["column_candidates"], [0, 1])
            self.assertIn("spans_columns", merged["review_reasons"])
            self.assertEqual(row["column_cells"][5], [])
            self.assertEqual(row["cells"][-2]["column_index"], 6)
            self.assertEqual(row["cells"][-1]["column_index"], 7)

    def test_multiline_description_retains_header_and_reviewable_link(self):
        _, _, rows = fixture("facture_00008")
        header = next(r for r in rows if r.get("table_role") == "header")
        body = [r for r in rows if r.get("table_role") == "body"]
        self.assertEqual(len(body), 3)
        for row in body:
            self.assertEqual(row["header_cell_ids"], header["header_cell_ids"])
            self.assertEqual(row["header_text"], header["text"])
        for row in body[1:]:
            self.assertEqual(row["possible_continuation_of"], [c["cell_id"] for c in body[0]["cells"]])
            self.assertIn("description_continuation_needs_review", row["review_reasons"])

    def test_bilingual_totals_ambiguity_is_flagged_not_silently_reassigned(self):
        _, _, rows = fixture("facture_00791")
        for text in ("28,961", "5,503", "1,723"):
            cell = next(c for r in rows for c in r["cells"] if c["text"] == text)
            self.assertIn("ambiguous_label_value_alignment", cell["review_reasons"])
            self.assertTrue(cell["requires_review"])

    def test_low_recognition_confidence_does_not_alter_score(self):
        cell = {"text": "???", "box": [.1, .2, .1, .02], "score": .6}
        row = assemble_rows([cell])[0]
        self.assertTrue(row["requires_review"])
        self.assertTrue(row["cells"][0]["low_confidence"])
        self.assertEqual(row["cells"][0]["score"], .6)

    def test_no_header_no_column_guess(self):
        cells = [{"text": str(i), "box": [x, .2, .05, .02]} for i, x in enumerate((.1, .4, .7))]
        rows = assemble_rows(cells)
        self.assertEqual(rows[0]["text"], "0 | 1 | 2")
        self.assertNotIn("table_id", rows[0])
        self.assertTrue(all("column_index" not in c for c in rows[0]["cells"]))

    def test_duplicate_cell_in_column_is_not_silently_merged(self):
        old, cells, _ = fixture("facture_00158")
        duplicate = deepcopy(next(c for c in cells if c["text"] == "6.000"))
        rows = assemble_rows(cells + [duplicate])
        matches = [c for r in rows for c in r["cells"] if c["box"] == duplicate["box"]]
        self.assertEqual(len(matches), 2)
        self.assertTrue(all("overlapping_cells_same_column" in c.get("review_reasons", []) for c in matches))

    def test_no_regression_in_existing_deterministic_article_extractor(self):
        from qa_archive_invoices import pipeline
        scope = pipeline()
        for path in FIXTURES.glob("*.ocr.json"):
            with self.subTest(document=path.name):
                data = json.loads(path.read_text(encoding="utf-8"))
                old, _, rows = fixture(path.name.removesuffix(".ocr.json"))
                before = scope["extract_invoice_line_items_from_text"](data["ocr_text"])
                after = scope["extract_invoice_line_items_from_text"]("\n".join(r["text"] for r in rows))
                clean = lambda items: [{k: v for k, v in item.items() if k != "source"} for item in items]
                self.assertEqual(clean(before), clean(after))


if __name__ == "__main__":
    unittest.main()
