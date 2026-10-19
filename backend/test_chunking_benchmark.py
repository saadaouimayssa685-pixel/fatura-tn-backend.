"""Tests of QA scoring only. No OCR, LLM, database or embedding execution."""
import copy
import unittest

from qa_chunking_benchmark import (
    OUTPUT, QUERY_KEYS, aggregate, canonical_chunk, chunk_stats, contains,
    diagnose, evaluate, inventory, read_json, resolve_units, sha,
)


def chunk(ids, page=1, doc="doc", **extra):
    return {"document_id": doc, "page": page, "line_ids": ids, "text": "test", **extra}


def unit(*alternatives, **extra):
    return {"id": "proof", "alternatives": [
        {"document_id": "doc", "page": 1, "line_ids": ids} for ids in alternatives], **extra}


class BenchmarkScoringTests(unittest.TestCase):
    def test_document_and_page_required_not_just_value(self):
        proof = unit([7])["alternatives"][0]
        self.assertTrue(contains(chunk([7]), proof))
        self.assertFalse(contains(chunk([7], doc="other"), proof))
        self.assertFalse(contains(chunk([7], page=2), proof))
        self.assertFalse(contains(chunk([8]), proof))

    def test_all_lines_in_one_chunk_and_explicit_alternatives(self):
        gold = unit([1, 2], [8, 9])
        self.assertEqual(evaluate([chunk([1]), chunk([2]), chunk([8, 9])], [gold])["ranks"], [3])
        self.assertEqual(evaluate([chunk([1, 9]), chunk([2, 8])], [gold])["ranks"], [None])

    def test_partial_table_coverage_and_body_only_sensitivity(self):
        units = [unit([1, 2], header_line_ids=[1], body_line_ids=[2]),
                 unit([1, 3], header_line_ids=[1], body_line_ids=[3])]
        selected = [chunk([1, 2]), chunk([3])]
        self.assertEqual(evaluate(selected, units)["recall"], {"1": .5, "3": .5, "5": .5})
        self.assertEqual(evaluate(selected, units, body_only=True)["recall"]["3"], 1)

    def test_no_extra_credit_after_rank_five(self):
        selected = [chunk([i]) for i in range(1, 7)]
        self.assertEqual(evaluate(selected, [unit([6])])["ranks"], [None])

    def test_macro_and_micro_denominators_are_different(self):
        scalar = {"units": [unit([1])]}
        table = {"units": [unit([1]), unit([2]), unit([3])]}
        for mode in ("legacy", "layout"):
            scalar[mode] = evaluate([chunk([1])], scalar["units"])
            table[mode] = evaluate([], table["units"])
        result = aggregate([scalar, table])["modes"]["legacy"]
        self.assertEqual(result["macro_recall"]["1"], .5)
        self.assertEqual(result["micro_recall"]["1"], .25)

    def test_failure_diagnostics_distinguish_loss_split_header_ranking(self):
        self.assertEqual(diagnose([chunk([1])], unit([2])), "A")
        self.assertEqual(diagnose([chunk([1, 2])], unit([1, 2])), "B")
        self.assertEqual(diagnose([chunk([1]), chunk([2])], unit([1, 2])), "D")
        table = unit([1, 2, 3], header_line_ids=[1], body_line_ids=[2, 3])
        self.assertEqual(diagnose([chunk([1]), chunk([2, 3])], table), "F")
        self.assertEqual(diagnose([chunk([1, 2]), chunk([3])], table), "E")

    def test_legacy_adapter_is_non_mutating_and_does_not_fabricate_geometry(self):
        original = {"source_id": "P2:L2-4", "global_line_start": 5, "global_line_end": 7,
                    "page_number": 2, "text": "a\nb\nc", "score": .0123}
        before = copy.deepcopy(original)
        converted = canonical_chunk(original, "doc")
        self.assertEqual(original, before)
        self.assertEqual(converted["line_ids"], [5, 6, 7])
        self.assertEqual(converted["page"], 2)
        self.assertEqual(converted["score"], .0123)
        self.assertNotIn("bbox", converted)
        self.assertNotIn("section", converted)

    def test_explicit_noncontiguous_layout_lines_not_replaced_by_range(self):
        original = {"line_ids": [1, 4], "global_line_start": 1, "global_line_end": 4, "page": 1}
        converted = canonical_chunk(original, "doc")
        self.assertFalse(contains(converted, unit([2])["alternatives"][0]))

    def test_chunk_statistics_do_not_call_overlap_a_split(self):
        units = [unit([1, 2], header_line_ids=[1], body_line_ids=[2]),
                 unit([1, 3], header_line_ids=[1], body_line_ids=[3])]
        stats = chunk_stats([chunk([1, 2, 3]), chunk([3])], {"line_items": units}, "legacy")
        self.assertEqual(stats["gold_tables_split"], 0)
        self.assertEqual(stats["table_overlapping_chunk_counts"], [2])
        self.assertIsNone(stats["totals_chunks"])
        self.assertEqual(stats["without_bbox"], 2)
        split = chunk_stats([chunk([1, 2]), chunk([3])], {"line_items": units}, "layout")
        self.assertEqual(split["gold_tables_split"], 1)

    def test_annotations_reject_changed_ocr_invalid_lines_and_header_loss(self):
        text = "Header\nProduct"
        document = {"document_id": "doc", "page": 1, "ocr_text_sha256": sha(text.encode()),
                    "fields": {"line_items": [{"id": "row", "header_line_ids": [1],
                                                "body_line_ids": [2], "alternatives": [[1, 2]]}]}}
        resolved = resolve_units(document, text)
        self.assertEqual(resolved["line_items"][0]["alternatives"][0]["source_text"], ["Header", "Product"])
        with self.assertRaises(ValueError):
            resolve_units(document, text + " changed")
        for invalid in ([1, 3], [2], [1, 1, 2]):
            document["fields"]["line_items"][0]["alternatives"] = [invalid]
            with self.assertRaises(ValueError):
                resolve_units(document, text)

    def test_frozen_gold_inventory_without_ocr_or_models(self):
        annotations = read_json(OUTPUT / "annotations.json")
        records, datasets = inventory(annotations)
        self.assertEqual(len(records), 8)
        self.assertEqual(sum(r["invoice"] for r in records), 7)
        fields, units = 0, 0
        for document in annotations["documents"]:
            resolved = resolve_units(document, datasets[document["document_id"]][0]["ocr_text"])
            fields += len(resolved)
            units += sum(len(values) for values in resolved.values())
            for name, values in resolved.items():
                self.assertIn(name, QUERY_KEYS)
                for value in values:
                    self.assertTrue(all(proof["document_id"] == document["document_id"] for proof in value["alternatives"]))
        self.assertEqual((fields, units), (63, 82))


if __name__ == "__main__":
    unittest.main()
