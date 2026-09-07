import ast
import re
import unittest
from pathlib import Path
from typing import List
from unittest.mock import MagicMock
from postgres_invoice_store import PostgresInvoiceStore


class RegressionTests(unittest.TestCase):
    def test_field_helpers_do_not_confuse_numbers(self):
        tree = ast.parse(Path("main.py").read_text(encoding="utf-8-sig"))
        names = {"extract_invoice_number", "clean_phone_number", "extract_phone_near_labels"}
        module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
        scope = {"re": re, "List": List}
        exec(compile(module, "main.py", "exec"), scope)
        text = "FACTURE DE TEST\nFOURNISSEUR: TEST\nMF: 1234567 A/M/000\nCLIENT: TEST\nFACTURE N: QA-2026-1001"
        self.assertEqual(scope["extract_invoice_number"](text), "QA-2026-1001")
        self.assertEqual(scope["extract_phone_near_labels"](text, ["client"]), "")
        self.assertEqual(scope["extract_phone_near_labels"]("Tel: 22 629 288", ["fournisseur"]), "22 629 288")

    def test_search_is_parameterized_and_bounded(self):
        store = PostgresInvoiceStore.__new__(PostgresInvoiceStore)
        connection, cursor = MagicMock(), MagicMock()
        connection.__enter__.return_value = connection
        connection.cursor.return_value.__enter__.return_value = cursor
        store._connect = lambda: connection
        cursor.fetchone.return_value = {"count": 0}
        cursor.fetchall.return_value = []
        result = store.search_invoices(supplier="' OR 1=1 --", page=-1, page_size=999)
        self.assertEqual(result["page"], 1)
        self.assertEqual(result["page_size"], 100)
        sql, params = cursor.execute.call_args.args
        self.assertNotIn("' OR 1=1", sql)
        self.assertEqual(params[0], "%' OR 1=1 --%")


if __name__ == "__main__":
    unittest.main()
