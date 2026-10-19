import ast
import re
import unittest
from pathlib import Path

from paddle_classifier import layout_decision


class DocumentClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(Path(__file__).with_name("main.py").read_text(encoding="utf-8-sig"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "classify_invoice_text")
        scope = {"re": re}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), scope)
        cls.classify = staticmethod(scope["classify_invoice_text"])

    def test_purchase_request_not_invoice_even_when_it_mentions_invoice(self):
        text = "Demande d'Achat Ordinaire /Immobilisation\nFacture a joindre\nClient Societe\n15/11/2019\nDesignation Quantite Prix unitaire\nMontant Total TTC 30.000\nTVA"
        result = self.classify(text)
        self.assertFalse(result[0])
        self.assertFalse(layout_decision(text, [{"label": "table", "score": .95}], result)[0])

    def test_delivery_note_not_invoice(self):
        text = "BON DE LIVRAISON\nMF 761604\n18/06/2018 Client\nDesignation Quantite PU HT TVA\nTotal TTC 100.000"
        self.assertFalse(self.classify(text)[0])

    def test_isolated_table_requires_review(self):
        text = "Designation Quantite Prix unitaire\nClient\nTotal HT 120.000\nTVA 22.800\nTotal TTC 142.800"
        result = self.classify(text)
        self.assertFalse(result[0])
        self.assertTrue(result[2]["requires_type_review"])

    def test_invoice_with_delivery_reference_stays_invoice(self):
        text = "FACTURE N 00633\n28/11/2019\nMF 1547376Z/N/C/000\nClient BRIDGE\nDesignation Quantite PU TTC\nTOTAL TTC 12.600\nBon de livraison 123"
        self.assertTrue(self.classify(text)[0])


if __name__ == "__main__":
    unittest.main()
