import unittest

from invoice_rules import extract_invoice_table_text, filter_invoice_line_items


class InvoiceLineRulesTests(unittest.TestCase):
    def test_english_rows_without_printed_totals_are_not_invented_or_removed(self):
        text = "Description | Qty | Unit Price\nStrategy food spend. | 1 | 91.01\nSubtotal 91.01"
        table = extract_invoice_table_text(text)
        rows, rejected = filter_invoice_line_items([
            dict(description="Strategy food spend.", quantity=1, unit_price=91.01, total=None)], table)
        self.assertEqual(rejected, [])
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["total"])
        self.assertEqual(rows[0]["unit_price"], 91.01)

    def test_printed_price_basis_survives_both_filter_passes(self):
        text = "Designation Qte PU TTC Montant\nRouleau Cristal 1 6.000 6.000\nTOTAL TTC 6.600"
        rows = [dict(description="Rouleau Cristal", quantity=1, unit_price=6, total=6)]
        for _ in range(2):
            rows, rejected = filter_invoice_line_items(rows, text, 6.6)
            self.assertEqual(rows[0]["price_basis"], "TTC")

    def test_total_ttc_does_not_imply_unit_price_ttc(self):
        rows, _ = filter_invoice_line_items([dict(description="Rouleau Cristal", quantity=1, unit_price=6, total=6)],
            "Designation Qte PU Montant\nRouleau Cristal 1 6.000 6.000\nTOTAL TTC 6.600", 6.6)
        self.assertNotIn("price_basis", rows[0])

    def setUp(self):
        self.text = """FACTURE N: 00633
        Nom du Client : BRIDGE IMMOBILIERE
        Adresse : Rue Jawdet Al-Hayet Choutrana
        Désignation Qte PU TTC Montant
        Rouleau Antigoutte Cristal 1 6.000 6.000
        Pinceau rond N6/5 SAP 1 6.000 6.000
        TOTAL TTC 12.600
        Téléphone : 22 629 288
        """

    def test_only_proven_article_rows_survive(self):
        table = extract_invoice_table_text(self.text)
        accepted, rejected = filter_invoice_line_items(
            [
                {"description": "Rouleau Antigoutte Cristal", "quantity": 1, "unit_price": "6.000", "total": "6.000"},
                {"description": "Pinceau rond N6/5 SAP", "quantity": 1, "unit_price": "6.000", "total": "6.000"},
                {"description": "Adresse Rue Jawdet Al-Hayet Choutrana", "quantity": 1, "unit_price": "1.000", "total": "1.000"},
                {"description": "Téléphone", "quantity": 1, "unit_price": "22629288", "total": "22629288"},
                {"description": "Désignation | Qte | PU TTC", "quantity": 1, "unit_price": "6.000", "total": "6.000"},
            ],
            table,
            "12.600",
        )
        self.assertEqual([item["description"] for item in accepted], ["Rouleau Antigoutte Cristal", "Pinceau rond N6/5 SAP"])
        self.assertEqual({item["reason"] for item in rejected}, {"metadata_or_address", "ocr_table_artifact"})

    def test_line_math_mismatch_is_rejected(self):
        table = extract_invoice_table_text(self.text)
        accepted, rejected = filter_invoice_line_items(
            [{"description": "Rouleau Antigoutte Cristal", "quantity": 2, "unit_price": "6.000", "total": "6.000"}],
            table,
            "12.600",
        )
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["reason"], "line_math_mismatch")

    def test_product_reference_digits_do_not_replace_quantity(self):
        table = extract_invoice_table_text(self.text)
        accepted, rejected = filter_invoice_line_items(
            [{"description": "Pinceau rond N6/5 SAP", "quantity": 1, "unit_price": "6.000", "total": "6.000"}],
            table,
            "12.600",
        )
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]["quantity"], 1.0)


if __name__ == "__main__":
    unittest.main()
