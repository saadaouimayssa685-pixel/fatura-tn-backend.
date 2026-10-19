import unittest
from invoice_rules import extract_alternating_table_rows, filter_invoice_line_items


class AlternatingRowsTests(unittest.TestCase):
    def test_complete_pairs_preserve_description_numbers(self):
        text = 'Désignation | Qte | Prix Unit | Total HT\n1 | 450.000 | 450.000\nRégie Son + Haut-parleur + Micro\n1 | 1400.000 | 1400.000\n35 Chaises plexi & Habillage Voile\nTotal HT | 1850.000'
        rows = extract_alternating_table_rows(text)
        self.assertEqual(len(filter_invoice_line_items(rows, text)[0]), 2)
        self.assertEqual(rows[1]['quantity'], 1)
        self.assertTrue(rows[1]['description'].startswith('35 Chaises'))
        self.assertEqual(rows[0]['price_basis'], 'HT')

    def test_incomplete_or_inconsistent_pairs_abstain(self):
        for body in ('1 | 450.000 | 400.000\nRoller\n1 | 6.000 | 6.000\nBrush',
                     '1 | 6.000 | 6.000\nBrush\nUnpaired description'):
            self.assertEqual(extract_alternating_table_rows('Désignation | Qte | Prix Unit | Total HT\n' + body), [])
