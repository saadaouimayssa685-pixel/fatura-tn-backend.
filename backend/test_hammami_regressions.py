import unittest
from extraction_guards import corroborated_supplier_header, complete_tax_id
from invoice_rules import extract_unit_table_rows, filter_invoice_line_items
from field_locations import locate_paddle_fields


class HammamiRegressionTests(unittest.TestCase):
    def test_unit_columns_and_footer(self):
        text = 'Code Article | Désignation | Unité | Qte | P.U.H.T. | R/se | T.V.A. | Montant H.T.\n902020106001 FER DE 06 EN ROULEAUX DEPOT | TONNE | 3 | 2 640,753 | 19 | 7 922,259\nTOTAL HT | 57 618.909\nBASE | 1 811.420 | 55 807.489'
        rows, rejected = filter_invoice_line_items(extract_unit_table_rows(text), text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['quantity'], 3)
        self.assertEqual(rows[0]['price_basis'], 'HT')
        self.assertFalse(rejected)
        rows, rejected = filter_invoice_line_items([{'description':'BASE', 'quantity':1, 'unit_price':811.420, 'total':807.489}], text)
        self.assertFalse(rows)

    def test_repeated_header_and_tax_label(self):
        text = 'COMPTOIR HAMMAMI | Arabic\nSociété S.C.P.C- Comptoir Hammami SARL\nS.S: Route de Gabes | Arabic\nR.C:B 183611999-MF:385191J/A/M/000'
        result = corroborated_supplier_header(text)['fields']
        self.assertEqual(result['vendor_name'], 'COMPTOIR HAMMAMI')
        self.assertEqual(result['vendor_tax_id'], '385191J/A/M/000')
        self.assertEqual(result['vendor_address'], 'Route de Gabes')
        self.assertFalse(complete_tax_id('R.C:B 183611999-MF:385191J/A/M/000', text))
        self.assertFalse(corroborated_supplier_header('Unrelated heading\nSociété SARL'))

    def test_cell_contains_value_but_repeated_cells_abstain(self):
        line = {'page_number':1, 'line_number':4, 'text':'R.C:B 183611999-MF:385191J/A/M/000',
                'cells':[{'text':'R.C:B 183611999-MF:385191J/A/M/000','box':[.1,.1,.5,.05]}]}
        fields = {'vendor_tax_id':'385191J/A/M/000'}
        self.assertIn('vendor_tax_id', locate_paddle_fields([line], fields))
        self.assertNotIn('vendor_tax_id', locate_paddle_fields([line, line], fields))
