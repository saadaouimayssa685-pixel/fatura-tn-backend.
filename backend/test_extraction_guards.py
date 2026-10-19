import unittest
from extraction_guards import complete_tax_id, party_conflicts, date_in_source, monetary_in_source


class PartyGuardTests(unittest.TestCase):
    def test_money_must_match_its_label_even_when_other_numbers_exist(self):
        text = "Subtotal | 10.000\nVAT | 19% | 1.900\nStamp | 0.600\nTotal | 12.500"
        self.assertTrue(monetary_in_source("total_amount", 12.5, text))
        self.assertFalse(monetary_in_source("subtotal", 12.5, text))
        self.assertFalse(monetary_in_source("tax_amount", 19, text))
        self.assertTrue(monetary_in_source("tax_amount", 0, "TVA: 0.000"))
        self.assertFalse(monetary_in_source("tax_amount", None, text))

    def test_dates_with_named_months_and_iso_are_supported_not_guessed(self):
        self.assertTrue(date_in_source("2011-06-19", "Date: 19-Jun-2011"))
        self.assertTrue(date_in_source("28-11-2019", "Date 28/11/2019"))
        self.assertFalse(date_in_source("2024-02-31", "Date 31/02/2024"))
        self.assertFalse(date_in_source("2024-11-28", "Facture 2019-11-28"))
    def test_partial_tax_identifier_is_not_a_valid_copy(self):
        self.assertFalse(complete_tax_id("1362434", "Code TVA: 1362434 /S/P/M/000"))
        self.assertTrue(complete_tax_id("1362434/S/P/M/000", "Code TVA: 1362434 /S/P/M/000"))
        self.assertTrue(complete_tax_id("1547376Z/N/C/000", "MF: 1547376 Z/N/C/000"))

    def test_customer_is_never_copied_to_supplier(self):
        self.assertEqual(party_conflicts({"vendor_name": "BRIDGE", "customer_name": "BRIDGE"},
                                        "Nom du Client: BRIDGE"), ["vendor_name"])
        self.assertEqual(party_conflicts({"vendor_name": "SOC ABC"}, "Raison sociale: SOC ABC"), [])

    def test_customer_tax_block_does_not_become_supplier_tax(self):
        text = "RAISONSOCIALE :CLIENT ABC\nCODE CLIENT\nCode TVA :1362434/S/P/M/000\nDesignation Qty Price\nCode TVA fournisseur 816153BAM000"
        self.assertIn("vendor_tax_id", party_conflicts({"vendor_tax_id": "1362434/S/P/M/000"}, text))
        self.assertNotIn("vendor_tax_id", party_conflicts({"vendor_tax_id": "816153BAM000"}, text))
