import unittest
from extraction_guards import named_party_contacts


class PartyContactTests(unittest.TestCase):
    def test_stamp_does_not_override_named_footer(self):
        text = 'NOVATION\nTel: +21673 55 77 53\nKAST Events : IMM RAYHANE APP 23 RIADH SOUKRA II AOUINA 2080 – TUNISIE\nTel: +216.22.380.330Email: khalil@kastevents.com\nMF: 1144983D/A/M/000'
        result = named_party_contacts('KAST Events', text)
        self.assertEqual(result['fields']['phone'], '+216.22.380.330')
        self.assertEqual(result['fields']['email'], 'khalil@kastevents.com')
        self.assertTrue(result['fields']['address'].startswith('IMM RAYHANE'))
        self.assertNotIn('73 55', result['text'])
        self.assertEqual(named_party_contacts('NOVATION CITY', text), {})

    def test_ambiguous_or_unlabelled_blocks_abstain(self):
        self.assertEqual(named_party_contacts('Supplier', 'Supplier\nAdresse : Rue du test\nTel: 22 222 222'), {})
        self.assertEqual(named_party_contacts('Supplier', 'Supplier: Rue A\nSupplier: Rue B'), {})

    def test_next_party_stops_contact_collection(self):
        result = named_party_contacts('Supplier', 'Supplier: Rue A\nCustomer: Rue B\nTel: 22 222 222')
        self.assertNotIn('phone', result['fields'])
