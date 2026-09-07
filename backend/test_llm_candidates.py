import unittest

from ollama_analyzer import OllamaInvoiceAnalyzer


class LocalLlmCandidatePromptTests(unittest.TestCase):
    def test_prompt_limits_the_model_to_rule_candidates(self):
        analyzer = OllamaInvoiceAnalyzer()
        prompt = analyzer._build_prompt(
            "FACTURE N: 00633\nTOTAL TTC: 12.600",
            rule_candidates={
                "field_values": {"invoice_number": "00633", "total_amount": 12.6},
                "article_rows": [{"description": "Rouleau", "quantity": 1, "unit_price": 6, "total": 6}],
            },
        )
        self.assertIn("liste fermee", prompt)
        self.assertIn("00633", prompt)
        self.assertIn("ne peux jamais creer une nouvelle valeur", prompt)


if __name__ == "__main__":
    unittest.main()
