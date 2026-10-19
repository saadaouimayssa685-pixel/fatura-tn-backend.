"""Financial-only regressions: frozen OCR evidence and audited printed values.

Audited fixtures are not fresh extraction results. In particular 00791's printed
label/value associations were visually audited; this test does not repair OCR.
"""
import copy
import json
from decimal import Decimal
from pathlib import Path
import unittest

from financial_validation import RoundingPolicy, financial_validation_issues, validate_invoice_finances
from test_hybrid_extraction import pipeline_functions


QA = Path(__file__).parent / "qa-results/generalization-audit-20260921T121302Z"


def proof(value, text, **metadata):
    return {"value": value, "supported": True, "source": {"text": text}, **metadata}


def simple(ht="100.000", vat="19.000", stamp="1.000", total="120.000", **extra):
    values = {"subtotal": ht, "tax_amount": vat, "stamp_duty": stamp, "total_amount": total,
              "currency": "TND", **extra}
    labels = {"subtotal": "HT", "tax_amount": "TVA", "stamp_duty": "Timbre", "total_amount": "TTC"}
    proofs = {k: proof(values[k], f"{label}: {values[k]}") for k, label in labels.items() if values.get(k) is not None}
    return values, proofs


def source(document, *lines):
    text = (QA / (document + ".ocr.txt")).read_text(encoding="utf-8").splitlines()
    return "\n".join(text[line - 1] for line in lines)


def audited_case(document, amounts, line_map, currency="TND"):
    fields = {**amounts, "currency": currency}
    proofs = {k: proof(amounts[k], source(document, *ids), line_ids=list(ids),
                       provenance="existing_visual_audit") for k, ids in line_map.items()}
    return fields, proofs


def add_tax(fields, proofs, label, amount, rate=None, text=None, **metadata):
    row = {"label": label, "amount": amount, "rate": rate,
           "source": text or f"{label}: {amount}", **metadata}
    fields.setdefault("taxes", []).append(row)
    proofs["taxes"] = {"supported": True, "source": "audited_printed_tax_rows"}
    return row


class FinancialValidationTests(unittest.TestCase):
    def check(self, fields, proofs, expected):
        before = copy.deepcopy((fields, proofs))
        result = validate_invoice_finances(fields, proofs)
        self.assertEqual(result["status"], expected, result)
        self.assertEqual((fields, proofs), before)
        json.dumps(result)  # No leaked Decimal or non-serializable evidence.
        return result

    def test_00040_simple_proven_ht_vat_stamp(self):
        fields, proofs = audited_case("facture_00040", {
            "subtotal": "4650.000", "tax_amount": "883.500", "stamp_duty": "0.600", "total_amount": "5534.100"},
            {"subtotal": [16], "tax_amount": [17], "stamp_duty": [18], "total_amount": [19]})
        result = self.check(fields, proofs, "VALID")
        self.assertEqual(Decimal(result["calculated_amount"]), Decimal("5534.100"))
        self.assertEqual(result["difference"], "0.000")
        self.assertEqual(result["printed_evidence"]["line_ids"], [19])

    def test_00040_actual_frozen_final_output_is_valid(self):
        trace = json.loads((QA / "final/facture_00040.json").read_text(encoding="utf-8"))
        self.check(trace["fields"], trace["proofs"], "VALID")

    def test_00008_actual_corrected_baseline_keeps_net_ht_fodec_and_printed_ttc(self):
        path = QA.parent / "functional-fixes-00008-20260921/result-final.json"
        trace = json.loads(path.read_text(encoding="utf-8"))
        result = self.check(trace["fields"], trace["proofs"], "ROUNDING_DIFFERENCE")
        self.assertEqual(result["difference"], "0.001")
        self.assertEqual(result["calculated_amount"], "6064.633")
        self.assertEqual(Decimal(result["printed_amount"]), Decimal("6064.634"))
        self.assertNotIn("discount", [c["kind"] for c in result["components"]])

    def test_00079_global_totals_do_not_require_a_complete_items_page(self):
        fields, proofs = audited_case("facture_00079", {
            "subtotal": "57618.909", "tax_amount": "10730.222", "stamp_duty": "1.000", "total_amount": "68350.131"},
            {"subtotal": [27, 30], "tax_amount": [34], "stamp_duty": [36], "total_amount": [38]})
        fields["items"] = [{"total": "41091.629", "source": "partial page 2/2", "price_basis": "HT"}]
        proofs["items"] = {"supported": True, "source": "ocr_table_rules", "table_complete": False}
        self.check(fields, proofs, "VALID")

    def test_00791_audited_levy_is_not_vat_or_fodec_and_rounding_is_exact(self):
        fields, proofs = audited_case("facture_00791", {
            "subtotal": "28.961", "tax_amount": "5.503", "stamp_duty": "0.831", "total_amount": "37.020"},
            {"subtotal": [21, 22], "tax_amount": [23, 24], "stamp_duty": [26, 27], "total_amount": [31, 33]})
        add_tax(fields, proofs, "Redevance Telecom", "1.723", "5", source("facture_00791", 24, 25), line_ids=[24, 25])
        result = self.check(fields, proofs, "ROUNDING_DIFFERENCE")
        self.assertEqual(result["calculated_amount"], "37.018")
        self.assertEqual(result["difference"], "0.002")
        self.assertEqual(result["rounding"]["bound"], "0.0025")
        levy = next(c for c in result["components"] if c["kind"] == "levy")
        self.assertEqual(levy["rate"], "5")
        self.assertEqual(levy["label"], "Redevance Telecom")

    def test_00791_bad_frozen_extraction_is_not_repaired_by_financial_validation(self):
        trace = json.loads((QA / "final/facture_00791.json").read_text(encoding="utf-8"))
        self.check(trace["fields"], trace["proofs"], "INSUFFICIENT_EVIDENCE")

    def test_template1_printed_discount_and_vat_reveal_real_document_mismatch(self):
        fields, proofs = audited_case("Template1_Instance175", {
            "subtotal": "700.94", "discount_amount": "26.92", "discount_rate": "3.84",
            "tax_amount": "25.04", "total_amount": "694.09"},
            {"subtotal": [18], "discount_amount": [19], "discount_rate": [19], "tax_amount": [20], "total_amount": [23]}, "USD")
        result = self.check(fields, proofs, "INVALID")
        self.assertEqual(result["calculated_amount"], "699.06")
        self.assertEqual(result["difference"], "-4.97")
        self.assertTrue(result["requires_review"])

    def test_template26_no_printed_financial_totals(self):
        trace = json.loads((QA / "final/Template26_Instance175.json").read_text(encoding="utf-8"))
        result = self.check(trace["fields"], trace["proofs"], "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(result["calculated_amount"])

    def ttc_case(self):
        fields, proofs = audited_case("facture_00158", {"stamp_duty": "0.600", "total_amount": "12.600"},
                                      {"stamp_duty": [18], "total_amount": [21]})
        fields["items"] = [{"total": "6.000", "price_basis": "TTC", "source": source("facture_00158", line)} for line in [13, 14]]
        proofs["items"] = {"supported": True, "source": source("facture_00158", 11, 12, 13, 14),
                           "table_complete": True, "provenance": "existing_visual_audit"}
        return fields, proofs

    def test_00158_complete_audited_ttc_table_never_invents_ht_or_vat(self):
        fields, proofs = self.ttc_case()
        result = self.check(fields, proofs, "VALID")
        self.assertEqual(result["calculated_amount"], "12.600")
        self.assertNotIn("subtotal", fields)
        self.assertNotIn("tax_amount", fields)
        self.assertNotIn("ht_base", [c["kind"] for c in result["components"]])

    def test_ttc_lines_with_unknown_completeness_abstain(self):
        fields, proofs = self.ttc_case()
        proofs["items"].pop("table_complete")
        result = self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")
        self.assertIn("ttc_table_completeness_unproven", result["reasons"])

    def test_ttc_unit_prices_cannot_manufacture_missing_line_amounts(self):
        fields, proofs = self.ttc_case()
        fields["items"][0].update(total=None, quantity=1, unit_price=6)
        self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")

    def test_ttc_tax_inclusion_is_not_guessed(self):
        fields, proofs = self.ttc_case()
        fields["tax_amount"] = 1
        proofs["tax_amount"] = proof(1, "TVA 1.000")
        self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")

    def test_global_discount_is_subtracted_only_from_gross_ht(self):
        fields, proofs = simple(total="115.000", discount_amount="5.000", discount_rate=5)
        proofs["subtotal"] = proof("100.000", "HT brut 100.000")
        proofs["discount_amount"] = proof("5.000", "Remise 5%: 5.000")
        result = self.check(fields, proofs, "VALID")
        self.assertEqual(next(c["contribution"] for c in result["components"] if c["kind"] == "discount"), "-5.000")

    def test_negative_printed_discount_still_means_a_deduction(self):
        fields, proofs = simple(total="115.000", discount_amount="-5.000")
        proofs["subtotal"] = proof("100.000", "SUB_TOTAL 100.000")
        proofs["discount_amount"] = proof("-5.000", "DISCOUNT: -5.000")
        self.check(fields, proofs, "VALID")

    def test_net_ht_discount_is_not_subtracted_twice(self):
        fields, proofs = simple(discount_amount="5.000")
        proofs["subtotal"] = proof("100.000", "HT net 100.000")
        proofs["discount_amount"] = proof("5.000", "Remise 5.000")
        result = self.check(fields, proofs, "VALID")
        self.assertEqual(result["observations"][0]["kind"], "discount_already_in_net_ht")

    def test_ambiguous_ht_basis_does_not_choose_formula_that_happens_to_balance(self):
        fields, proofs = simple(discount_amount="5.000")
        proofs["discount_amount"] = proof("5.000", "Remise 5.000")
        self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")

    def test_discount_rate_only_never_becomes_an_amount(self):
        fields, proofs = simple(discount_rate=5)
        proofs["discount_rate"] = proof(5, "Remise 5%")
        self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")

    def test_fodec_and_vat_rows_never_duplicate_scalar_vat(self):
        fields, proofs = simple(total="121.000")
        add_tax(fields, proofs, "FODEC", "1.000", 1)
        add_tax(fields, proofs, "TVA", "19.000", 19)
        result = self.check(fields, proofs, "VALID")
        self.assertEqual(len([c for c in result["components"] if c["kind"] == "vat"]), 1)

    def test_multiple_distinct_printed_taxes_keep_label_rate_and_evidence(self):
        fields, proofs = simple(total="126.000")
        add_tax(fields, proofs, "FODEC", "1.000", 1)
        add_tax(fields, proofs, "Redevance telecom", "2.000", 5)
        add_tax(fields, proofs, "Contribution environnementale", "3.000")
        result = self.check(fields, proofs, "VALID")
        self.assertEqual({c["kind"] for c in result["components"]}, {"ht_base", "vat", "stamp", "fodec", "levy", "other_tax"})

    def test_additional_tax_aggregate_is_not_added_twice(self):
        fields, proofs = simple(total="123.000", additional_tax_total="3.000")
        proofs["additional_tax_total"] = proof("3.000", "Total taxes additionnelles 3.000")
        add_tax(fields, proofs, "FODEC", "1.000", 1)
        add_tax(fields, proofs, "Redevance", "2.000")
        self.check(fields, proofs, "VALID")

    def test_conflicting_tax_aggregate_is_not_used_to_choose_a_total(self):
        fields, proofs = simple(total="122.000", additional_tax_total="2.000")
        proofs["additional_tax_total"] = proof("2.000", "Total taxes additionnelles 2.000")
        add_tax(fields, proofs, "FODEC", "1.000")
        self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")

    def test_stamp_inside_taxes_is_deduplicated(self):
        fields, proofs = simple()
        add_tax(fields, proofs, "Timbre fiscal", "1.000")
        self.check(fields, proofs, "VALID")

    def test_duplicate_row_with_same_evidence_is_used_once(self):
        fields, proofs = simple(total="121.000")
        row = add_tax(fields, proofs, "FODEC", "1.000", 1)
        fields["taxes"].append(copy.deepcopy(row))
        self.check(fields, proofs, "VALID")

    def test_equal_amount_different_taxes_are_not_deduplicated(self):
        fields, proofs = simple(total="122.000")
        add_tax(fields, proofs, "FODEC", "1.000")
        add_tax(fields, proofs, "Redevance", "1.000")
        self.check(fields, proofs, "VALID")

    def test_equal_levies_at_distinct_evidence_locations_are_not_deduplicated(self):
        fields, proofs = simple(total="122.000")
        add_tax(fields, proofs, "Redevance", "1.000", line_ids=[10])
        add_tax(fields, proofs, "Redevance", "1.000", line_ids=[11])
        self.check(fields, proofs, "VALID")

    def test_rejected_tax_breakdown_does_not_silently_mean_zero_additional_taxes(self):
        fields, proofs = simple()
        fields["taxes"] = []
        proofs["taxes"] = {"supported": False, "requires_review": True, "source": None}
        self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")

    def test_missing_tax_amount_cannot_be_calculated_from_rate(self):
        fields, proofs = simple()
        add_tax(fields, proofs, "FODEC", None, 1)
        self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")

    def test_missing_vat_is_not_zero(self):
        fields, proofs = simple(vat=None, total="101.000")
        self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")

    def test_printed_zero_vat_is_a_proven_component(self):
        fields, proofs = simple(vat="0.000", total="101.000")
        self.check(fields, proofs, "VALID")

    def test_missing_mentioned_stamp_is_not_zero(self):
        fields, proofs = simple(stamp=None, total="119.000")
        proofs["total_amount"] = proof("119.000", "Timbre fiscal illisible\nTTC 119.000")
        self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")

    def test_missing_mentioned_fodec_is_not_ignored(self):
        fields, proofs = simple()
        proofs["total_amount"] = proof("120.000", "FODEC 1%\nTTC 120.000")
        self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")

    def test_unproven_or_rejected_scalar_abstains_even_when_arithmetic_balances(self):
        for key in ("subtotal", "tax_amount", "stamp_duty", "total_amount"):
            for change in ({"supported": False}, {"requires_review": True}, {"source": None}, {"value": 999}):
                with self.subTest(key=key, change=change):
                    fields, proofs = simple()
                    proofs[key].update(change)
                    self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")

    def test_no_amounts_means_no_calculation(self):
        result = self.check({}, {}, "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(result["difference"])
        self.assertIsNone(result["printed_amount"])
        self.assertEqual(result["components"], [])

    def test_missing_total_is_not_filled(self):
        fields, proofs = simple(total=None)
        self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(fields["total_amount"])

    def test_unproven_printed_value_is_kept_in_diagnostic_not_validated(self):
        fields, proofs = simple()
        proofs["total_amount"]["supported"] = False
        result = self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["printed_amount"], "120.000")
        self.assertIsNone(result["difference"])

    def test_missing_value_with_proof_is_not_silently_omitted_or_filled(self):
        fields, proofs = simple(additional_tax_total=None)
        proofs["additional_tax_total"] = proof("1.000", "Autres taxes 1.000")
        self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")

    def test_item_discount_cannot_establish_net_ht_using_a_ttc_row(self):
        fields, proofs = simple(discount_amount="5.000")
        row_text = "Article | 105.000 | 5.000"
        fields["items"] = [{"total": "105.000", "source": row_text, "price_basis": "TTC"}]
        proofs["items"] = {"supported": True, "source": row_text}
        proofs["discount_amount"] = {"value": "5.000", "supported": True,
                                     "source": {"scope": "item", "row_text": row_text, "text": row_text}}
        self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")

    def test_thousands_spaces_are_decimal_exact(self):
        fields, proofs = simple(ht="2 640,753", vat="0", stamp="1", total="2\u202f641,753")
        result = self.check(fields, proofs, "VALID")
        self.assertEqual(Decimal(result["calculated_amount"]), Decimal("2641.753"))

    def test_rounding_is_based_on_currency_precision_and_can_be_disabled(self):
        fields, proofs = simple(total="120.001")
        self.check(fields, proofs, "ROUNDING_DIFFERENCE")
        strict = validate_invoice_finances(fields, proofs, rounding_policy=RoundingPolicy(enabled=False))
        self.assertEqual(strict["status"], "INVALID")

    def test_rounding_precision_can_be_configured_without_changing_values(self):
        fields, proofs = simple(total="120.001", currency="XXX")
        self.check(fields, proofs, "INVALID")
        result = validate_invoice_finances(fields, proofs, rounding_policy=RoundingPolicy(currency_decimals={"XXX": 3}))
        self.assertEqual(result["status"], "ROUNDING_DIFFERENCE")

    def test_old_fixed_005_tolerance_is_not_used(self):
        fields, proofs = simple(total="120.040")
        result = self.check(fields, proofs, "INVALID")
        self.assertEqual(result["difference"], "0.040")

    def test_explicit_signed_rounding_adjustment_is_used_not_invented(self):
        fields, proofs = simple(total="119.999", rounding_adjustment="-0.001")
        proofs["rounding_adjustment"] = proof("-0.001", "Ajustement arrondi: -0.001")
        result = self.check(fields, proofs, "VALID")
        self.assertIn("printed_adjustment", [c["kind"] for c in result["components"]])

    def test_adjustment_without_evidence_cannot_reconcile(self):
        fields, proofs = simple(total="119.999", rounding_adjustment="-0.001")
        self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")

    def test_vat_breakdown_needs_known_completeness_if_no_aggregate(self):
        fields, proofs = simple(vat=None)
        add_tax(fields, proofs, "TVA", "19.000", 19)
        self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")
        proofs["taxes"]["vat_complete"] = True
        self.check(fields, proofs, "VALID")

    def test_aggregate_vat_allows_a_partial_breakdown_without_double_counting(self):
        fields, proofs = simple()
        add_tax(fields, proofs, "TVA", "7.000", 7)
        self.check(fields, proofs, "VALID")

    def test_nonfinite_or_boolean_money_abstains(self):
        for value in ("NaN", "Infinity", float("nan"), True):
            fields, proofs = simple(vat=value)
            self.check(fields, proofs, "INSUFFICIENT_EVIDENCE")

    def test_net_payable_without_settlement_scope_is_not_assumed_equal_to_ttc(self):
        fields, proofs = simple(net_to_pay="120.000")
        proofs["net_to_pay"] = proof("120.000", "Net a payer 120.000")
        result = self.check(fields, proofs, "VALID")
        self.assertEqual(result["related_checks"][0]["status"], "INSUFFICIENT_EVIDENCE")

    def test_proven_net_payable_and_printed_settlement_are_checked_separately(self):
        fields, proofs = simple(net_to_pay="110.000", payment_adjustment="-10.000")
        proofs["net_to_pay"] = proof("110.000", "Net a payer 110.000", settlement_complete=True)
        proofs["payment_adjustment"] = proof("-10.000", "Acompte deduit -10.000")
        result = self.check(fields, proofs, "VALID")
        self.assertEqual(result["related_checks"][0]["status"], "VALID")

    def test_net_payable_mismatch_is_reported_without_replacing_it(self):
        fields, proofs = simple(net_to_pay="119.000")
        proofs["net_to_pay"] = proof("119.000", "Net a payer 119.000", settlement_complete=True)
        result = self.check(fields, proofs, "VALID")
        self.assertEqual(result["related_checks"][0]["status"], "INVALID")
        self.assertTrue(any(i["code"] == "net_to_pay_invalid" and i["severity"] == "blocking"
                            for i in financial_validation_issues(fields, proofs)))

    def test_existing_pipeline_exposes_structured_check_without_mutation(self):
        fields, proofs = simple(total="160.000")
        original = copy.deepcopy((fields, proofs))
        issues = pipeline_functions()["validate_normalized_invoice_fields"](fields, proofs)
        financial = next(i for i in issues if i["code"] == "invoice_total_math_mismatch")
        self.assertEqual(financial["status"], "INVALID")
        self.assertEqual(financial["metadata"]["printed_amount"], "160.000")
        self.assertEqual((fields, proofs), original)


if __name__ == "__main__":
    unittest.main()
