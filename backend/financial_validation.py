"""Read-only arithmetic checks on grounded, printed invoice components.

This module does not extract, complete or repair invoice data. A supported proof
is the upstream grounding contract; missing/rejected proofs prevent validation.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import re
import unicodedata


ZERO = Decimal("0")


@dataclass(frozen=True)
class RoundingPolicy:
    # Maximum accumulated nearest-rounding error: half a unit per printed term,
    # including the printed result. Unknown currencies have no automatic margin.
    currency_decimals: dict = field(default_factory=lambda: {"TND": 3, "USD": 2, "EUR": 2})
    enabled: bool = True

    def bound(self, currency, terms):
        places = self.currency_decimals.get(str(currency or "").upper())
        if not self.enabled or places is None:
            return ZERO
        if not isinstance(places, int) or isinstance(places, bool) or not 0 <= places <= 8:
            raise ValueError("Currency precision must be an integer between 0 and 8")
        return Decimal(1).scaleb(-places) * Decimal("0.5") * (terms + 1)


def _decimal(value):
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        value = Decimal(str(value).replace("\u00a0", "").replace("\u202f", "").replace(" ", "").replace(",", "."))
        return value if value.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def _fold(value):
    return "".join(c for c in unicodedata.normalize("NFKD", str(value or "")) if not unicodedata.combining(c)).lower()


def _source_text(proof):
    source = proof.get("source") or proof.get("evidence")
    if isinstance(source, dict):
        return str(source.get("text") or source.get("source") or source.get("preview") or "")
    return str(source or "")


def _supported(proof, value=None):
    if not isinstance(proof, dict) or not proof.get("supported") or proof.get("requires_review"):
        return False
    if not (proof.get("source") or proof.get("evidence")):
        return False
    if value is not None and "value" in proof and _decimal(proof["value"]) != value:
        return False
    return True


def _tax_kind(label):
    label = _fold(label)
    if re.search(r"\b(t[. ]*v[. ]*a|vat)\b", label):
        return "vat"
    if re.search(r"\bfodec\b", label):
        return "fodec"
    if re.search(r"\b(timbre|stamp)\b", label):
        return "stamp"
    if re.search(r"\b(redevance|levy)\b", label):
        return "levy"
    return "other_tax"


def validate_invoice_finances(fields, proofs=None, *, rounding_policy=None):
    """Return JSON-safe validation, without mutating fields/proofs.

    Optional metadata on existing proofs may establish subtotal_basis (gross/net),
    table_complete or tax_exempt. Without such evidence, ambiguous checks abstain.
    Monetary rates alone are never converted to amounts. Optional absent charges
    are omitted, not materialized as zero; an indicated but unproved charge blocks
    the calculation. No item sum is treated as a document total on a partial table.
    """
    proofs = proofs or {}
    policy = rounding_policy or RoundingPolicy()
    reasons, components, checks = [], [], []
    observations = []

    def read(*names):
        present = [name for name in names if fields.get(name) is not None and fields.get(name) != ""]
        if not present:
            if any(proofs.get(name, {}).get("requires_review") for name in names):
                reasons.append("rejected_component:" + names[0])
            if any(proofs.get(name, {}).get("value") not in (None, "") for name in names):
                reasons.append("missing_extracted_component:" + names[0])
            return None, {}, names[0]
        name = present[0]
        value, proof = _decimal(fields[name]), proofs.get(name, {})
        if value is None or not _supported(proof, value):
            reasons.append("unproven_component:" + name)
            return None, proof, name
        if any(_decimal(fields[alias]) != value for alias in present[1:]):
            reasons.append("conflicting_aliases:" + names[0])
        return value, proof, name

    def component(name, amount, proof, kind, label=None, rate=None, sign=1):
        row = {"field": name, "kind": kind, "label": label or name, "rate": rate,
               "amount": str(amount), "contribution": str(amount * sign), "evidence": deepcopy(proof)}
        components.append(row)
        return row

    total, total_proof, total_key = read("total_amount", "total_ttc")
    subtotal, base_proof, base_key = read("subtotal", "total_ht")
    vat, vat_proof, vat_key = read("tax_amount", "vat")
    stamp, stamp_proof, stamp_key = read("stamp_duty")
    discount, discount_proof, discount_key = read("discount_amount", "discount")
    extra_total, extra_proof, extra_key = read("additional_tax_total")
    adjustment, adjustment_proof, adjustment_key = read("rounding_adjustment", "adjustment_amount")
    if total is None:
        reasons.append("missing_printed_total")

    # Rates describe printed components but cannot stand in for missing amounts.
    if discount is None and _decimal(fields.get("discount_rate")) not in (None, ZERO):
        reasons.append("discount_rate_without_printed_amount")

    taxes, seen = [], set()
    tax_proof = proofs.get("taxes", {})
    if tax_proof.get("requires_review") or tax_proof.get("rejected"):
        reasons.append("rejected_tax_rows")
    for index, row in enumerate(fields.get("taxes") or []):
        amount = _decimal(row.get("amount"))
        evidence = row.get("evidence")
        proof = evidence if isinstance(evidence, dict) and "supported" in evidence else {
            "supported": _supported(tax_proof), "source": row.get("source") or evidence,
            "line_ids": row.get("line_ids"), "value": row.get("amount")}
        label = row.get("label")
        if not label or amount is None or not _supported(proof, amount):
            reasons.append("unproven_tax_row:" + str(index))
            continue
        # Grounded tax sources can overlap. Deduplicate the actual labelled source
        # line, not merely equal amounts (different levies may have equal values).
        source = _source_text(proof)
        source_lines = [line.strip() for line in source.splitlines() if _fold(label) in _fold(line)]
        location = tuple(row.get("line_ids") or proof.get("line_ids") or [])
        identity = (_fold(label), str(_decimal(row.get("rate"))), amount, location, tuple(source_lines) or source)
        if identity in seen:
            continue
        seen.add(identity)
        taxes.append((index, row, amount, proof, _tax_kind(label)))

    # Scalar TVA is the aggregate, not another tax to add on top of TVA rows.
    vat_rows = [t for t in taxes if t[4] == "vat"]
    if vat is not None:
        rate = fields.get("tax_rate") if _supported(proofs.get("tax_rate", {}), _decimal(fields.get("tax_rate"))) else None
        component(vat_key, vat, vat_proof, "vat", "TVA", rate)
        if vat_rows:
            observations.append({"kind": "vat_breakdown", "included_in_scalar": True,
                                 "rows": deepcopy([t[1] for t in vat_rows])})
            row_sum = sum((t[2] for t in vat_rows), ZERO)
            if (tax_proof.get("vat_complete") and row_sum != vat) or (vat >= ZERO and row_sum > vat):
                reasons.append("conflicting_vat_aggregate_and_rows")
    elif vat_rows:
        if not tax_proof.get("vat_complete"):
            reasons.append("vat_breakdown_completeness_unknown")
        for index, row, amount, proof, kind in vat_rows:
            component(f"taxes[{index}]", amount, proof, kind, row["label"], row.get("rate"))

    extra_rows = [t for t in taxes if t[4] not in ("vat", "stamp")]
    if extra_total is not None:
        if extra_rows and sum((t[2] for t in extra_rows), ZERO) != extra_total:
            reasons.append("conflicting_additional_tax_aggregate_and_rows")
        elif extra_rows:
            observations.append({"kind": "additional_tax_aggregate", "amount": str(extra_total),
                                 "evidence": deepcopy(extra_proof), "included_in_rows": True})
        else:
            component(extra_key, extra_total, extra_proof, "additional_tax_aggregate")
    for index, row, amount, proof, kind in extra_rows:
        component(f"taxes[{index}]", amount, proof, kind, row["label"], row.get("rate"))

    stamp_rows = [t for t in taxes if t[4] == "stamp"]
    if stamp is not None:
        component(stamp_key, stamp, stamp_proof, "stamp", "Timbre")
        if stamp_rows and sum((t[2] for t in stamp_rows), ZERO) != stamp:
            reasons.append("conflicting_stamp_aggregate_and_rows")
    else:
        for index, row, amount, proof, kind in stamp_rows:
            component(f"taxes[{index}]", amount, proof, kind, row["label"], row.get("rate"))
    if adjustment is not None:
        component(adjustment_key, adjustment, adjustment_proof, "printed_adjustment")

    financial_keys = (base_key, vat_key, stamp_key, discount_key, extra_key, adjustment_key, total_key, "taxes")
    context = "\n".join(_source_text(proofs.get(key, {})) for key in financial_keys)
    context += "\n" + "\n".join(_source_text(t[3]) for t in taxes)
    folded = _fold(context)
    for pattern, kind in ((r"\bfodec\b", "fodec"), (r"\b(redevance|levy)\b", "levy"),
                          (r"\b(timbre|stamp)\b", "stamp")):
        if re.search(pattern, folded) and not any(c["kind"] == kind for c in components):
            reasons.append("mentioned_component_without_proven_amount:" + kind)
    if re.search(r"\b(remise|discount)\b", folded) and discount is None:
        reasons.append("mentioned_discount_without_proven_amount")

    items, item_proof = fields.get("items") or [], proofs.get("items", {})
    if subtotal is not None:
        component(base_key, subtotal, base_proof, "ht_base", "HT")
        if vat is None and not vat_rows and not (_supported(base_proof) and base_proof.get("tax_exempt")):
            reasons.append("missing_proven_vat_or_exemption")
        if discount is not None and discount != ZERO:
            basis = base_proof.get("subtotal_basis")
            base_text = _fold(_source_text(base_proof))
            # Only the label attached to this printed amount establishes gross/net.
            for line in base_text.splitlines():
                amounts = [_decimal(v) for v in re.findall(r"(?<!\w)\d+(?:[ ,.]\d+)*(?!\w)", line)]
                if subtotal in amounts:
                    if re.search(r"\b(net\s*h\.?t|h\.?t\s*net)\b", line):
                        basis = "net"
                    elif re.search(r"\b(brut|gross|sub[_ ]?total)\b", line):
                        basis = "gross"
            discount_source = discount_proof.get("source")
            scope = discount_source.get("scope") if isinstance(discount_source, dict) else None
            # A printed item gross amount and item discount may establish that the
            # printed HT is already net. This does not use TTC to choose a formula.
            if basis is None and scope == "item" and len(items) == 1 and _supported(item_proof):
                row = items[0]
                gross = _decimal(row.get("total"))
                row_source = str(row.get("source") or "")
                ht_row = row.get("price_basis") == "HT" or re.search(r"\b(mt|montant|total)\s*h\.?t\b", _fold(row_source))
                same_row = discount_source.get("row_text") and discount_source["row_text"] in row_source
                if (ht_row and row.get("price_basis") != "TTC" and same_row
                        and gross is not None and gross - abs(discount) == subtotal):
                    basis = "net"
                    observations.append({"kind": "item_discount_already_in_ht", "gross_amount": str(gross),
                                         "discount_amount": str(discount), "evidence": deepcopy(discount_proof),
                                         "item_evidence": row["source"]})
            if basis == "net":
                observations.append({"kind": "discount_already_in_net_ht", "amount": str(discount),
                                     "evidence": deepcopy(discount_proof)})
            elif basis == "gross" and scope != "item":
                rate = fields.get("discount_rate") if _supported(proofs.get("discount_rate", {}), _decimal(fields.get("discount_rate"))) else None
                component(discount_key, abs(discount), discount_proof, "discount", "Remise", rate, sign=-1)
            else:
                reasons.append("discount_ht_basis_or_scope_ambiguous")
    elif items and all(item.get("price_basis") == "TTC" for item in items):
        if not _supported(item_proof) or not item_proof.get("table_complete"):
            reasons.append("ttc_table_completeness_unproven")
        for index, item in enumerate(items):
            amount = _decimal(item.get("total"))
            if amount is None or not item.get("source"):
                reasons.append("unproven_ttc_line_amount:" + str(index))
            else:
                component(f"items[{index}].total", amount, {"source": item["source"], "line_ids": item.get("line_ids")},
                          "ttc_line", item.get("description"))
        if vat is not None or vat_rows or extra_rows or extra_total is not None:
            reasons.append("tax_inclusion_in_ttc_lines_ambiguous")
        if discount not in (None, ZERO):
            reasons.append("discount_inclusion_in_ttc_lines_ambiguous")
    else:
        reasons.append("missing_proven_ht_or_complete_ttc_lines")

    def result_for(name, printed, evidence, used, missing):
        calculated = None if missing else sum((_decimal(c["contribution"]) for c in used), ZERO)
        difference = None if calculated is None or printed is None else printed - calculated
        bound = policy.bound(fields.get("currency"), sum(_decimal(c["contribution"]) != ZERO for c in used))
        if missing or difference is None:
            status = "INSUFFICIENT_EVIDENCE"
        elif difference == ZERO:
            status = "VALID"
        elif abs(difference) <= bound:
            status = "ROUNDING_DIFFERENCE"
        else:
            status = "INVALID"
        return {"check": name, "status": status, "scope": "proven_printed_components_only",
                "calculated_amount": None if calculated is None else str(calculated),
                "printed_amount": None if printed is None else str(printed),
                "difference": None if difference is None else str(difference),
                "components": deepcopy(used), "printed_evidence": deepcopy(evidence),
                "reasons": list(dict.fromkeys(missing)), "requires_review": status in ("INVALID", "INSUFFICIENT_EVIDENCE"),
                "rounding": {"bound": str(bound), "rule": "half_currency_unit_per_printed_term_and_result",
                             "currency": fields.get("currency"), "currency_evidence": deepcopy(proofs.get("currency")),
                             "enabled": policy.enabled}, "observations": deepcopy(observations)}

    primary = result_for("printed_total", _decimal(fields.get(total_key)), total_proof, components, reasons)
    # Net payable may differ from TTC (deposits, withholding). Without explicit
    # settlement evidence, equality with TTC must not be silently assumed.
    if fields.get("net_to_pay") is not None or proofs.get("net_to_pay", {}).get("requires_review"):
        before = len(reasons)
        net, net_proof, _ = read("net_to_pay")
        settlement_reasons = reasons[before:]
        settlement = []
        if total is None:
            settlement_reasons.append("missing_proven_ttc_for_net_payable")
        else:
            settlement.append({"field": total_key, "kind": "printed_ttc", "amount": str(total),
                               "contribution": str(total), "evidence": deepcopy(total_proof)})
        offset, offset_proof, offset_key = read("payment_adjustment")
        if offset is not None:
            settlement.append({"field": offset_key, "kind": "printed_settlement_adjustment", "amount": str(offset),
                               "contribution": str(offset), "evidence": deepcopy(offset_proof)})
        settlement_reasons.extend(reasons[before:])
        if not net_proof.get("settlement_complete"):
            settlement_reasons.append("settlement_components_unknown")
        if net is None:
            settlement_reasons.append("missing_proven_net_payable")
        checks.append(result_for("net_to_pay", _decimal(fields.get("net_to_pay")), net_proof, settlement, settlement_reasons))
    primary["related_checks"] = checks
    primary["overall_requires_review"] = primary["requires_review"] or any(c["requires_review"] for c in checks)
    return primary


def financial_validation_issues(fields, proofs=None, *, rounding_policy=None):
    """Adapt structured results to the existing validation-issues API."""
    result = validate_invoice_finances(fields, proofs, rounding_policy=rounding_policy)
    codes = {"VALID": ("invoice_total_valid", "info"),
             "INVALID": ("invoice_total_math_mismatch", "blocking"),
             "INSUFFICIENT_EVIDENCE": ("invoice_financial_insufficient_evidence", "warning"),
             "ROUNDING_DIFFERENCE": ("invoice_total_rounding_difference", "info")}
    issues = []
    for check in [result, *result["related_checks"]]:
        code, severity = codes[check["status"]]
        if check["check"] != "printed_total":
            code = "net_to_pay_" + check["status"].lower()
        issues.append({"code": code, "severity": severity, "status": check["status"],
                       "requires_review": check["requires_review"],
                       "message": "Printed amounts retained. Financial check: " + check["status"],
                       "metadata": check})
    return issues
