"""Structured visual proposals, with independent OCR corroboration metadata."""
import math
import re
import unicodedata
from datetime import datetime

from models import CustomerInfo, InvoiceData, InvoiceItem, PaymentInfo, SupplierInfo
from invoice_rules import parse_invoice_amount


TEXT_FIELDS = (
    "invoice_number date due_date vendor_name vendor_address vendor_tax_id vendor_phone vendor_email "
    "customer_name customer_address customer_tax_id customer_phone customer_email currency "
    "payment_method payment_terms rib amount_in_words tax_rate"
).split()
MONEY_FIELDS = ("subtotal", "tax_amount", "stamp_duty", "total_amount", "discount")
MONEY_LABELS = {
    "subtotal": r"\bht\b|hors\s*tax|subtotal|sous.total|دون.*(?:ضريب|أداء|اداء)",
    "tax_amount": r"\btva\b|\bvat\b|tax|القيمة.*المضافة",
    "stamp_duty": r"timbre|stamp|طابع",
    "total_amount": r"ttc|net.*payer|grand.*total|total.*(?:due|amount)|شامل|الجمل",
    "discount": r"remise|discount|rabais|تخفيض|خصم",
}


def response_schema():
    props = {field: {"type": ["string", "null"]} for field in TEXT_FIELDS}
    props.update({field: {"type": ["number", "null"]} for field in MONEY_FIELDS})
    proof = {"type": "object", "properties": {
        "quote": {"type": "string"}, "page": {"type": "integer"},
    }, "required": ["quote", "page"], "additionalProperties": False}
    props["evidence"] = {"type": "array", "items": {**proof, "properties": {
        **proof["properties"], "field": {"type": "string", "enum": TEXT_FIELDS + list(MONEY_FIELDS)},
    }, "required": ["field", "quote", "page"]}}
    props["items"] = {"type": "array", "items": {
        "type": "object", "properties": {
            "description": {"type": "string"}, "reference": {"type": ["string", "null"]},
            **{key: {"type": ["number", "null"]} for key in ("quantity", "unit_price", "total")},
            "price_basis": {"type": "string", "enum": ["HT", "TTC", "unknown"]},
            **proof["properties"],
        }, "required": ["description", "reference", "quantity", "unit_price", "total", "price_basis", "quote", "page"],
        "additionalProperties": False,
    }}
    # Required nullable fields avoid an exponential number of optional schema states.
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def canonical(value):
    return "".join(c for c in unicodedata.normalize("NFKD", str(value or "")).casefold() if c.isalnum())


def numbers_in_quote(quote, split_columns=False):
    text = str(quote).replace("\u066b", ".").replace("\u066c", " ")
    # A row such as '3 100,00 300,00' contains quantity 3 and price 100,
    # while '1 853,130' can be a grouped amount. Keep both readings for row checks.
    matches = list(re.finditer(r"\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?", text))
    if split_columns:
        matches += list(re.finditer(r"\d+(?:[.,]\d+)?", text))
    return [parse_invoice_amount(m.group()) for m in matches
            if not text[m.end():].lstrip().startswith(("%", "\u066a"))]


def value_in_quote(field, value, quote):
    if field in MONEY_FIELDS or field in {"quantity", "unit_price", "total"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            return False
        if field in MONEY_LABELS and not re.search(MONEY_LABELS[field], quote, re.I):
            return False
        return any(abs(n - value) <= .0005 for n in numbers_in_quote(quote, split_columns=field not in MONEY_FIELDS))
    if not isinstance(value, str) or not canonical(value):
        return False
    if re.fullmatch(r"\s*\[[^\]]+\]\s*", value):
        return False
    if field == "currency" and not re.fullmatch(r"[A-Z]{3}", value):
        return False
    if field == "currency" and value == "EUR" and "\u20ac" in quote:
        return True
    if field in {"date", "due_date"}:
        match = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", value.strip())
        if not match:
            return False
        try:
            datetime(int(match[3]), int(match[2]), int(match[1]))
        except ValueError:
            return False
    return canonical(value) in canonical(quote)


def parse_visual_invoice(result, ocr_text, page_count, has_document, model, provider="gemini"):
    """A model quote is a proposal, never independent proof of its own accuracy."""
    evidence = {}
    for item in result.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        field, quote, page = item.get("field"), item.get("quote"), item.get("page")
        if (field in TEXT_FIELDS + list(MONEY_FIELDS) and isinstance(quote, str) and 0 < len(quote.strip()) <= 1200
                and type(page) is int and 1 <= page <= page_count):
            evidence[field] = {"text": quote.strip(), "page_number": page}
    fields, proofs, rejected = {}, {}, []
    for field in TEXT_FIELDS + list(MONEY_FIELDS):
        value, proof = result.get(field), evidence.get(field)
        if value is None:
            continue
        if not proof or not value_in_quote(field, value, proof["text"]):
            rejected.append(field)
            continue
        corroborated = canonical(proof["text"]) in canonical(ocr_text)
        if not has_document and not corroborated:
            rejected.append(field)
            continue
        fields[field] = value
        proofs[field] = {**proof, "value": value, "ocr_corroborated": corroborated,
                         "requires_review": not corroborated, "model": model}

    rows = []
    for row in result.get("items") or []:
        if not isinstance(row, dict):
            continue
        quote, page = row.get("quote"), row.get("page")
        if not isinstance(quote, str) or not quote.strip() or len(quote) > 1200 or type(page) is not int or not 1 <= page <= page_count:
            continue
        if not value_in_quote("description", row.get("description"), quote):
            continue
        if any(row.get(key) is None or not value_in_quote(key, row[key], quote) for key in ("quantity", "unit_price", "total")):
            continue
        corroborated = canonical(quote) in canonical(ocr_text)
        if not has_document and not corroborated:
            continue
        rows.append({**row, "source": f"{provider}_visual" if has_document else f"{provider}_ocr",
                     "requires_review": not corroborated, "ocr_corroborated": corroborated})
    fields["items"] = rows
    return InvoiceData(
        invoice_number=fields.get("invoice_number"), invoice_date=fields.get("date"), due_date=fields.get("due_date"),
        subtotal=fields.get("subtotal"), tax_amount=fields.get("tax_amount"), total_amount=fields.get("total_amount"),
        currency=fields.get("currency"),
        supplier=SupplierInfo(**{key: fields.get("vendor_" + field) for key, field in
                               {"name": "name", "address": "address", "phone": "phone", "email": "email", "vat_number": "tax_id"}.items()}),
        customer=CustomerInfo(**{key: fields.get("customer_" + field) for key, field in
                               {"name": "name", "address": "address", "phone": "phone", "email": "email", "vat_number": "tax_id"}.items()}),
        items=[InvoiceItem(description=r["description"], designation=r["description"], quantity=r["quantity"],
                           unit_price=r["unit_price"], total_price=r["total"], reference=r.get("reference")) for r in rows],
        payment_info=PaymentInfo(method=fields.get("payment_method"), terms=fields.get("payment_terms"), bank_details=fields.get("rib")),
        additional_data={"cloud_result": fields, "cloud_evidence": proofs, "grounding_rejected_fields": rejected,
                         "model_used": model, "analysis_method": f"{provider}_visual" if has_document else f"{provider}_ocr"},
    )
