"""Deterministic invoice-table rules used before and after LLM extraction.

The LLM may help decide between OCR candidates, but it must never turn a
header, address or phone number into an invoice line.  These small, dependency
free rules make that boundary explicit and easy to test.
"""
import re
import unicodedata
from typing import Any, Dict, List, Tuple


LINE_ITEM_FORBIDDEN_TERMS = {
    "adresse", "address", "rue", "avenue", "route", "telephone", "tel",
    "fax", "email", "mail", "site", "web", "client", "customer",
    "fournisseur", "supplier", "vendeur", "seller", "facture", "invoice",
    "date", "page", "mf", "tva", "taxe", "timbre", "total", "montant",
    "cachet", "signature", "livraison", "reglement", "paiement",
}

LINE_ITEM_HEADER_TERMS = {
    "code", "article", "reference", "ref", "designation", "désignation",
    "quantite", "quantité", "qte", "qté", "prix", "unitaire", "pu",
    "ht", "ttc", "montant", "total", "tva", "remise", "taxes",
}

TABLE_START_PATTERN = re.compile(
    r"\b(?:code\s+article|reference|r[ée]f[ée]rence|designation|d[ée]signation|"
    r"quantit[ée]|qte|qt[ée]|pu\s*(?:ht|ttc)?|prix\s+unitaire)\b",
    flags=re.IGNORECASE,
)
TABLE_END_PATTERN = re.compile(
    r"\b(?:base\s+tva|total\s+tva|total\s+ttc|montant\s+ttc|montant\s+total\s+en\s+dinar|droit\s+de\s+timbre|"
    r"arr[ée]t[ée]e?\s+la\s+pr[ée]sente|cachet\s+et\s+signature|signature\s+client)\b",
    flags=re.IGNORECASE,
)


def parse_invoice_amount(value: Any) -> float:
    """Parse Tunisian monetary OCR without treating millimes as thousands."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^\d,.\-]", "", str(value).replace(" ", ""))
    if not cleaned:
        return 0.0
    if "," in cleaned and "." in cleaned:
        decimal_separator = "," if cleaned.rfind(",") > cleaned.rfind(".") else "."
        thousands_separator = "." if decimal_separator == "," else ","
        cleaned = cleaned.replace(thousands_separator, "").replace(decimal_separator, ".")
    elif cleaned.count(".") > 1:
        parts = cleaned.split(".")
        cleaned = "".join(parts[:-1]) + "." + parts[-1] if all(len(part) == 3 for part in parts[1:]) else "".join(parts)
    else:
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def extract_invoice_table_text(text: str) -> str:
    """Return the OCR section between an article-table header and its totals."""
    match = TABLE_START_PATTERN.search(text or "")
    if not match:
        return ""
    table_text = (text or "")[match.start():]
    end = TABLE_END_PATTERN.search(table_text)
    return table_text[:end.start()] if end else table_text


def _canonical(value: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or "").lower())
    ascii_value = decomposed.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", ascii_value)


def _description_words(value: str) -> List[str]:
    return [word.lower() for word in re.findall(r"[A-Za-zÀ-ÿ]{3,}", value or "")]


def _line_item_reason(item: Dict[str, Any], table_text: str, total_amount: float) -> str:
    description = str(item.get("description") or item.get("designation") or "").strip()
    words = _description_words(description)
    if len(description) < 3 or not words:
        return "description_missing"
    if any(character in description for character in ("|", "_", "[", "]")):
        return "ocr_table_artifact"
    normalized_words = {_canonical(word) for word in words}
    header_terms = {_canonical(term) for term in LINE_ITEM_HEADER_TERMS}
    forbidden_terms = {_canonical(term) for term in LINE_ITEM_FORBIDDEN_TERMS}
    if normalized_words.issubset(header_terms) or normalized_words.intersection(header_terms):
        return "table_header"
    if normalized_words.intersection(forbidden_terms):
        return "metadata_or_address"

    table_canonical = _canonical(table_text)
    if not table_canonical or not all(_canonical(word) in table_canonical for word in words):
        return "not_proven_inside_article_table"

    quantity = parse_invoice_amount(item.get("quantity"))
    unit_price = parse_invoice_amount(item.get("unit_price"))
    line_total = parse_invoice_amount(item.get("total", item.get("total_price")))
    if quantity <= 0 or unit_price <= 0 or line_total <= 0:
        return "missing_line_amount"
    tolerance = max(0.05, line_total * 0.01)
    if abs((quantity * unit_price) - line_total) > tolerance:
        return "line_math_mismatch"
    if total_amount > 0 and line_total > total_amount + tolerance:
        return "line_exceeds_invoice_total"
    return ""


def filter_invoice_line_items(
    items: List[Dict[str, Any]],
    table_text: str,
    total_amount: Any = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep only structurally proven invoice lines and return auditable rejects."""
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen = set()
    invoice_total = parse_invoice_amount(total_amount)

    for original in items or []:
        item = dict(original or {})
        reason = _line_item_reason(item, table_text, invoice_total)
        description = str(item.get("description") or item.get("designation") or "").strip()
        total = parse_invoice_amount(item.get("total", item.get("total_price")))
        key = (_canonical(description), round(total, 3))
        if not reason and key in seen:
            reason = "duplicate_ocr_line"
        if reason:
            rejected.append({"description": description[:180], "reason": reason})
            continue
        seen.add(key)
        accepted.append({
            "description": description[:180],
            "quantity": parse_invoice_amount(item.get("quantity")),
            "unit_price": parse_invoice_amount(item.get("unit_price")),
            "total": total,
            "confidence": min(float(item.get("confidence", 0.72)), 0.92),
            "source": item.get("source") or "ocr_table_rules",
        })
    return accepted[:30], rejected[:30]
