"""Deterministic invoice-table rules used before and after LLM extraction.

The LLM may help decide between OCR candidates, but it must never turn a
header, address or phone number into an invoice line.  These small, dependency
free rules make that boundary explicit and easy to test.
"""
import re
import unicodedata
from typing import Any, Dict, List, Tuple


def extract_alternating_table_rows(text):
    """Recognize complete numeric/description pairs, never guess missing cells."""
    from decimal import Decimal
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header = next((i for i, line in enumerate(lines)
                   if re.search(r"d[ée]signation", line, re.I)
                   and re.search(r"qte|quantit", line, re.I)), None)
    if header is None:
        return []
    body = lines[header + 1:]
    end = next((i for i, line in enumerate(body)
                if re.match(r"(?:total|tva|timbre|subtotal)\b", line, re.I)), len(body))
    body = body[:end]
    if len(body) < 4 or len(body) % 2:
        return []
    rows = []
    for i in range(0, len(body), 2):
        numeric, description = body[i:i + 2]
        match = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*\|\s*(\d+[.,]\d+)\s*\|\s*(\d+[.,]\d+)", numeric)
        if not match or not re.search(r"[A-Za-zÀ-ÿ]", description) or "|" in description:
            return []
        quantity, price, total = [Decimal(value.replace(',', '.')) for value in match.groups()]
        if abs(quantity * price - total) > Decimal('0.005'):
            return []
        rows.append({"description": description, "quantity": float(quantity),
                     "unit_price": float(price), "total": float(total),
                     "price_basis": "HT" if re.search(r"\bHT\b", lines[header]) else "unknown",
                     "source": numeric + "\n" + description})
    return rows


def extract_unit_table_rows(text):
    """Read explicit unit/quantity/price/tax/total cells; never use footer bases."""
    rows, active = [], False
    for line in text.splitlines():
        if re.search(r'd[ée]signation', line, re.I) and re.search(r'P\.?U\.?H\.?T', line, re.I):
            active = True
            continue
        if active and re.match(r'\s*(?:TOTAL|TAUX|BASE|T\.V\.A)', line, re.I):
            break
        if not active:
            continue
        cells = [cell.strip() for cell in line.split('|')]
        if len(cells) != 6 or not re.fullmatch(r'TONNE|KG|U|UNITE|M2|M3', cells[1], re.I):
            continue
        if not all(re.fullmatch(r'\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?', c) for c in cells[2:]):
            continue
        description = re.sub(r'^[\d$]{6,}\s+', '', cells[0])
        rows.append({'description': description, 'quantity': parse_invoice_amount(cells[2]),
                     'unit_price': parse_invoice_amount(cells[3]), 'total': parse_invoice_amount(cells[5]),
                     'unit': cells[1], 'source': line})
        reference = re.match(r'^(\d{6,})\s+', cells[0])
        if reference:
            rows[-1]['reference'] = reference[1]
    return rows


LINE_ITEM_FORBIDDEN_TERMS = {
    "adresse", "address", "rue", "avenue", "route", "telephone", "tel",
    "fax", "email", "mail", "site", "web", "client", "customer",
    "fournisseur", "supplier", "vendeur", "seller", "facture", "invoice",
    "date", "page", "mf", "tva", "taxe", "timbre", "total", "montant",
    "cachet", "signature", "livraison", "reglement", "paiement",
}

LINE_ITEM_HEADER_TERMS = {
    "base", "taux",
    "code", "article", "reference", "ref", "designation", "désignation",
    "quantite", "quantité", "qte", "qté", "prix", "unitaire", "pu",
    "ht", "ttc", "montant", "total", "tva", "remise", "taxes",
}

TABLE_START_PATTERN = re.compile(
    r"\b(?:code\s+article|reference|r[ée]f[ée]rence|designation|d[ée]signation|"
    r"quantit[ée]|qte|qt[ée]|pu\s*(?:ht|ttc)?|prix\s+unitaire|description|quantity|qty|unit\s+price)\b",
    flags=re.IGNORECASE,
)
TABLE_END_PATTERN = re.compile(
    r"\b(?:base\s+tva|total\s+tva|total\s+ttc|montant\s+ttc|montant\s+total\s+en\s+dinar|droit\s+de\s+timbre|"
    r"arr[ée]t[ée]e?\s+la\s+pr[ée]sente|cachet\s+et\s+signature|signature\s+client|subtotal|grand\s+total)\b",
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
    return "".join(char for char in decomposed if char.isalnum())


def _description_words(value: str) -> List[str]:
    return [word.lower() for word in re.findall(r"[^\W\d_]{3,}", value or "", flags=re.UNICODE)]


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
    present = [item.get("quantity"), item.get("unit_price"), item.get("total", item.get("total_price"))]
    if sum(value is not None and value != "" for value in present) < 2:
        return "missing_line_amount"
    if quantity < 0 or unit_price < 0 or line_total < 0:
        return "negative_line_amount"
    tolerance = max(0.05, line_total * 0.01)
    if all(value is not None and value != "" for value in present) and abs((quantity * unit_price) - line_total) > tolerance:
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
    basis_labels = set(re.findall(r"\b(?:p\.?\s*u\.?|prix\s+unitaire)\s*\|?\s*(H\.?T\.?|TTC)\b", table_text or "", re.I))
    basis_labels = {label.upper().replace('.', '') for label in basis_labels}
    printed_basis = next(iter(basis_labels)) if len(basis_labels) == 1 else None

    for original in items or []:
        item = dict(original or {})
        reason = _line_item_reason(item, table_text, invoice_total)
        description = str(item.get("description") or item.get("designation") or "").strip()
        raw_total = item.get("total", item.get("total_price"))
        total = parse_invoice_amount(raw_total) if raw_total is not None and raw_total != "" else None
        key = (_canonical(description), total,
               parse_invoice_amount(item["quantity"]) if item.get("quantity") is not None else None,
               parse_invoice_amount(item["unit_price"]) if item.get("unit_price") is not None else None)
        duplicate = key in seen
        if reason:
            rejected.append({"description": description[:180], "reason": reason})
            continue
        accepted_row = {
            "description": description[:180],
            "quantity": parse_invoice_amount(item["quantity"]) if item.get("quantity") is not None else None,
            "unit_price": parse_invoice_amount(item["unit_price"]) if item.get("unit_price") is not None else None,
            "total": total,
            "confidence": min(float(item.get("confidence", 0.72)), 0.92),
            "source": item.get("source") or "ocr_table_rules",
        }
        source_is_proven = str(accepted_row["source"]) in (table_text or "")
        if source_is_proven and item.get("unit") and any(str(item["unit"]).casefold() == cell.strip().casefold()
                                    for line in str(accepted_row["source"]).splitlines() for cell in line.split("|")):
            accepted_row["unit"] = item["unit"]
        # Only metadata cells in this cited row may complete its fallback.
        header = None
        for line in (table_text or "").splitlines():
            cells = [cell.strip() for cell in line.split("|")]
            if any(re.fullmatch(r"d[\u00e9e]signation|description", c, re.I) for c in cells):
                header = cells
                continue
            if (not source_is_proven or line not in str(accepted_row["source"]).splitlines() or not header or len(cells) != len(header)
                    or _canonical(description) not in _canonical(line)):
                continue
            for column, label in enumerate(header):
                metadata_key = ("unit" if re.fullmatch(r"unit[\u00e9e]?", label, re.I) else
                                "reference" if re.fullmatch(r"r[\u00e9e]f[\u00e9e]rence|ref\.?|code\s*article", label, re.I) else None)
                if metadata_key and cells[column]:
                    accepted_row[metadata_key] = cells[column]
        if source_is_proven and item.get("reference") and "reference" not in accepted_row:
            token = str(item["reference"])
            source = str(accepted_row["source"])
            if any(token == cell.strip() for line in source.splitlines() for cell in line.split("|")) or re.match(
                    re.escape(token) + r"\s+" + re.escape(description) + r"\s*\|", source):
                accepted_row["reference"] = token
        if printed_basis:
            accepted_row["price_basis"] = printed_basis
        elif item.get("price_basis") in {"HT", "TTC"} and str(item.get("source", "")).endswith("_visual"):
            accepted_row["price_basis"] = item["price_basis"]
        if duplicate:
            existing = next(row for row in accepted if (
                _canonical(row["description"]), row["total"], row["quantity"], row["unit_price"]) == key)
            if existing["source"] == accepted_row["source"]:
                for metadata_key in ("unit", "reference"):
                    if not existing.get(metadata_key) and accepted_row.get(metadata_key):
                        existing[metadata_key] = accepted_row[metadata_key]
            rejected.append({"description": description[:180], "reason": "duplicate_ocr_line"})
            continue
        seen.add(key)
        accepted.append(accepted_row)
    return accepted[:30], rejected[:30]
