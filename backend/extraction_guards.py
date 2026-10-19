"""Conservative party/identifier checks independent of the OCR or LLM provider."""
import re
import unicodedata
from datetime import datetime
from decimal import Decimal, InvalidOperation


DATE_TOKEN = re.compile(r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[./-]\d{1,2}[./-]\d{4}|\d{1,2}[- ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[- ]\d{4})\b", re.I)


def normalized_date(value):
    for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d-%b-%Y", "%d %b %Y"):
        try:
            return datetime.strptime(str(value).strip(), pattern).date().isoformat()
        except (ValueError, TypeError):
            pass
    return ""


def date_in_source(value, text):
    expected = normalized_date(value)
    return bool(expected and any(normalized_date(m.group()) == expected for m in DATE_TOKEN.finditer(text or "")))


def monetary_in_source(field, value, text):
    if field == "discount":
        from invoice_value_evidence import decimal_value, discount_evidence
        return any(decimal_value(value) == decimal_value(entry["amount"]) for entry in discount_evidence(text))
    if field == "stamp_duty":
        from invoice_value_evidence import decimal_value, stamp_label_value_pairs
        if any(decimal_value(value) == row["value"] for row in stamp_label_value_pairs(text or "")):
            return True
    labels = {
        "subtotal": r"(?:total\s+(?:net\s+)?HT|montant\s+HT|sub[ _]?total|hors\s+taxes?|المبلغ\s+دون\s+احتساب\s+[اأ]ل[اأ]داء)",
        "tax_amount": r"(?:T\.?V\.?A\.?|VAT|tax(?:\s+amount)?|[اأ]ل[اأ]داء\s+على\s+القيمة\s+المضافة)",
        "stamp_duty": r"(?:timbre(?:\s+fiscal)?|stamp(?:\s+duty)?|معلوم\s+الطابع)",
        "total_amount": r"(?:total\s+TTC|montant\s+(?:total\s+(?:de\s+la\s+facture\s+)?)?(?:\(en\s+(?:DT|TND)\s+TTC\)|TTC)|net\s+[aà]\s+payer|total\s+amount|grand\s+total|المبلغ\s+الجملي|(?m:^\s*total\b(?!\s*(?:HT|TVA|tax))))",
        "discount": r"(?:remise|discount)",
    }
    try:
        expected = Decimal(str(value).replace(",", "."))
    except InvalidOperation:
        return False
    rate = r"(?:[\s|]*\(?\d{1,2}(?:[.,]\d+)?\s*%\)?)*"
    text = "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in (text or ""))
    text = text.replace("\u066b", ".").replace("\u066c", " ").replace("\u066a", "%")
    number = r"([0-9]+(?:[ \u00a0\u202f][0-9]{3})*(?:[.,][0-9]+)?)"
    pattern = labels[field] + rate + r"[\s:|=]*(?:TND|USD|EUR|\$|\u20ac)?\s*" + number + r"(?![\d.,]|\s*%)"
    for token in re.findall(pattern, text or "", re.I):
        amount = Decimal(re.sub(r"[ \u00a0\u202f]", "", token).replace(",", "."))
        if amount == expected:
            return True
    return False


def compact(text):
    return "".join(c for c in unicodedata.normalize("NFKD", str(text or "")).casefold() if c.isalnum())


def named_party_contacts(name, text):
    """Use only a unique explicitly named address block, not nearby stamps."""
    if len(compact(name)) < 5:
        return {}
    lines = text.splitlines()
    anchors = []
    for index, line in enumerate(lines):
        prefix, separator, address = line.partition(':')
        if separator and compact(prefix) == compact(name) and re.match(
                r'\s*(?:IMM\b|immeuble\b|rue\b|avenue\b|route\b|adresse\b|\d+\s+)', address, re.I):
            anchors.append((index, address.strip()))
    if len(anchors) != 1:
        return {}
    index, address = anchors[0]
    block = [lines[index]]
    for line in lines[index + 1:index + 4]:
        if not re.match(r'\s*(?:t[ée]l(?:[ée]phone)?\.?|email|e-mail|MF)\s*:', line, re.I):
            break
        block.append(line)
    source = '\n'.join(block)
    contacts = {'address': address}
    phone = re.search(r't[ée]l(?:[ée]phone)?\.?\s*:\s*(\+?\d[\d .()-]{6,}\d)', source, re.I)
    if phone and len(re.sub(r'\D', '', phone.group(1))) in (8, 11):
        contacts['phone'] = phone.group(1).strip()
    email = re.search(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', source)
    if email:
        contacts['email'] = email.group()
    return {'fields': contacts, 'text': source, 'line_start': index + 1, 'line_end': index + len(block)}


def complete_tax_id(value, source):
    if re.search(r'[:|]|\b(?:MF|R\.?C)\b', str(value), re.I):
        return False
    expected = compact(value)
    if not expected or expected not in compact(source):
        return False
    candidates = re.findall(r"\b\d{7}\s*[A-Za-z]?(?:\s*/\s*[A-Za-z0-9]{1,3}){0,4}\b", source)
    same_root = [compact(c) for c in candidates if compact(c).startswith(expected[:7])]
    return expected in same_root if same_root else True


def corroborated_supplier_header(text):
    """Require the heading to recur in the immediately following company line."""
    lines = text.splitlines()
    if len(lines) < 2:
        return {}
    name = lines[0].split('|')[0].strip()
    if len(compact(name)) < 8 or not re.search(r'soci[ée]t[ée]|SARL|S\.A\.', lines[1], re.I):
        return {}
    if compact(name) not in compact(lines[1]):
        return {}
    end = next((i for i, line in enumerate(lines[:10]) if re.search(r'\b(?:client|facture|invoice|designation)\b', line, re.I)), min(10, len(lines)))
    source = '\n'.join(lines[:end])
    fields = {'vendor_name': name}
    identifiers = re.findall(r'\bMF\s*:\s*(\d{6,7}[A-Z](?:/[A-Z0-9]{1,3}){3})\b', source, re.I)
    if len(set(identifiers)) == 1:
        fields['vendor_tax_id'] = identifiers[0]
    address = re.search(r'^S\.S\s*:\s*([^|\n]+)', source, re.M | re.I)
    if address:
        fields['vendor_address'] = address.group(1).strip()
    return {'fields': fields, 'text': source, 'line_start': 1, 'line_end': end}


def explicitly_customer(value, source):
    expected = compact(value)
    if len(expected) < 4:
        return False
    for line in source.splitlines():
        match = re.search(r"(?:nom\s+du\s+client|client|customer|bill\s+to|sold\s+to)\s*[:|=-]\s*(.+)", line, re.I)
        if match and expected in compact(match.group(1)):
            return True
    return False


def customer_zone(source):
    lines = source.splitlines()
    start = None
    for index, line in enumerate(lines):
        explicit = re.search(r"(?:nom\s+du\s+client|bill\s+to|sold\s+to|customer)\s*[:|]", line, re.I)
        named_block = re.search(r"raison\s*sociale", line, re.I) and re.search(r"code\s+client", "\n".join(lines[index:index + 9]), re.I)
        if explicit or named_block:
            start = index
            break
    if start is None:
        return ""
    end = min(start + 12, len(lines))
    for index in range(start + 1, end):
        if re.search(r"\b(?:designation|d[ée]signation|description|prix\s+unitaire|quantity)\b", lines[index], re.I):
            end = index
            break
    return "\n".join(lines[start:end])


def party_conflicts(fields, text):
    conflicts = []
    client_text = customer_zone(text)
    for suffix in ("name", "tax_id", "address", "phone", "email"):
        vendor, customer = fields.get("vendor_" + suffix), fields.get("customer_" + suffix)
        if not vendor:
            continue
        if suffix == "name" and compact(vendor) in {"invoice", "facture", "commercialinvoice", "fournisseur", "supplier"}:
            conflicts.append("vendor_name")
            continue
        same = customer and compact(vendor) == compact(customer)
        if suffix == "name" and explicitly_customer(vendor, text):
            conflicts.append("vendor_name")
        elif suffix == "tax_id" and client_text and compact(vendor) in compact(client_text):
            from identifier_evidence import identifier_spans, _line_roles
            lines = text.splitlines()
            visible = set(range(1, len(lines) + 1))
            owners = set()
            for i, line in enumerate(lines, 1):
                for start, _ in identifier_spans("supplier_tax_id", str(vendor), line):
                    owners.update(_line_roles(lines, i, start, visible, None))
            if owners != {"supplier"}:
                conflicts.append("vendor_tax_id")
        elif same and explicitly_customer(fields.get("customer_name"), text):
            conflicts.append("vendor_" + suffix)
    return conflicts
