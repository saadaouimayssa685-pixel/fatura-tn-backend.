"""Cell-scoped ownership of supplied evidence; never retrieve hidden context."""
import re


ROLE = re.compile(
    r"\b(?P<ship_to>ship\s+to|livr[ee]\s+[a\u00e0]|destinataire\s+livraison)\b|"
    r"\b(?P<supplier>fournisseur|supplier|vendor|seller|[e\u00e9]metteur)\b|"
    r"\b(?P<customer>(?:nom\s+du\s+)?client|customer|bill\s+to|sold\s+to)\b", re.I)
TABLE = re.compile(r"\b(?:d[\u00e9e]signation|description|quantity|quantit[\u00e9e]|items|prix\s+unitaire)\b|---\s*Page", re.I)
ADDRESS = re.compile(r"\b(?:adresse|address)\s*:", re.I)
POSTAL = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\s+(?:US|USA)\s*$", re.I)
NEW_LABEL = re.compile(
    r"^(?:invoice|facture|date|due\s+date|PO\s+number|BCN|code\s*(?:client|TVA)|"
    r"raison\s*sociale?|nom\s+du\s+client|ship\s+to|bill\s+to|sold\s+to|"
    r"supplier|customer|fournisseur|client|chantier|adresse|address|t[\u00e9e]l|fax|"
    r"e-?mail|GSTIN|VAT\s*ID|MF|R\.?C\.?|registre|registration|total|sub_total|"
    r"montant|TVA|FODEC|paiement|payment|net\s+[a\u00e0]\s+payer|"
    r"d[\u00e9e]signation|description|items|quantit[\u00e9e]|quantity|unit\s+price|"
    r"terms|conditions|bank|banque|website|site|www\.)\b", re.I)


def key(value):
    return re.sub(r"\s+", "", str(value)).casefold()


def cited_lines(text, reference, visible):
    lines = text.splitlines()
    match = re.fullmatch(r"L(\d+)(?:-L?(\d+))?", reference or "") if isinstance(reference, str) else None
    if match:
        first, last = int(match[1]), int(match[2] or match[1])
        ids = set(range(first, last + 1))
        return ids if 1 <= first <= last <= len(lines) and ids <= visible else set()
    if isinstance(reference, str) and reference:
        parts = reference.splitlines()
        starts = [i for i in range(len(lines)) if lines[i:i + len(parts)] == parts]
        if len(starts) == 1:
            ids = set(range(starts[0] + 1, starts[0] + len(parts) + 1))
            return ids if ids <= visible else set()
    return set()


def cells_by_line(lines, visible):
    """Keep physical columns. Resolve a collapsed postal tail only if unique."""
    result = {}
    previous = []
    for i, line in enumerate(lines, 1):
        if i not in visible or not line.strip():
            previous = []
            continue
        cells = [c.strip() for c in line.split("|")]
        columns = list(range(len(cells)))
        if len(cells) == 1 and len(previous) > 1 and POSTAL.search(cells[0]):
            unfinished = [col for col, cell in previous
                          if col is not None and re.match(r"\d+\s+\w", cell) and not POSTAL.search(cell)]
            columns = [unfinished[0] if len(unfinished) == 1 else None]
        result[i] = list(zip(columns, cells))
        previous = result[i]
    return result


def cell_role(lines, line_id, column, visible, metadata=None):
    cells = cells_by_line(lines, visible)
    if column is None or line_id not in cells:
        return set()
    own = next((cell for col, cell in cells[line_id] if col == column), "")
    paired_label = next((cell for col, cell in cells[line_id] if col == column - 1), "")
    if re.fullmatch(r"\s*(?:code\s+client|customer\s+code)\s*:?\s*", paired_label, re.I) and not ROLE.search(own):
        sections = {p.get("section") for p in (metadata or {}).values()
                    if line_id in p.get("line_ids", []) and p.get("section") in {"supplier", "customer", "ship_to"}}
        return {"customer"} | sections
    if ADDRESS.search(own) and len(cells[line_id]) == 1:
        following = []
        for i in range(line_id + 1, min(len(lines), line_id + 4) + 1):
            if i not in cells or TABLE.search(lines[i - 1]):
                break
            following.append(lines[i - 1])
        if re.search(r"^\s*code\s+client\s*$", "\n".join(following), re.I | re.M):
            return {"customer"}
    block = []
    for i in range(line_id, max(0, line_id - 12), -1):
        if i not in cells or TABLE.search(lines[i - 1]):
            break
        matches = [cell for col, cell in cells[i] if col == column]
        if not matches:
            continue
        cell = matches[0]
        parties = list(ROLE.finditer(cell))
        if parties:
            return {parties[-1].lastgroup}
        block.append((i, cell))
        if re.search(r"raison\s*sociale?", cell, re.I):
            return {"customer"}
        # A billing column and a labelled sender address establish two roles,
        # without importing Bill To into the sender's neighbouring column.
        if ADDRESS.search(cell) and any(
                col != column and re.search(r"\bbill\s+to\b|\bclient\b", other, re.I)
                for col, other in cells[i]):
            return {"supplier"}
        # A named contact footer must repeat the issuer's visible heading.
        prefix, sep, address = cell.partition(":")
        if sep and re.match(r"\s*(?:IMM\b|rue\b|route\b|avenue\b|\d+\s)", address, re.I):
            if any(len(key(h)) >= 4 and key(prefix).startswith(key(h))
                   for j in range(1, min(4, len(lines) + 1)) if j in visible
                   for _, h in cells.get(j, []) if not NEW_LABEL.search(h)):
                return {"supplier"}
    sections = {p.get("section") for p in (metadata or {}).values()
                if line_id in p.get("line_ids", []) and p.get("section") in {"supplier", "customer", "ship_to"}}
    if sections:
        return sections
    from extraction_guards import corroborated_supplier_header
    masked = "\n".join(line if i in visible else "" for i, line in enumerate(lines, 1))
    heading = corroborated_supplier_header(masked)
    if heading and line_id <= heading["line_end"]:
        return {"supplier"}
    # Only an actual visible issuer header, not an isolated MF, supplies this role.
    visible_header = [lines[i - 1] for i in range(1, min(len(lines), 12) + 1) if i in visible]
    local = "\n".join(cell for _, cell in reversed(block))
    is_header = line_id <= 10 and 1 in visible and not any(
        ROLE.search(cell) for _, cell in cells.get(1, []))
    if is_header and re.search(r"\bfacture\b|\binvoice\b", "\n".join(visible_header), re.I):
        if ADDRESS.search(local) or (line_id <= 3 and re.search(r"[A-Za-z]{3}", local)
                                     and not re.search(r"\bMF\b|GSTIN|code\s*TVA", local, re.I)):
            return {"supplier"}
    # The separate legal footer carries address, telephone and registration,
    # after an article table; it is not assigned by subtracting the client ID.
    tail = []
    for direction in (-1, 1):
        for offset in range(0 if direction == -1 else 1, 4):
            i = line_id + direction * offset
            if i not in cells or TABLE.search(lines[i - 1]):
                break
            parts = [cell for col, cell in cells[i] if col == column]
            if any(ROLE.search(cell) for cell in parts):
                break
            tail.extend(parts)
    legal = "\n".join(tail)
    if (ADDRESS.search(legal) and re.search(r"\bt[\u00e9e]l\b", legal, re.I)
            and re.search(r"registre\s+de\s+commerce|registration\s+number", legal, re.I)
            and not ROLE.search(legal)
            and any(re.search(r"\bfacture\b|\binvoice\b", lines[i - 1], re.I)
                    for i in visible if i < line_id)):
        return {"supplier"}
    return set()


def party_role_supported(field, value, text, reference, allowed_lines=None, metadata=None):
    lines = text.splitlines()
    visible = set(range(1, len(lines) + 1)) if allowed_lines is None else set(allowed_lines)
    cited = cited_lines(text, reference, visible)
    cells = cells_by_line(lines, visible)
    roles = set()
    value_key = key(value)
    for i in cited:
        for column, cell in cells.get(i, []):
            unlabelled = cell.split(":", 1)[-1].strip()
            token = key(unlabelled)
            if value_key and (value_key in key(cell) or (len(token) >= 4 and value_key.startswith(token))):
                roles.update(cell_role(lines, i, column, visible, metadata))
    expected = "supplier" if field.startswith(("supplier_", "vendor_")) else "customer"
    return roles == {expected}
