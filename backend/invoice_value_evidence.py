"""Source-bound invoice values, independent of OCR and retrieval engines."""
import json
import re
from decimal import Decimal, InvalidOperation
from party_evidence import NEW_LABEL, POSTAL, cells_by_line


NUMBER = r"[0-9]+(?:[ \u00a0\u202f][0-9]{3})*(?:[.,][0-9]+)?"


def decimal_value(value):
    token = str(value).strip()
    if not re.fullmatch(NUMBER, token):
        return None
    try:
        return Decimal(re.sub(r"[ \u00a0\u202f]", "", token).replace(",", "."))
    except InvalidOperation:
        return None


def stamp_label_value_pairs(text, allowed_lines=None):
    """A stamp label and one adjacent amount; no intervening row or guessed tax."""
    lines = text.splitlines()
    visible = set(range(1, len(lines) + 1)) if allowed_lines is None else set(allowed_lines)
    found = []
    for index in range(len(lines) - 1):
        if not {index + 1, index + 2} <= visible:
            continue
        if not re.fullmatch(r"\s*(?:timbre(?:\s+fiscal)?|stamp(?:\s+duty)?)[\s:|=().%]*", lines[index], re.I):
            continue
        amount = decimal_value(lines[index + 1])
        if amount is not None:
            found.append({"value": amount, "line_start": index + 1, "line_end": index + 2})
    return found


def ambiguous_monetary_row(field, value, text, reference, allowed_lines=None):
    """Flag an orphan amount following a cited labelled amount, never choose one."""
    from extraction_guards import monetary_in_source
    from party_evidence import cited_lines
    lines = text.splitlines()
    visible = set(range(1, len(lines) + 1)) if allowed_lines is None else set(allowed_lines)
    for line_id in sorted(cited_lines(text, reference, visible)):
        if line_id + 1 not in visible or line_id >= len(lines):
            continue
        if not monetary_in_source(field, value, lines[line_id - 1]):
            continue
        cells = [cell.strip() for cell in lines[line_id].split("|")]
        candidate = decimal_value(cells[0])
        if (candidate is not None and re.search(r"[.,]\d{2,3}$", cells[0])
                and candidate != decimal_value(value) and not any(NEW_LABEL.search(cell) for cell in cells[1:])):
            return {"reason": "ambiguous_adjacent_monetary_cells", "line_ids": [line_id, line_id + 1],
                    "source": "\n".join(lines[line_id - 1:line_id + 1])}
    return None


def explicit_customer_name(text, allowed_lines=None):
    """Copy a company cell in an explicit customer-code block, not any company."""
    from party_evidence import ROLE
    lines = text.splitlines()
    visible = set(range(1, len(lines) + 1)) if allowed_lines is None else set(allowed_lines)
    candidates = []
    for index, line in enumerate(lines[:-1]):
        if not {index + 1, index + 2} <= visible:
            continue
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) != 2 or not re.fullmatch(r"(?:code\s+client|customer\s+code)\s*:?", cells[0], re.I):
            continue
        if ROLE.search(cells[1]) or NEW_LABEL.search(cells[1]):
            continue
        if not re.search(r"\b(?:STE|SOCI[E\u00c9]T[E\u00c9]|SARL|SUARL|EURL|LTD)\b", cells[1], re.I):
            continue
        code = lines[index + 1].split("|")[0].strip()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9./-]{3,}", code, re.I) or not re.search(r"\d", code):
            continue
        candidates.append({"value": cells[1], "line_start": index + 1, "line_end": index + 2})
    return candidates[0] if len(candidates) == 1 else None


def discount_evidence(text):
    """Read a discount amount from its own cell, never the next row's ID."""
    lines = (text or "").splitlines()
    found = []
    for index, line in enumerate(lines):
        cells = [cell.strip() for cell in line.split("|")]
        columns = [i for i, cell in enumerate(cells)
                   if re.fullmatch(r"(?:remise|discount|rabais)(?:\s+\d+(?:[.,]\d+)?\s*%)?", cell, re.I)]
        if len(cells) >= 3 and len(columns) == 1 and re.search(
                r"designation|d\u00e9signation|description|article|quantit|quantity|prix|unit\s*price|MT\s*HT", line, re.I):
            column = columns[0]
            rate_match = re.search(r"(\d+(?:[.,]\d+)?)\s*%", cells[column])
            rate = float(decimal_value(rate_match[1])) if rate_match else None
            for row_index in range(index + 1, len(lines)):
                row = [cell.strip() for cell in lines[row_index].split("|")]
                if len(row) == 1 and not re.search(r"total|montant|taxe|TVA|FODEC", row[0], re.I):
                    continue  # Wrapped description, not an amount cell.
                if len(row) != len(cells):
                    break
                amount = decimal_value(row[column])
                if amount is None:
                    break
                found.append({"amount": float(amount), "rate": rate,
                              "source": line + "\n" + lines[row_index],
                              "line_ids": [index + 1, row_index + 1],
                              "row_text": lines[row_index], "scope": "item"})
            continue
        # Horizontal whitespace is intentional: no match can cross a line.
        pattern = (r"\b(?:remise|discount|rabais)[ \t:|=]*"
                   r"(?:\(?[ \t]*(\d+(?:[.,]\d+)?)[ \t]*%[ \t]*\)?[ \t:|=]*)?"
                   r"(?:\(-\)[ \t]*)?"
                   r"(" + NUMBER + r")(?![\d.,]|[ \t]*%)")
        if not re.search(r"designation|d\u00e9signation|description|quantit|quantity|prix\s+unitaire", line, re.I):
            match = re.search(pattern, line, re.I)
            if match:
                found.append({"amount": float(decimal_value(match[2])),
                              "rate": float(decimal_value(match[1])) if match[1] else None,
                              "source": line, "line_ids": [index + 1], "scope": "invoice"})
    return found


def invoice_discount(text):
    entries = discount_evidence(text)
    candidates = [entry for entry in entries if entry["scope"] == "invoice"] or entries
    # Multiple article discounts are not a printed invoice discount total.
    return candidates[0] if len(candidates) == 1 else None


def item_discount(row, text=None):
    entries = discount_evidence(row.get("source", ""))
    if not entries and text:
        quoted_lines = row.get("source", "").splitlines()
        entries = [entry for entry in discount_evidence(text) if entry.get("row_text") in quoted_lines]
    return entries[0] if len(entries) == 1 else None


def printed_tax_rows(text):
    """Only labelled taxes with an explicit rate and amount are interpreted."""
    if not isinstance(text, str):
        return []
    rows = []
    pattern = (r"\b(FODEC|TVA|VAT)\b[ \t|:]*\(?[ \t]*"
               r"(\d+(?:[.,]\d+)?)[ \t]*%[ \t]*\)?[ \t|:]*(" + NUMBER + r")(?![\d.,])")
    for index, line in enumerate((text or "").splitlines(), 1):
        for match in re.finditer(pattern, line, re.I):
            rows.append({"label": match[1].upper(), "rate": float(decimal_value(match[2])),
                         "amount": float(decimal_value(match[3])), "base_amount": None,
                         "source": line, "line_ids": [index]})
    return rows


def supported_tax(row, source):
    candidates = [candidate for candidate in printed_tax_rows(source)
                  if decimal_value(row.get("rate")) == decimal_value(candidate["rate"])
                  and decimal_value(row.get("amount")) == decimal_value(candidate["amount"])]
    if len(candidates) != 1:
        return None
    candidate = candidates[0]
    # Never infer the taxable base from a rate or other invoice totals.
    base = row.get("base_amount")
    if base is not None:
        match = re.search(r"\bbase(?:\s+(?:HT|imposable))?[ \t:|=]*(" + NUMBER + r")", source, re.I)
        if not match or decimal_value(base) != decimal_value(match[1]):
            return None
        candidate["base_amount"] = float(decimal_value(base))
    return {**candidate, "source": source if base is not None else candidate["source"]}


def additional_tax_total(rows, proof=None):
    """Count grounded FODEC once. TVA is already the separate tax_amount."""
    total = Decimal("0")
    seen = set()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        source = row.get("source", "")
        checked = supported_tax(row, source) if source else None
        if checked is None or not proof or not proof.get("supported"):
            continue
        key = (checked["label"], checked["rate"], checked["amount"], source)
        if key in seen:
            continue
        seen.add(key)
        if checked["label"] == "FODEC":
            total += decimal_value(checked["amount"])
    return total


PARTY_LABEL = NEW_LABEL


def _space_key(value):
    return re.sub(r"\s+", "", str(value)).casefold()


def party_block_candidates(field, quote):
    """Assemble only the column anchored by the requested business label."""
    if field.endswith("address"):
        anchor = re.compile(r"(?:adresse|address)\s*:\s*(.+)", re.I)
    elif field == "customer_name":
        anchor = re.compile(r"(?:raison\s*sociale?|nom\s+du\s+client|client|customer|bill\s+to)\s*:\s*(.+)", re.I)
    else:
        return []
    lines = quote.splitlines()
    cells_index = cells_by_line(lines, set(range(1, len(lines) + 1)))
    values = []
    for index, line in enumerate(lines):
        for column, cell in enumerate(line.split("|")):
            match = anchor.search(cell)
            if not match:
                continue
            pieces = [match[1].strip()]
            end = index + 1
            for next_index, next_line in enumerate(lines[index + 1:index + 3], index + 1):
                cells = cells_index.get(next_index + 1, [])
                matching = [part for col, part in cells if col == column]
                if not matching or POSTAL.search(pieces[-1]):
                    break
                if len(line.split("|")) > 1 and len(next_line.split("|")) == 1 and not POSTAL.search(next_line):
                    break  # Lost column position is not a continuation proof.
                continuation = matching[0]
                if not continuation or PARTY_LABEL.search(continuation) or ":" in continuation:
                    break
                if not re.match(r"[^\W\d_]", continuation, re.UNICODE):
                    break
                # Neighbouring headers cannot become part of this cell.
                if field == "customer_name" and re.search(r"\d", continuation):
                    break
                pieces.append(continuation)
                end = next_index + 1
                if any(PARTY_LABEL.search(other) for col, other in cells if col != column):
                    break
            values.append({"value": " ".join(pieces), "line_start": index + 1, "line_end": end})
    return values


def party_block_values(field, quote):
    return [candidate["value"] for candidate in party_block_candidates(field, quote)]


def customer_code_candidates(text, allowed_lines):
    lines = text.splitlines()
    entries = []
    for index, line in enumerate(lines):
        if index + 1 not in allowed_lines:
            continue
        match = re.fullmatch(r"\s*(?:code\s+client|customer\s+id)\s*:?[ \t]*(.*)", line, re.I)
        if not match:
            continue
        for next_index in range(index, min(index + 3, len(lines))):
            if not set(range(index + 1, next_index + 2)).issubset(allowed_lines):
                break
            token = match[1].strip() if next_index == index else lines[next_index].strip()
            if re.fullmatch(r"[A-Z0-9/-]+", token, re.I) and any(c.isdigit() for c in token):
                entries.append({"value": token, "line_start": index + 1, "line_end": next_index + 1})
                break
    return entries


def customer_code_supported(value, text, reference, allowed_lines):
    match = re.fullmatch(r"L(\d+)(?:-L?(\d+))?", reference or "") if isinstance(reference, str) else None
    if not match:
        return False
    first, last = int(match[1]), int(match[2] or match[1])
    visible = allowed_lines if allowed_lines is not None else set(range(1, len(text.splitlines()) + 1))
    entries = [entry for entry in customer_code_candidates(text, visible)
               if first <= entry["line_end"] <= last]
    return len(entries) == 1 and _space_key(entries[0]["value"]) == _space_key(value)


def address_in_validated_party_block(field, tax_id, text, allowed_lines):
    """A labelled address in the same short block as an already-grounded fiscal ID."""
    from identifier_evidence import identifier_spans
    if not tax_id:
        return None
    lines = text.splitlines()
    visible = allowed_lines if allowed_lines is not None else set(range(1, len(lines) + 1))
    masked = "\n".join(line if index in visible else "" for index, line in enumerate(lines, 1))
    role = "supplier" if field == "supplier_address" else "customer"
    fiscal_lines = [index for index, line in enumerate(lines, 1) if index in visible
                    and identifier_spans(role + "_tax_id", tax_id, line)]
    candidates = []
    for candidate in party_block_candidates(field, masked):
        for fiscal_line in fiscal_lines:
            if not 0 <= fiscal_line - candidate["line_end"] <= 3:
                continue
            span = set(range(candidate["line_start"], fiscal_line + 1))
            if not span.issubset(visible):
                continue
            between = "\n".join(lines[candidate["line_end"]:fiscal_line])
            opposite = r"\b(?:client|customer|chantier|bill\s+to)\b" if role == "supplier" else r"\b(?:fournisseur|supplier|seller)\b"
            if re.search(opposite + r"|\b(?:designation|total|montant|facture)\b", between, re.I):
                continue
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates[0] if len(candidates) == 1 else None


def party_cell_hints(fields, text, allowed_lines):
    """Expose exact cells already in the context; no extra retrieval or OCR."""
    if not {"supplier_address", "customer_address", "customer_name", "customer_id"}.intersection(fields):
        return ""
    visible = "\n".join(line if index in allowed_lines else "" for index, line in enumerate(text.splitlines(), 1))
    hints = {}
    for field in ("customer_name", "address"):
        if field == "customer_name" and field not in fields:
            continue
        entries = party_block_candidates(field, visible)
        if entries:
            hints[field] = [{"value": entry["value"], "source": f"L{entry['line_start']}-L{entry['line_end']}"} for entry in entries]
    if "customer_id" in fields:
        entries = [{"value": entry["value"], "source": f"L{entry['line_start']}-L{entry['line_end']}"}
                   for entry in customer_code_candidates(text, allowed_lines)]
        if entries:
            hints["customer_id"] = entries
    if not hints:
        return ""
    return ("Cellules candidates recopiees des MEMES preuves visibles (verifie leur role; address ne presume "
            "pas fournisseur/client). Les voisins DATE, BCN et CHANTIER ne sont pas des valeurs de ces cellules. "
            "Le nom peut etre espace entre mots sans changer ses lettres : "
            + json.dumps(hints, ensure_ascii=False, separators=(",", ":")) + "\n")


def party_value_supported(field, value, quote):
    candidates = party_block_values(field, quote)
    if candidates:
        return len(candidates) == 1 and _space_key(value) == _space_key(candidates[0])
    return bool(_space_key(value) and _space_key(value) in _space_key(quote))


def party_citation_complete(field, value, text, reference, allowed_lines):
    """Detect a truncated citation when an available next cell continues it."""
    match = re.fullmatch(r"L(\d+)(?:-L?(\d+))?", reference or "") if isinstance(reference, str) else None
    if not match:
        return True
    first, last = int(match[1]), int(match[2] or match[1])
    lines = text.splitlines()
    end = last
    while end < min(len(lines), last + 2) and (allowed_lines is None or end + 1 in allowed_lines):
        end += 1
    candidates = party_block_values(field, "\n".join(lines[first - 1:end]))
    if candidates and not any(_space_key(value) == _space_key(candidate) for candidate in candidates):
        return False
    if field.endswith("address"):
        visible = set(range(1, len(lines) + 1)) if allowed_lines is None else set(allowed_lines)
        cells = cells_by_line(lines, visible)
        # A city-only citation is incomplete even without an Address: label.
        # Match the printed street, not an inferred or calculated address.
        for i in range(max(1, first - 2), last + 1):
            for column, cell in cells.get(i, []):
                if not re.match(r"\d+[A-Za-z]?\s+[^|\r\n]*[^\W\d_]", cell, re.UNICODE):
                    continue
                parts, stop = [cell], i
                for j in range(i + 1, min(len(lines), i + 2) + 1):
                    following = [c for col, c in cells.get(j, []) if col == column]
                    if not following or not POSTAL.search(following[0]):
                        break
                    parts.append(following[0])
                    stop = j
                complete = _space_key(" ".join(parts))
                if first <= stop and i <= last and _space_key(value) in complete:
                    if _space_key(value) != complete or first > i or last < stop:
                        return False
    return True


def structured_party_instruction(fields):
    instructions = []
    if {"supplier_address", "customer_address", "customer_name"}.intersection(fields):
        instructions.append(
            "IDENTITE MULTILIGNE : recopie l'adresse complete, y compris sa continuation dans la meme colonne. "
            "Le pied de page peut porter l'adresse du fournisseur apres un separateur |; ignore la cellule voisine. "
            "Pour customer_name, assemble les cellules RAISON SOCIALE et leur continuation, jamais DATE, BCN "
            "ou une date. Tu peux restituer les espaces entre mots colles sans changer les lettres. "
            "Les valeurs ne contiennent pas leurs libelles (Adresse:, Raison sociale:). "
            "Cite TOUTES les lignes utilisees (Lx-Ly); n'accepte pas un debut de nom ou d'adresse seul.")
    if {"supplier_registration_number", "customer_id"}.intersection(fields):
        instructions.append("Le registre de commerce fournisseur et le code client sont distincts des matricules fiscaux. "
                            "Recopie leurs identifiants complets avec la preuve de leur libelle et valeur. "
                            "Le CODE CLIENT peut etre sur une ligne et sa valeur numerique sur une ligne suivante; "
                            "ne le confonds pas avec le Code TVA intercale.")
    if "taxes" in fields:
        instructions.append("taxes contient les lignes de taxes imprimees: rate est le pourcentage, amount le montant, "
                            "base_amount reste null si absent, source cite la ligne exacte. Inclure FODEC et TVA "
                            "separement, jamais Total Taxes ni timbre. tax_amount reste TVA seule.")
    return "\n".join(instructions) + ("\n" if instructions else "")
