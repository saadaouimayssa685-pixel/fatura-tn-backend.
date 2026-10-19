"""Validate identifier copies, never repair or complete OCR identifiers."""
import re

from extraction_guards import complete_tax_id
from party_evidence import ROLE, cell_role, cited_lines


IDENTIFIER_FIELDS = frozenset({"supplier_tax_id", "customer_tax_id", "invoice_number", "rib"})
_JOINERS = "/-_."
_PARTY = re.compile(
    r"\b(?P<supplier>fournisseur|supplier|vendor|seller|emetteur)\b|"
    r"\b(?P<customer>(?:nom\s+du\s+)?client|customer|bill\s+to|sold\s+to|destinataire)\b", re.I)
_TABLE = re.compile(r"\b(?:d[ée]signation|description|quantity|prix\s+unitaire)\b|---\s*Page", re.I)
_TAX = re.compile(r"\b(?:M\.?F\.?|matricule\s+fiscal|identifiant\s+fiscal|code\s*T\.?\s*V\.?\s*A\.?)\s*:?", re.I)


def identifier_instruction(fields):
    identifiers = sorted(IDENTIFIER_FIELDS.intersection(fields))
    if not identifiers:
        return ""
    instruction = (
        f"CONTROLE DES IDENTIFIANTS ({', '.join(identifiers)}) : copie la valeur COMPLETE "
        "telle qu'imprimee dans la ligne citee, comme une chaine JSON, jamais un nombre. "
        "Preserve lettres, zeros initiaux, /, - et TOUS les suffixes. Ne garde pas seulement "
        "la partie numerique. Relis jusqu'a la fin de la cellule avant de repondre ; "
        "ne transforme pas un separateur en un autre et n'ajoute aucun caractere absent. "
        "Verifie le bloc proprietaire : supplier_tax_id appartient au fournisseur, "
        "customer_tax_id au client. Cite les lignes de ce bloc, pas un autre matricule "
        "du document. Si la copie complete ou son attribution reste ambigue, retourne null.\n"
    )
    if "supplier_tax_id" in identifiers:
        instruction += (
            "Pour supplier_tax_id uniquement : les libelles Code TVA, Code T.V.A, MF, "
            "matricule fiscal ou VAT ID peuvent designer l'identifiant fiscal de l'emetteur. "
            "Examine aussi son bloc de coordonnees legales (adresse, telephone, registre de commerce), "
            "pas seulement son nom en en-tete. Un Code TVA explicitement present dans ce bloc "
            "doit etre copie integralement avec sa citation. L'extraction de sa graphie OCR "
            "n'est pas une certification de son format administratif : ne retourne pas null "
            "au seul motif que le code est compact, contient des espaces ou n'a pas de barres obliques. "
            "N'ajoute ni chiffre ni separateur absent. Le pied de page seul ne prouve pas le role : "
            "verifie le contexte de l'emetteur. Ne deduis ni la valeur ni son proprietaire du "
            "matricule client. Si la mention fiscale manque ou si son attribution reste ambigue, "
            "conserve null.\n"
        )
    return instruction


def identifier_spans(field, value, source):
    """Return only complete occurrences; OCR spacing may differ, punctuation may not."""
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        return []
    expected = re.sub(r"\s+", "", value)
    if not re.fullmatch(r"[\w/.-]+", expected) or not any(c.isdigit() for c in expected):
        return []
    if field.endswith("tax_id") and not complete_tax_id(value, source):
        return []
    positions = [i for i, char in enumerate(source) if not char.isspace()]
    compact = "".join(source[i] for i in positions)
    spans = []
    for match in re.finditer(re.escape(expected), compact):
        start, end = positions[match.start()], positions[match.end() - 1] + 1
        if "\n" in source[start:end] or "\r" in source[start:end]:
            continue
        before, after = source[:start], source[end:]
        # A value inside an adjacent token, or before/after a separated suffix,
        # is not a full identifier even if its digits occur in the evidence.
        if (before and (before[-1].isalnum() or before[-1] == "_")) or (
                after and (after[0].isalnum() or after[0] == "_")):
            continue
        # -MF starts a new labelled cell, not a registration-number suffix.
        next_label = field in {"supplier_registration_number", "vendor_registration_number"} and re.match(
            r"\s*-\s*M\.?F\.?\s*:", after, re.I)
        if re.search(r"[\w][/-]\s*$", before) or (re.match(r"\s*[/-]\s*\w", after) and not next_label):
            continue
        if before and before[-1] in _JOINERS and len(before) > 1 and before[-2].isalnum():
            continue
        if after and after[0] in _JOINERS and len(after) > 1 and after[1].isalnum() and not next_label:
            continue
        if field == "rib" or field.endswith("tax_id"):
            # Spaced bank account groups and numeric fiscal suffixes remain significant.
            if re.search(r"\d[^\S\r\n]+$", before) or re.match(r"[^\S\r\n]+\d", after):
                continue
            if re.match(r"[^\S\r\n]+[A-Za-z]{1,3}\s*[/\d]", after):
                continue
        spans.append((start, end))
    return spans


def _line_roles(lines, line_id, column, visible, metadata):
    """The character offset identifies a cell, not the entire preceding row."""
    # A collapsed fiscal row loses its column index. Only an immediately visible
    # party label opposite invoice metadata can recover ownership unambiguously.
    current = lines[line_id - 1]
    if "|" not in current and _TAX.match(current.strip()) and line_id - 1 in visible:
        previous = [cell.strip() for cell in lines[line_id - 2].split("|")]
        roles = {match.lastgroup for cell in previous for match in ROLE.finditer(cell)}
        if len(previous) > 1 and roles:
            nonparty = [cell for cell in previous if not ROLE.search(cell)]
            explicit_party = all(re.match(
                r"^(?:(?:nom\s+du\s+)?client|customer|bill\s+to|sold\s+to|ship\s+to|"
                r"fournisseur|supplier|vendor|seller|[e\u00e9]metteur)\s*[:\-]", cell, re.I)
                for cell in previous if ROLE.search(cell))
            if len(roles) > 1 or (explicit_party and nonparty and all(
                    re.match(r"^(?:facture|invoice)\b", cell, re.I) for cell in nonparty)):
                sections = {p.get("section") for p in (metadata or {}).values()
                            if line_id in p.get("line_ids", []) and p.get("section") in {"supplier", "customer", "ship_to"}}
                return roles | sections
    return cell_role(lines, line_id, lines[line_id - 1][:column].count("|"), visible, metadata)


def identifier_role_supported(field, value, text, reference, allowed_lines=None, metadata=None):
    if not field.endswith("tax_id"):
        return True
    lines = text.splitlines()
    visible = set(range(1, len(lines) + 1)) if allowed_lines is None else set(allowed_lines)
    cited = cited_lines(text, reference, visible)
    expected = "supplier" if field == "supplier_tax_id" else "customer"
    roles = set()
    for i in sorted(cited & visible):
        for start, _ in identifier_spans(field, value, lines[i - 1]):
            roles.update(_line_roles(lines, i, start, visible, metadata))
    return roles == {expected}
