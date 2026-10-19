"""Locate unambiguous field values using word coordinates, never guessed boxes."""
from pathlib import Path
import re
import unicodedata
from decimal import Decimal, InvalidOperation

import pytesseract
from PIL import Image


MONEY_FIELDS = {"subtotal", "tax_amount", "stamp_duty", "total_amount", "discount", "net_payable"}


def canonical(value):
    value = unicodedata.normalize("NFKD", str(value)).casefold()
    return "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in value
                   if c.isalnum())


def value_matches(key, value, text):
    if key in MONEY_FIELDS:
        cleaned = text.strip().replace("\u00a0", " ")
        if not re.fullmatch(r"[+-]?\d+(?: \d{3})*(?:[.,]\d{1,3})?", cleaned):
            return False
        try:
            return Decimal(cleaned.replace(" ", "").replace(",", ".")) == Decimal(str(value).replace(",", "."))
        except InvalidOperation:
            return False
    expected, observed = canonical(value), canonical(text)
    if expected == observed and expected:
        return True
    # OCR may attach a label to the invoice identifier within a single word.
    if key == "invoice_number":
        return bool(expected and re.fullmatch(r"(?:n|no|numero|facturen|factureno)" + re.escape(expected), observed))
    return False


def minimal_word_matches(line, key, value):
    words = line.get("words") or []
    candidates = []
    for start in range(len(words)):
        for end in range(start + 1, min(len(words), start + 35) + 1):
            text = " ".join(word["text"] for word in words[start:end])
            if value_matches(key, value, text):
                candidates.append((start, end))
                break
    return [words[start:end] for start, end in candidates
            if not any(start <= a and b <= end and (a, b) != (start, end) for a, b in candidates)]


def match_field_lines(lines, fields, supports):
    result = {}
    for key, value in fields.items():
        if value is None or value == "" or isinstance(value, (list, dict)):
            continue
        matches = []
        for index, line in enumerate(lines):
            if not line.get("words"):
                if supports(key, value, line["text"]):
                    matches.append(line)
                continue
            for words in minimal_word_matches(line, key, value):
                matches.append({**line, "boxes": [word["box"] for word in words],
                                "matched_text": " ".join(word["text"] for word in words),
                                "context": [{"line_number": other["line_number"], "text": other["text"]}
                                            for other in lines[max(0, index - 2):index + 3]
                                            if other.get("page_number") == line.get("page_number")]})
        if len(matches) == 1:
            result[key] = {"method": "ocr_word_coordinates", **matches[0]}
    return result


def locate_paddle_fields(lines, fields, proofs=None):
    """Highlight actual OCR cells; ambiguous repeated values have no guessed box."""
    result = {}
    for key, value in fields.items():
        if value is None or value == "" or isinstance(value, (list, dict)):
            continue
        matches = []
        source = (proofs or {}).get(key, {}).get("source")
        source = source if isinstance(source, dict) else {}
        for line in lines:
            if source.get("page_line_ids") is not None and (
                line["page_number"] != source.get("page_number", 1)
                or line["line_number"] not in source["page_line_ids"]
            ):
                continue
            if source.get("line_start") is not None and not (
                line["page_number"] == source.get("page_number", 1)
                and source["line_start"] <= line["line_number"] <= source.get("line_end", source["line_start"])
            ):
                continue
            for cell in line.get("cells", []):
                candidate = re.sub(r"^\s*(?:adresse|address|telephone|tel|email|e-mail|mf|code\s+t\.?v\.?a\.?|date)\s*[:=]\s*", "", cell["text"], flags=re.I)
                contained = False
                if key in {"vendor_name", "customer_name", "vendor_address", "customer_address", "vendor_tax_id", "customer_tax_id", "vendor_email", "customer_email"}:
                    expected = canonical(value)
                    contained = len(expected) >= 6 and expected in canonical(candidate)
                if value_matches(key, value, candidate) or contained:
                    matches.append((line, cell))
        if len(matches) > 1:
            exact = [(line, cell) for line, cell in matches if value_matches(key, value, cell["text"])]
            if len(exact) == 1:
                matches = exact
            elif key in {"vendor_tax_id", "customer_tax_id"}:
                labelled = [(line, cell) for line, cell in matches if re.search(r'\bMF\s*:', cell["text"], re.I)]
                if len(labelled) == 1:
                    matches = labelled
        if len(matches) == 1:
            line, cell = matches[0]
            result[key] = {"method": "paddle_ocr_cell_coordinates", "page_number": line["page_number"],
                           "line_number": line["line_number"], "text": line["text"],
                           "matched_text": cell["text"], "boxes": [cell["box"]], "box": cell["box"]}
    return result


def image_lines(image, config, page_number=1):
    width, height = image.size
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT, timeout=30)
    groups = {}
    for i, word in enumerate(data["text"]):
        if not word.strip() or float(data["conf"][i]) < 0:
            continue
        key = tuple(data[k][i] for k in ("block_num", "par_num", "line_num"))
        groups.setdefault(key, []).append(i)
    lines = []
    for indices in groups.values():
        left = min(data["left"][i] for i in indices)
        top = min(data["top"][i] for i in indices)
        right = max(data["left"][i] + data["width"][i] for i in indices)
        bottom = max(data["top"][i] + data["height"][i] for i in indices)
        lines.append({"text": " ".join(data["text"][i] for i in indices), "page_number": page_number,
                      "line_number": len(lines) + 1,
                      "box": [left / width, top / height, (right - left) / width, (bottom - top) / height],
                      "words": [{"text": data["text"][i], "box": [data["left"][i] / width, data["top"][i] / height,
                                 data["width"][i] / width, data["height"][i] / height]} for i in indices]})
    return lines


def locate_image_fields(path, fields, config, supports):
    extension = Path(path).suffix.lower()
    if extension == ".pdf":
        import pdfplumber
        lines = []
        with pdfplumber.open(path) as pdf:
            # Never claim uniqueness if only part of the document was examined.
            if len(pdf.pages) > 10:
                return {}
            for page_number, page in enumerate(pdf.pages, 1):
                # Render the crop box so coordinates match the browser PDF viewport.
                with page.to_image(resolution=120).original as image:
                    page_lines = image_lines(image, config, page_number)
                for line in page_lines:
                    line["method"] = "pdf_page_ocr_coordinates"
                lines.extend(page_lines)
        return match_field_lines(lines, fields, supports)
    if extension not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return {}
    with Image.open(path) as image:
        lines = image_lines(image, config)
    return match_field_lines(lines, fields, supports)
