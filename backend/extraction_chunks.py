"""Invoice-only structured passages. Coordinates never come from generated text."""
import hashlib
import math
import re
from collections import defaultdict

from assistant_language import normalize_text


def image_layout_frame(path):
    """Only unrotated image frames can be joined to current Paddle OCR safely."""
    from PIL import Image
    try:
        with Image.open(path) as image:
            if image.getexif().get(274, 1) != 1:
                return {}
            return {"coordinate_space": "exif_oriented_image", "image_width": image.width,
                    "image_height": image.height}
    except (OSError, ValueError):
        return {}


def document_key(text, document_id=None):
    return str(document_id) if document_id is not None else hashlib.sha256(text.encode()).hexdigest()


def text_lines(text):
    page, local = 1, 0
    result = []
    for index, line in enumerate(text.splitlines(), 1):
        marker = re.fullmatch(r"\s*---\s*Page\s+(\d+)\s*---\s*", line, re.I)
        if marker:
            page, local = int(marker[1]), 0
            continue
        local += 1
        if line.strip():
            result.append({"id": index, "page": page, "local": local, "text": line.strip()})
    return result


def rectangle(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        return None
    x, y, w, h = value
    return list(value) if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= 1.001 and y + h <= 1.001 else None


def union_box(boxes):
    if not boxes or any(b is None for b in boxes):
        return None
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    return [x0, y0, max(b[0] + b[2] for b in boxes) - x0,
            max(b[1] + b[3] for b in boxes) - y0]


def section_hint(text):
    value = normalize_text(text)
    if re.search(r"\b(?:designation|description)\b|البيان|وصف", value) and re.search(r"\b(?:qte|qty|quantity|quantite|prix|price|pu|unite)\b|كمية|سعر", value):
        return "line_items"
    tax_identifier = bool(re.search(r"\b(?:code|identifiant|matricule)\s+(?:t\.?\s*v\.?\s*a|fiscal)", value))
    if (not tax_identifier and re.search(r"\b(?:total|subtotal|montant\s+(?:h\.?t|t\.?t\.?c)|net\s+a\s+payer|timbre|t\.?v\.?a|vat)\b|المبلغ الجملي|دون احتساب|الطابع", value)):
        return "totals"
    if re.search(r"\b(?:payment|paiement|paye|especes|iban|rib|arretee)\b|طريقة الدفع|نقدا", value):
        return "payment_information"
    customer = bool(re.search(r"\b(?:client|customer|bill to)\b|الحريف|العميل", value))
    invoice = bool(re.search(r"\b(?:facture|invoice|echeance|date)\b|فاتورة|تاريخ", value))
    if customer and invoice:
        return "other"
    if customer and not re.search(r"signature|cachet", value):
        return "customer"
    if re.search(r"\b(?:supplier|fournisseur|emetteur)\b|المورد", value):
        return "supplier"
    if invoice:
        return "invoice_info"
    return None


def _ocr_records(text, metadata):
    source = defaultdict(list)
    for line in metadata.get("lines", []):
        source[(line.get("page_number"), line.get("line_number"))].append(line)
    records = text_lines(text)
    for record in records:
        matches = source[(record["page"], record["local"])]
        line = matches[0] if len(matches) == 1 and matches[0].get("text", "").strip() == record["text"] else {}
        cells = line.get("cells", [])
        boxes = [rectangle(c.get("box")) for c in cells]
        scores = [c.get("score") for c in cells]
        valid_scores = scores and all(isinstance(s, (int, float)) and math.isfinite(s) and 0 <= s <= 1 for s in scores)
        record.update(bbox=union_box(boxes), boxes=[b for b in boxes if b is not None],
                      confidence=sum(scores) / len(scores) if valid_scores else None,
                      layout_id=None, layout_label=None)
    return records


def _attach_layout(records, metadata, layout):
    width, height = metadata.get("image_width"), metadata.get("image_height")
    compatible = (metadata.get("coordinate_space") == "exif_oriented_image"
                  and layout.get("coordinate_space") == metadata.get("coordinate_space")
                  and width and height and width == layout.get("image_width")
                  and height == layout.get("image_height"))
    if not compatible:
        return False
    for record in records:
        if not record["bbox"]:
            continue
        x, y, w, h = record["bbox"]
        candidates = []
        for index, block in enumerate(layout.get("layout_boxes", [])):
            if block.get("page_number") != record["page"]:
                continue
            coord = block.get("coordinate")
            if not isinstance(coord, (list, tuple)) or len(coord) != 4:
                continue
            try:
                bx, by, ex, ey = [float(v) for v in coord]
                box = rectangle([bx / width, by / height, (ex - bx) / width, (ey - by) / height])
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if box is None:
                continue
            bx, by, bw, bh = box
            overlap = max(0, min(x + w, bx + bw) - max(x, bx)) * max(0, min(y + h, by + bh) - max(y, by)) / (w * h)
            if overlap >= .65:
                candidates.append((overlap, -(bw * bh), index, block.get("label")))
        if candidates:
            _, _, index, label = max(candidates)
            record.update(layout_id=f"P{record['page']}:layout:{index}", layout_label=label)
    return True


def structured_passages(text, ocr_metadata, layout_metadata=None, document_id=None, max_chars=900):
    """Group source lines; split oversized blocks only at line boundaries."""
    document_id = document_key(text, document_id)
    records = _ocr_records(text, ocr_metadata)
    frame_joined = _attach_layout(records, ocr_metadata, layout_metadata or {})
    blocks, current = [], []
    current_section, current_key = None, None
    for record in records:
        hint = section_hint(record["text"])
        key = (record["page"], record["layout_id"])
        same = current and key == current_key
        # Logical amounts within a layout block stay together, including value-only rows.
        section = hint or (current_section if same else None)
        if not section:
            section = "footer" if record["layout_label"] == "footer" else "header" if record["layout_label"] in {"header", "paragraph_title", "doc_title"} else "other"
        if current_section == "line_items" and hint is None and current and record["page"] == current[-1]["page"] and record["layout_label"] == "table":
            section = "line_items"
        same_amount_block = (same and record["layout_id"] is not None
                             and current_section in {"totals", "payment_information"}
                             and section in {"totals", "payment_information"})
        if same_amount_block and "totals" in {section, current_section}:
            section = "totals"
            current_section = "totals"
        continues_table = (current and current_section == section == "line_items"
                           and record["page"] == current[-1]["page"] and hint is None
                           and record["layout_label"] == "table")
        if current and ((key != current_key and not continues_table)
                        or (hint is not None and (section != current_section or hint == "other"))):
            blocks.append((current, current_section))
            current = []
        current.append(record)
        current_section, current_key = section, key
    if current:
        blocks.append((current, current_section))
    passages = []
    for block_index, (block, section) in enumerate(blocks):
        parent = f"{document_id}:P{block[0]['page']}:B{block_index}"
        header = [r for r in block if section_hint(r["text"]) == "line_items"][:1] if section == "line_items" else []
        parts, pending = [], []
        for record in block:
            if pending and pending != header and sum(len(r["text"]) + 1 for r in pending) + len(record["text"]) > max_chars:
                parts.append(pending)
                pending = list(header) if record not in header else []
            if record not in pending:
                pending.append(record)
        if pending:
            parts.append(pending)
        for part in parts:
            part = sorted(part, key=lambda r: r["id"])
            line_ids = [r["id"] for r in part]
            payload = "\n".join(r["text"] for r in part)
            digest = hashlib.sha256((parent + str(line_ids) + payload).encode()).hexdigest()[:20]
            chunk_id = f"C{digest}"
            labels = list(dict.fromkeys(r["layout_label"] for r in part if r["layout_label"]))
            scores = [r["confidence"] for r in part]
            passages.append({"chunk_id": chunk_id, "source_id": chunk_id, "document_id": document_id,
                "page": part[0]["page"], "page_number": part[0]["page"], "section": section,
                "text": payload, "bbox": union_box([r["bbox"] for r in part]),
                "boxes": [b for r in part for b in r["boxes"]], "line_ids": line_ids,
                "page_line_ids": [r["local"] for r in part],
                "ocr_confidence": sum(scores) / len(scores) if scores and all(s is not None for s in scores) else None,
                "source_type": "ocr", "layout_label": labels[0] if len(labels) == 1 else None,
                "layout_labels": labels, "parent_block_id": parent,
                "layout_joined": frame_joined, "header_line_ids": [r["id"] for r in header],
                "coordinate_space": ocr_metadata.get("coordinate_space"), "bbox_format": "normalized_xywh",
                "line_start": part[0]["local"], "line_end": part[-1]["local"],
                "global_line_start": line_ids[0], "global_line_end": line_ids[-1],
                "kind": "table" if section == "line_items" else "text", "chunk_index": len(passages),
                "chunking_mode": "layout"})
    return passages


class RetrievedEvidence(dict):
    """Server-created source catalogue, not reconstructed from client-supplied offsets."""
    def __init__(self, values, passages):
        super().__init__(values)
        self.passages = passages
