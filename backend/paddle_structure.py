"""Conservative geometry over Paddle cells; never rewrite recognized characters."""
from copy import deepcopy
import re
from statistics import median
import unicodedata


def _label(text):
    value = unicodedata.normalize("NFKD", text).lower()
    return re.sub(r"[^a-z0-9]", "", value)


def _center(cell):
    x, y, w, h = cell["box"]
    return x + w / 2, y + h / 2


def _review(cell, reason):
    cell["requires_review"] = True
    reasons = cell.setdefault("review_reasons", [])
    if reason not in reasons:
        reasons.append(reason)


def _group(cells, slope=0.0, columns=False):
    rows = []
    def ordinate(c):
        x, y = _center(c)
        return y - slope * x
    for cell in sorted(cells, key=lambda c: (ordinate(c), c["box"][0])):
        y = ordinate(cell)
        candidates = [row for row in rows[-5:] if
                      abs(y - row["center"]) <= .45 * min(cell["box"][3], row["height"])]
        if columns:
            for row in candidates:
                for other in row["cells"]:
                    if other["column_index"] == cell["column_index"]:
                        _review(cell, "overlapping_cells_same_column")
                        _review(other, "overlapping_cells_same_column")
            candidates = [row for row in candidates if
                          all(c["column_index"] != cell["column_index"] for c in row["cells"])]
        if len(candidates) > 1:
            _review(cell, "ambiguous_row")
        match = candidates[-1] if len(candidates) == 1 else None
        if match is None:
            rows.append({"center": y, "height": cell["box"][3], "cells": [cell]})
        else:
            match["cells"].append(cell)
    return rows


def _header(row):
    labels = [_label(c["text"]) for c in row["cells"]]
    return (bool(set(labels) & {"designation", "description", "items"})
            and bool(set(labels) & {"qte", "quantity", "quantite", "qty"})
            and bool(set(labels) & {"price", "prixunit", "prixunitaire", "puht", "puttc"}))


def _table_end(row):
    return any(re.match(r"^(?:total|subtotal|montantht|montantttc|droits|taxe|tva|timbre|terms|cachet)",
                        _label(c["text"])) for c in row["cells"])


def _table_columns(header):
    cells = sorted(header["cells"], key=lambda c: c["box"][0])
    boundaries = [0.0] + [(a["box"][0] + a["box"][2] + b["box"][0]) / 2
                          for a, b in zip(cells, cells[1:])] + [1.0]
    return cells, boundaries


def _assign_column(cell, boundaries):
    x, _, w, _ = cell["box"]
    candidates = [i for i, (left, right) in enumerate(zip(boundaries, boundaries[1:]))
                  if max(0, min(x + w, right) - max(x, left)) > .15 * w]
    cell["column_candidates"] = candidates
    cell["column_index"] = candidates[0] if len(candidates) == 1 else None
    if len(candidates) != 1:
        _review(cell, "spans_columns" if candidates else "unassigned_column")


def _header_slope(cells):
    points = [_center(c) for c in cells]
    slopes = [(by - ay) / (bx - ax) for i, (ax, ay) in enumerate(points)
              for bx, by in points[i + 1:] if bx - ax >= .1]
    slope = median(slopes) if slopes else 0.0
    residuals = [y - slope * x for x, y in points]
    height = median(c["box"][3] for c in cells)
    if abs(slope) > .05 or max(residuals) - min(residuals) > .45 * height:
        return None
    return slope


def _mark_neighbour_ambiguity(cells):
    # A number close to two overlapping label columns is not a proven label/value pair.
    numbers = [c for c in cells if c.get("table_id") is None
               and re.fullmatch(r"[+-]?\d[\d\s.,]*", c["text"].strip())]
    labels = [c for c in cells if any(ch.isalpha() for ch in c["text"])]
    for number in numbers:
        _, ny = _center(number)
        near = [c for c in labels if c.get("table_id") is None
                and abs(_center(c)[1] - ny) <= .9 * min(c["box"][3], number["box"][3])
                and (c["box"][0] + c["box"][2] <= number["box"][0]
                     or c["box"][0] >= number["box"][0] + number["box"][2])]
        for i, a in enumerate(near):
            for b in near[i + 1:]:
                ax, _, aw, ah = a["box"]
                bx, _, bw, bh = b["box"]
                overlap = min(ax + aw, bx + bw) - max(ax, bx)
                if overlap > .5 * min(aw, bw) and abs(_center(a)[1] - _center(b)[1]) > .45 * min(ah, bh):
                    for cell in (number, a, b):
                        _review(cell, "ambiguous_label_value_alignment")


def assemble_rows(cells):
    cells = deepcopy(cells)
    for index, cell in enumerate(cells):
        cell.setdefault("cell_id", f"cell-{index + 1}")
        if cell.get("score") is not None and cell["score"] < .85:
            cell["low_confidence"] = True
            _review(cell, "low_recognition_confidence")
    rows = _group(cells)
    output, index, table_count = [], 0, 0
    while index < len(rows):
        header = rows[index]
        if not _header(header):
            output.append(header)
            index += 1
            continue
        table_count += 1
        table_id = f"table-{table_count}"
        headers, boundaries = _table_columns(header)
        for cell in headers:
            cell["table_id"] = table_id
        columns = [{"index": i, "text": c["text"], "box": c["box"], "cell_id": c["cell_id"]}
                   for i, c in enumerate(headers)]
        shared = {"table_id": table_id, "header_cell_ids": [c["cell_id"] for c in headers],
                  "header_text": " | ".join(c["text"] for c in headers), "table_columns": columns}
        header.update(shared, table_role="header")
        output.append(header)
        end = index + 1
        previous = header
        while end < len(rows) and not _table_end(rows[end]) and not _header(rows[end]):
            if rows[end]["center"] - previous["center"] > max(.04, 3 * previous["height"]):
                break
            previous = rows[end]
            end += 1
        body = rows[index + 1:end]
        body_cells = [c for row in body for c in row["cells"]]
        for cell in body_cells:
            cell["table_id"] = table_id
            _assign_column(cell, boundaries)
        # Never split a single recognized string into fictitious reference/description cells.
        slope = _header_slope(headers)
        if slope is None:
            for cell in body_cells:
                _review(cell, "unreliable_header_alignment")
        elif body_cells and all(c["column_index"] is not None for c in body_cells):
            body = _group(body_cells, slope, columns=True)
        anchor = None
        desc_column = next(i for i, c in enumerate(headers) if _label(c["text"]) in {"designation", "description", "items"})
        for row in body:
            row.update(shared, table_role="body")
            row["column_cells"] = [[c["cell_id"] for c in row["cells"] if c["column_index"] == i]
                                   for i in range(len(headers))]
            if len(row["cells"]) > 1:
                anchor = row
            elif anchor and row["cells"][0]["column_index"] == desc_column:
                row["possible_continuation_of"] = [c["cell_id"] for c in sorted(anchor["cells"], key=lambda c: c["box"][0])]
                _review(row["cells"][0], "description_continuation_needs_review")
            output.append(row)
        index = end
    _mark_neighbour_ambiguity(cells)
    lines = []
    for row in output:
        ordered = sorted(row["cells"], key=lambda c: c["box"][0])
        reasons = sorted({reason for c in ordered for reason in c.get("review_reasons", [])})
        metadata = {key: value for key, value in row.items() if key not in {"center", "height", "cells"}}
        lines.append({**metadata, "line_number": len(lines) + 1, "page_number": 1,
                      "text": " | ".join(c["text"] for c in ordered), "cells": ordered,
                      "boxes": [c["box"] for c in ordered], "requires_review": bool(reasons),
                      "review_reasons": reasons})
    return lines
