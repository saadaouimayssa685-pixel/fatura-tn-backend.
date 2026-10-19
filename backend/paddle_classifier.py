"""PaddleOCR layout + document text; layout alone is not an invoice classifier."""
import os
import re
import threading
import time
from pathlib import Path
from extraction_chunks import image_layout_frame


def layout_decision(text, boxes, base_result):
    decision, score, details = base_result
    details = dict(details)
    tables = [box for box in boxes if box.get("label") == "table" and box.get("score", 0) >= .5]
    title = bool(re.search(r"\bfactu\s*re\b|\binvoice\b|فاتورة", text, re.I))
    # A delivery note or quotation may contain exactly the same table and totals.
    other_title = bool(re.search(r"\b(?:devis|quotation|pro\s*forma)\b|عرض\s+سعر", text, re.I))
    delivery_only = bool(re.search(r"\bbon\s+de\s+livraison\b", text, re.I)) and not title
    if other_title or delivery_only or details.get("other_document_title"):
        decision, score = False, .15
    elif title and tables and details.get("negative_keyword_matches", 0) < 3 and details.get("amount_label_matches", 0) >= 1:
        decision, score = True, max(score, .85)
    details.update({
        "analysis_method": "paddleocr_layout_analysis",
        "score_kind": "decision_score_not_probability",
        "layout_model": "PP-DocLayout_plus-L",
        "layout_boxes": boxes,
        "table_count": len(tables),
        "evidence": [f"PaddleOCR : {len(tables)} zone(s) de tableau", "Type déterminé avec le texte du document et sa structure"],
    })
    return decision, score, details


class PaddleDocumentClassifier:
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()
        self.error = None
        self.elapsed = None

    def _load(self):
        if self.model is None:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import LayoutDetection
            self.model = LayoutDetection(model_name="PP-DocLayout_plus-L", device="cpu",
                                         enable_mkldnn=False, cpu_threads=4)

    def classify(self, file_path, text, base_result):
        started = time.monotonic()
        try:
            with self.lock:
                self._load()
                boxes = []
                for page, result in enumerate(self.model.predict(file_path, batch_size=1, layout_nms=True, threshold=.3), 1):
                    data = result.json
                    data = data.get("res", data)
                    boxes.extend({"page_number": page, "label": b["label"], "score": float(b["score"]),
                                  "coordinate": [float(c) for c in b["coordinate"]]} for b in data.get("boxes", []))
                    if page >= 10:
                        break
                decision, score, details = layout_decision(text, boxes, base_result)
                details.update(image_layout_frame(file_path))
                self.error = None
                self.elapsed = round(time.monotonic() - started, 3)
                details["layout_seconds"] = self.elapsed
                return decision, score, details
        except Exception as error:
            self.error = type(error).__name__
            decision, score, details = base_result
            return decision, score, {**details, "analysis_method": "text_ocr_classification_fallback",
                                     "paddle_error": self.error, "reason": "Analyse PaddleOCR indisponible; décision OCR de secours"}

    def status(self):
        return {"status": "ready" if self.model is not None and not self.error else "error" if self.error else "lazy",
                "model": "PP-DocLayout_plus-L", "error": self.error, "last_seconds": self.elapsed}
