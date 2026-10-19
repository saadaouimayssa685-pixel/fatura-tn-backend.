"""Local Latin/Arabic OCR with source cells; no invoice values are inferred here."""
import os
import re
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageOps
from paddle_structure import assemble_rows


ARABIC = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")


def result_data(result):
    data = result.json
    return data.get("res", data)


def prefer_arabic(latin_text, latin_score, arabic_text, arabic_score):
    letters = [c for c in arabic_text if c.isalpha()]
    fraction = sum(bool(ARABIC.match(c)) for c in letters) / max(1, len(letters))
    return fraction >= .5 and arabic_score >= .6 and (latin_score < .8 or arabic_score > latin_score + .05)


def needs_arabic(text, score):
    if re.fullmatch(r"[\d\s.,/+%():-]+", text or ""):
        return False
    return score < .85 or bool(re.search(r"[\u0600-\u06ff\u3400-\u9fff\u3040-\u30ff]", text))


def polygon_crop(image, polygon):
    # Paddle's quadrilateral order is top-left, top-right, bottom-right, bottom-left.
    points = np.asarray(polygon, dtype=np.float32)
    width = max(1, round(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))))
    height = max(1, round(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))))
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, target), (width, height),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


class PaddleTextExtractor:
    def __init__(self):
        self.ocr = None
        self.arabic = None
        self.lock = threading.Lock()
        size = os.getenv("PADDLE_TEXT_SIZE", "small")
        if size not in {"tiny", "small", "medium"}:
            raise ValueError("Unsupported Paddle text model size")
        self.detector_name = f"PP-OCRv6_{size}_det"
        self.recognizer_name = f"PP-OCRv6_{size}_rec"

    def _load(self):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR, TextRecognition
        if self.ocr is None:
            self.ocr = PaddleOCR(text_detection_model_name=self.detector_name,
                text_recognition_model_name=self.recognizer_name, use_doc_orientation_classify=False,
                use_doc_unwarping=False, use_textline_orientation=False, text_rec_score_thresh=0.0,
                text_det_limit_side_len=1600, text_det_limit_type="max",
                device="cpu", enable_mkldnn=False, cpu_threads=4)
        if self.arabic is None and os.getenv("PADDLE_TEXT_ARABIC", "true").lower() == "true":
            self.arabic = TextRecognition(model_name="arabic_PP-OCRv5_mobile_rec", device="cpu",
                                          enable_mkldnn=False, cpu_threads=4)

    def extract_text(self, file_path):
        started = time.monotonic()
        with self.lock:
            self._load()
            with Image.open(file_path) as source:
                image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
            height, width = image.shape[:2]
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            result = result_data(next(iter(self.ocr.predict(bgr))))
            polygons = result["rec_polys"]
            arabic_results = {}
            if self.arabic is not None and len(polygons):
                indices = [i for i, (t, s) in enumerate(zip(result["rec_texts"], result["rec_scores"]))
                           if needs_arabic(t, float(s))]
                crops = [polygon_crop(bgr, polygons[i]) for i in indices]
                if crops:
                    arabic_results = dict(zip(indices, (result_data(r) for r in self.arabic.predict(crops, batch_size=1))))
            cells = []
            for index, (value, score, polygon) in enumerate(zip(result["rec_texts"], result["rec_scores"], polygons)):
                language = "latin"
                alternatives = []
                if index in arabic_results:
                    ar = arabic_results[index]
                    alternative, ar_score = ar.get("rec_text", ""), float(ar.get("rec_score", 0))
                    if prefer_arabic(value, float(score), alternative, ar_score):
                        alternatives.append({"text": value, "score": float(score), "language": "latin"})
                        value, score, language = alternative, ar_score, "arabic"
                if not value.strip() or score < .4:
                    continue
                points = np.asarray(polygon)
                x0, y0 = points.min(axis=0)
                x1, y1 = points.max(axis=0)
                cells.append({"text": value, "score": float(score), "language": language,
                              "box": [float(x0 / width), float(y0 / height), float((x1 - x0) / width), float((y1 - y0) / height)],
                              "alternatives": alternatives})
        lines = assemble_rows(cells)
        text = "\n".join(line["text"] for line in lines)
        confidence = sum(c["score"] for c in cells) / len(cells) if cells else 0.0
        return text, confidence, {"method_used": "paddle_text", "engines_tried": ["paddle_text"],
            "models": [self.detector_name, self.recognizer_name] + (["arabic_PP-OCRv5_mobile_rec"] if self.arabic else []),
            "score_kind": "ocr_recognition_score_not_field_accuracy", "lines": lines,
            "requires_review": any(line["requires_review"] for line in lines),
            "review_lines": [line["line_number"] for line in lines if line["requires_review"]],
            "seconds": round(time.monotonic() - started, 3), "coordinate_space": "exif_oriented_image",
            "image_width": width, "image_height": height}
