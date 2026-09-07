import os
import tempfile
from typing import Dict, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image


class OptimizedOCRExtractor:
    """OCR image extractor using local Tesseract with safe Windows configuration."""

    def __init__(self):
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(backend_dir)
        configured_tessdata_dir = os.getenv("TESSDATA_DIR", os.path.join(project_root, "backend", "tessdata"))
        if not os.path.isabs(configured_tessdata_dir):
            configured_tessdata_dir = os.path.join(project_root, configured_tessdata_dir)
        self.tessdata_dir = configured_tessdata_dir
        self.base_ocr_config = f"-l fra+eng --tessdata-dir {self.tessdata_dir}"
        self.ocr_config = f"--psm 6 {self.base_ocr_config}"
        self._configure_tesseract()
        self.fast_mode = os.getenv("OCR_FAST_MODE", "false").lower() == "true"

    def _configure_tesseract(self):
        common_windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(common_windows_path):
            pytesseract.pytesseract.tesseract_cmd = common_windows_path

        try:
            pytesseract.get_tesseract_version()
            print("Tesseract OCR available")
        except Exception as error:
            print(f"Tesseract OCR unavailable: {error}")

    def extract_text(self, image_path: str, method: str = "optimized") -> Tuple[str, float, Dict]:
        text, confidence = self.extract_full_text(image_path)
        return text, confidence, {
            "method_used": "tesseract",
            "engines_tried": ["tesseract"],
            "preprocessing": "grayscale_denoise_threshold",
        }

    def extract_full_text(self, image_path: str) -> Tuple[str, float]:
        if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            return "", 0.0

        image = cv2.imread(image_path)
        if image is None:
            return "", 0.0

        if self.fast_mode:
            return self._fast_extract(image)

        variants = self._build_preprocessing_variants(image)
        return self._run_tesseract_best(variants)

    def _fast_extract(self, image):
        height, width = image.shape[:2]
        # Render Free provides only 0.1 CPU. At 800px, headers, table labels
        # and totals remain usable while Tesseract finishes before its timeout.
        max_dimension = max(600, int(os.getenv("OCR_MAX_DIMENSION", "800")))
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Suppress faint reverse-side printing before segmenting the full page.
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = self.ocr_config.replace("--psm 6", "--psm 3")
        data = pytesseract.image_to_data(
            Image.fromarray(image), config=config,
            output_type=pytesseract.Output.DICT, timeout=90,
        )
        lines = {}
        confidences = []
        for index, word in enumerate(data["text"]):
            if not word.strip():
                continue
            key = tuple(data[field][index] for field in ("page_num", "block_num", "par_num", "line_num"))
            lines.setdefault(key, []).append(word)
            confidence = float(data["conf"][index])
            if confidence >= 0:
                confidences.append(confidence)
        return ("\n".join(" ".join(words) for words in lines.values()),
                sum(confidences) / len(confidences) / 100 if confidences else 0.0)

    def extract_full_text_from_array(self, image_array) -> Dict:
        try:
            if image_array is None or image_array.size == 0:
                return {"text": "", "confidence": 0.0}

            if self.fast_mode:
                text, confidence = self._fast_extract(image_array)
                return {"text": text, "confidence": confidence}

            variants = self._build_preprocessing_variants(image_array)
            text, confidence = self._run_tesseract_best(variants)
            return {"text": text, "confidence": confidence}
        except Exception as error:
            print(f"OCR array extraction error: {error}")
            return {"text": "", "confidence": 0.0}

    def get_available_methods(self) -> Dict:
        return {
            "optimized": True,
            "tesseract": True,
        }

    def _preprocess_array(self, image_array):
        return self._adaptive_threshold_variant(image_array)

    def _build_preprocessing_variants(self, image_array):
        variants = []

        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array.copy()

        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)

        variants.append(("gray", gray))

        height, width = gray.shape[:2]
        if max(height, width) > 2600:
            scale = 2200 / max(height, width)
            resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            variants.append(("gray_resized", resized))

        variants.append(("adaptive_threshold", self._adaptive_threshold_variant(gray)))

        try:
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(("otsu_threshold", otsu))
        except Exception:
            pass

        return variants

    def _adaptive_threshold_variant(self, image_array):
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array.copy()

        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)

        height, width = gray.shape[:2]
        if height > 50 and width > 50:
            gray = cv2.bilateralFilter(gray, 9, 75, 75)

        try:
            return cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )
        except Exception:
            _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            return threshold

    def _run_tesseract(self, image_array) -> Tuple[str, float]:
        return self._run_tesseract_best([("single", image_array)])

    def _run_tesseract_best(self, variants) -> Tuple[str, float]:
        configs = [
            f"--psm 6 {self.base_ocr_config}",
            f"--psm 4 {self.base_ocr_config}",
            f"--psm 11 {self.base_ocr_config}",
        ]

        best_text = ""
        best_confidence = 0.0
        best_score = 0.0

        for variant_name, image_array in variants:
            pil_image = Image.fromarray(image_array)
            for config in configs:
                try:
                    text = pytesseract.image_to_string(pil_image, config=config).strip()
                except Exception as error:
                    print(f"Tesseract text extraction error ({variant_name}): {error}")
                    continue

                confidence = self._calculate_confidence(pil_image, config=config)
                useful_chars = len([char for char in text if char.isalnum()])
                score = useful_chars * max(confidence, 0.05)

                if score > best_score:
                    best_text = text
                    best_confidence = confidence
                    best_score = score

        return best_text, best_confidence

    def _run_tesseract_single(self, image_array) -> Tuple[str, float]:
        pil_image = Image.fromarray(image_array)

        try:
            text = pytesseract.image_to_string(pil_image, config=self.ocr_config)
        except Exception as error:
            print(f"Tesseract text extraction error: {error}")
            return "", 0.0

        confidence = self._calculate_confidence(pil_image)
        return text.strip(), confidence

    def _calculate_confidence(self, pil_image: Image.Image, config: str = None) -> float:
        try:
            data = pytesseract.image_to_data(
                pil_image,
                config=config or self.ocr_config,
                output_type=pytesseract.Output.DICT,
            )
            confidences = []
            for raw_confidence in data.get("conf", []):
                try:
                    confidence = float(raw_confidence)
                except ValueError:
                    continue
                if confidence > 0:
                    confidences.append(confidence)

            if not confidences:
                return 0.0

            return sum(confidences) / len(confidences) / 100
        except Exception as error:
            print(f"Tesseract confidence error: {error}")
            return 0.0
