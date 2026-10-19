"""Local GLM-OCR adapter for complex invoice images."""
import base64
import os
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageOps


class GLMOCRExtractor:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.model_name = os.getenv("OLLAMA_OCR_MODEL", "glm-ocr:q8_0")
        self.timeout = int(os.getenv("OLLAMA_OCR_TIMEOUT_SECONDS", "180"))

    def is_available(self):
        try:
            models = requests.get(f"{self.base_url}/api/tags", timeout=5).json().get("models", [])
            return any(model.get("name") == self.model_name for model in models)
        except Exception:
            return False

    def extract(self, image_path):
        path = Path(image_path)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            return ""
        with Image.open(path) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
            max_side = int(os.getenv("OLLAMA_OCR_MAX_IMAGE_SIDE", "1600"))
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, "JPEG", quality=92, optimize=True)
        prompt = (
            "Text Recognition and Table Recognition. Transcribe every visible printed French, Arabic and numeric "
            "element in reading order. Preserve identifiers, leading zeros, decimal separators and line breaks. "
            "Render tables as Markdown with one product per row. Do not summarize, translate, calculate, correct, "
            "classify or invent missing text. Return only the transcription."
        )
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model_name, "stream": False,
                  "messages": [{"role": "user", "content": prompt,
                                "images": [base64.b64encode(buffer.getvalue()).decode("ascii")]}],
                  "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 4096}},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("done_reason") == "length" or payload.get("done") is False:
            raise ValueError("GLM OCR transcription incomplete")
        return str(payload.get("message", {}).get("content", "")).strip()
