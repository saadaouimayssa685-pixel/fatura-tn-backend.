"""Targeted OCR of coloured payment stamps; no payment inference from a signature."""
import re

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps


PAYMENT_PHRASE = re.compile(r"\bpay[ée]e?\s+en\s+(?:esp[eèé]ces|ch[eèé]que)\b", re.I)


def extract_payment_stamp(path, languages="fra+ara+eng"):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        pixels = np.asarray(image)
    hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (90, 65, 35), (135, 255, 255))
    mask[:height // 3] = 0
    ys, xs = np.where(mask > 0)
    if len(xs) < 80:
        return None
    left, right = max(0, int(xs.min()) - 12), min(width, int(xs.max()) + 13)
    top, bottom = max(0, int(ys.min()) - 12), min(height, int(ys.max()) + 13)
    crop = Image.fromarray(255 - mask[top:bottom, left:right])
    crop.thumbnail((1200, 650))
    readings = []
    for angle in (0, -15, -10, -20, -30, 15, 30):
        rotated = crop.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
        try:
            text = pytesseract.image_to_string(rotated, config=f"--psm 11 -l {languages}", timeout=10)
        except RuntimeError:
            continue
        match = PAYMENT_PHRASE.search(text)
        if match and re.search(r"(?:non|pas(?:\s+encore)?)\s*$", text[:match.start()], re.I):
            match = None
        if match:
            readings.append({"text": match.group(), "angle": angle})
        # Require two agreeing OCR readings before accepting a coloured stamp.
        if len(readings) >= 2:
            methods = {"especes" if re.search(r"esp[eèé]ces", row["text"], re.I) else "cheque" for row in readings}
            if len(methods) == 1:
                return {"text": readings[0]["text"], "method": "coloured_stamp_tesseract",
                        "readings": readings, "box": [left / width, top / height,
                                                       (right - left) / width, (bottom - top) / height]}
    return None
