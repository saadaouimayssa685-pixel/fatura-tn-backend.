import unittest
from unittest.mock import patch
import numpy as np
from ocr_extractor import OptimizedOCRExtractor


class FastOCRTests(unittest.TestCase):
    @patch("ocr_extractor.pytesseract.image_to_data")
    def test_single_pass_preserves_lines(self, extract):
        extract.return_value = {
            "text": ["", "FACTURE", "QA-1", "TOTAL", "24,800"],
            "conf": [-1, 90, 80, 100, 90],
            "page_num": [1] * 5, "block_num": [1] * 5,
            "par_num": [1] * 5, "line_num": [0, 1, 1, 2, 2],
        }
        instance = OptimizedOCRExtractor.__new__(OptimizedOCRExtractor)
        instance.ocr_config = "--psm 6"
        text, confidence = instance._fast_extract(np.zeros((30, 40), dtype=np.uint8))
        self.assertEqual(text, "FACTURE QA-1\nTOTAL 24,800")
        self.assertEqual(confidence, 0.9)
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(extract.call_args.kwargs["timeout"], 90)


if __name__ == "__main__":
    unittest.main()
