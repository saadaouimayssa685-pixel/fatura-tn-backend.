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
        self.assertEqual(extract.call_args.kwargs["timeout"], 40)
        self.assertEqual(extract.call_args.kwargs["config"], "--psm 4")
        self.assertTrue(set(np.unique(np.array(extract.call_args.args[0]))) <= {0, 255})

    @patch("ocr_extractor.pytesseract.image_to_data")
    def test_client_crop_does_not_exceed_hosted_pixel_cap(self, extract):
        extract.return_value = {
            "text": ["FACTURE"], "conf": [90], "page_num": [1],
            "block_num": [1], "par_num": [1], "line_num": [1],
        }
        instance = OptimizedOCRExtractor.__new__(OptimizedOCRExtractor)
        instance.ocr_config = "--psm 6"
        with patch.dict("os.environ", {"OCR_FAST_REGION_MAX_DIMENSION": "1000"}, clear=False):
            instance._fast_extract(np.zeros((1654, 2338), dtype=np.uint8))

        canvas = np.array(extract.call_args.args[0])
        self.assertLessEqual(max(canvas.shape), 5000)


if __name__ == "__main__":
    unittest.main()
