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

    @patch("ocr_extractor.pytesseract.image_to_string", return_value="")
    @patch("ocr_extractor.pytesseract.image_to_data")
    def test_client_crop_does_not_exceed_hosted_pixel_cap(self, extract, extract_header):
        extract.return_value = {
            "text": ["FACTURE"], "conf": [90], "page_num": [1],
            "block_num": [1], "par_num": [1], "line_num": [1],
        }
        instance = OptimizedOCRExtractor.__new__(OptimizedOCRExtractor)
        instance.ocr_config = "--psm 6"
        with patch.dict("os.environ", {"OCR_FAST_REGION_MAX_DIMENSION": "1000"}, clear=False):
            instance._fast_extract(np.zeros((1654, 2338), dtype=np.uint8))

        canvas = np.array(extract.call_args.args[0])
        self.assertLessEqual(max(canvas.shape), 3000)

    @patch("ocr_extractor.pytesseract.image_to_string", return_value="Date : 28-11-2019")
    @patch("ocr_extractor.pytesseract.image_to_data")
    def test_missing_date_gets_a_small_header_retry(self, extract_data, extract_header):
        extract_data.return_value = {
            "text": ["FACTURE", "00633"], "conf": [90, 90], "page_num": [1, 1],
            "block_num": [1, 1], "par_num": [1, 1], "line_num": [1, 1],
        }
        instance = OptimizedOCRExtractor.__new__(OptimizedOCRExtractor)
        instance.ocr_config = "--psm 4 -l fra+eng"

        text, _ = instance._fast_extract(np.zeros((1654, 2338), dtype=np.uint8))

        self.assertIn("Date : 28-11-2019", text)
        self.assertEqual(extract_header.call_count, 1)
        self.assertEqual(extract_header.call_args.kwargs["timeout"], 45)
        self.assertIn("--psm 4", extract_header.call_args.kwargs["config"])

    @patch("ocr_extractor.pytesseract.image_to_string", return_value="Date : 28-11-2019")
    @patch("ocr_extractor.pytesseract.image_to_data")
    def test_invalid_date_gets_a_small_header_retry(self, extract_data, extract_header):
        extract_data.return_value = {
            "text": ["Date", ":", "20-14-2019"], "conf": [90, 90, 90], "page_num": [1] * 3,
            "block_num": [1] * 3, "par_num": [1] * 3, "line_num": [1] * 3,
        }
        instance = OptimizedOCRExtractor.__new__(OptimizedOCRExtractor)
        instance.ocr_config = "--psm 4 -l fra+eng"

        text, _ = instance._fast_extract(np.zeros((1654, 2338), dtype=np.uint8))

        self.assertIn("Date : 28-11-2019", text)
        self.assertEqual(extract_header.call_count, 1)

    @patch.object(OptimizedOCRExtractor, "_extract_header_date_evidence", return_value="Date : 28-11-2019")
    def test_header_date_retry_is_shared_by_full_ocr(self, extract_header):
        instance = OptimizedOCRExtractor.__new__(OptimizedOCRExtractor)

        text, confidence = instance._append_header_date_evidence(
            "Date : 20-14-2019", 0.55, np.zeros((1654, 2338), dtype=np.uint8)
        )

        self.assertEqual(confidence, 0.55)
        self.assertIn("Date : 28-11-2019", text)
        self.assertEqual(extract_header.call_count, 1)

    @patch("ocr_extractor.pytesseract.image_to_string", return_value="FACTURE QA-1")
    def test_timeout_uses_small_focused_regions(self, extract):
        instance = OptimizedOCRExtractor.__new__(OptimizedOCRExtractor)
        instance.ocr_config = "--psm 4 -l fra+eng"
        text, confidence = instance._extract_focused_regions(np.zeros((1654, 2338), dtype=np.uint8))

        self.assertEqual(text.count("FACTURE QA-1"), 5)
        self.assertEqual(confidence, 0.55)
        self.assertEqual(extract.call_count, 5)
        self.assertEqual(extract.call_args.kwargs["timeout"], 20)
        self.assertIn("--psm 6", extract.call_args.kwargs["config"])

    @patch("ocr_extractor.pytesseract.image_to_string", return_value="FACTURE QA-1")
    @patch("ocr_extractor.pytesseract.image_to_data", side_effect=RuntimeError("timeout"))
    def test_fast_timeout_retries_with_focused_regions(self, extract_data, extract_text):
        instance = OptimizedOCRExtractor.__new__(OptimizedOCRExtractor)
        instance.ocr_config = "--psm 4 -l fra+eng"

        text, confidence = instance._fast_extract(np.zeros((80, 100), dtype=np.uint8))

        self.assertIn("FACTURE QA-1", text)
        self.assertEqual(confidence, 0.55)
        self.assertEqual(extract_data.call_count, 1)
        self.assertGreater(extract_text.call_count, 0)

    @patch.object(OptimizedOCRExtractor, "_extract_header_date_evidence", return_value="Date : 28-11-2019")
    @patch.object(OptimizedOCRExtractor, "_extract_focused_regions", return_value=("Date : 20-14-2019", 0.55))
    @patch("ocr_extractor.pytesseract.image_to_data", side_effect=RuntimeError("timeout"))
    def test_fast_timeout_adds_header_date_evidence(
        self, extract_data, extract_focused, extract_header
    ):
        instance = OptimizedOCRExtractor.__new__(OptimizedOCRExtractor)
        instance.ocr_config = "--psm 4 -l fra+eng"

        text, confidence = instance._fast_extract(np.zeros((1654, 2338), dtype=np.uint8))

        self.assertEqual(confidence, 0.55)
        self.assertIn("Date : 28-11-2019", text)
        self.assertEqual(extract_data.call_count, 1)
        self.assertEqual(extract_focused.call_count, 1)
        self.assertEqual(extract_header.call_count, 1)


if __name__ == "__main__":
    unittest.main()
