import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from payment_stamp_ocr import extract_payment_stamp


class PaymentStampTests(unittest.TestCase):
    @patch("payment_stamp_ocr.pytesseract.image_to_string", return_value="Non paye en especes")
    def test_negated_payment_is_not_accepted(self, ocr):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(extract_payment_stamp(self.scan(directory, "blue")))

    def scan(self, directory, color):
        path = Path(directory) / "scan.png"
        image = Image.new("RGB", (160, 200), "white")
        image.paste(color, (20, 110, 120, 140))
        image.save(path)
        return path

    @patch("payment_stamp_ocr.pytesseract.image_to_string", return_value="Signature client")
    def test_signature_does_not_imply_payment(self, ocr):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(extract_payment_stamp(self.scan(directory, "blue")))

    @patch("payment_stamp_ocr.pytesseract.image_to_string", side_effect=["Paye en especes", "Paye en especes"])
    def test_two_matching_reads_preserve_exact_phrase(self, ocr):
        with tempfile.TemporaryDirectory() as directory:
            result = extract_payment_stamp(self.scan(directory, "blue"))
            self.assertEqual(result["text"], "Paye en especes")
            self.assertEqual(len(result["readings"]), 2)

    @patch("payment_stamp_ocr.pytesseract.image_to_string")
    def test_monochrome_scan_does_not_trigger_coloured_stamp_ocr(self, ocr):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(extract_payment_stamp(self.scan(directory, "black")))
            ocr.assert_not_called()
