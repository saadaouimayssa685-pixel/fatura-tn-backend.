import os
import tempfile
from typing import Any, Dict, List, Tuple

import pdfplumber


class PDFExtractor:
    """Extracts text from PDFs with a native-text first strategy and OCR fallback."""

    def __init__(self, ocr_extractor, min_native_text_chars: int = 80):
        self.ocr_extractor = ocr_extractor
        self.min_native_text_chars = min_native_text_chars

    def extract_text(self, pdf_path: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Extract text from a PDF.

        Strategy:
        1. Use pdfplumber for native PDF text and tables.
        2. If the native text is too poor, render pages as images and OCR them.
        """
        native_text, native_metadata = self._extract_native_text(pdf_path)

        if self._is_text_sufficient(native_text):
            return native_text, 0.95, {
                "method_used": "pdfplumber",
                "fallback_used": False,
                **native_metadata,
            }

        ocr_text, ocr_confidence, ocr_metadata = self._extract_with_ocr_fallback(pdf_path)

        if ocr_text.strip():
            return ocr_text, ocr_confidence, {
                "method_used": "pdf_rendered_pages_ocr",
                "fallback_used": True,
                "native_text_length": len(native_text),
                "native_metadata": native_metadata,
                "ocr_metadata": ocr_metadata,
            }

        return native_text, 0.3 if native_text.strip() else 0.0, {
            "method_used": "pdfplumber",
            "fallback_used": True,
            "fallback_error": ocr_metadata.get("error"),
            **native_metadata,
        }

    def _extract_native_text(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        pages_payload: List[str] = []
        tables_count = 0

        with pdfplumber.open(pdf_path) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                page_chunks = [f"--- Page {index} ---"]
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""

                if text.strip():
                    page_chunks.append(text.strip())

                tables = page.extract_tables() or []
                for table_index, table in enumerate(tables, start=1):
                    tables_count += 1
                    page_chunks.append(f"--- Table {table_index} ---")
                    page_chunks.extend(self._format_table(table))

                pages_payload.append("\n".join(page_chunks).strip())

            metadata = {
                "pages_count": len(pdf.pages),
                "tables_count": tables_count,
                "native_text_length": sum(len(chunk) for chunk in pages_payload),
            }

        return "\n\n".join(chunk for chunk in pages_payload if chunk).strip(), metadata

    def _extract_with_ocr_fallback(self, pdf_path: str) -> Tuple[str, float, Dict[str, Any]]:
        try:
            import fitz  # PyMuPDF
        except Exception as error:
            return "", 0.0, {
                "error": f"PyMuPDF is required for scanned PDF OCR fallback: {error}",
            }

        page_texts: List[str] = []
        page_confidences: List[float] = []
        page_metadata: List[Dict[str, Any]] = []

        try:
            with tempfile.TemporaryDirectory(prefix="pdf_ocr_") as temp_dir:
                document = fitz.open(pdf_path)

                for page_index in range(len(document)):
                    page = document.load_page(page_index)
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    image_path = os.path.join(temp_dir, f"page_{page_index + 1}.png")
                    pixmap.save(image_path)

                    text, confidence, metadata = self.ocr_extractor.extract_text(image_path)
                    if text.strip():
                        page_texts.append(f"--- Page {page_index + 1} ---\n{text.strip()}")
                    page_confidences.append(confidence)
                    page_metadata.append({
                        "page": page_index + 1,
                        "confidence": confidence,
                        "text_length": len(text),
                        "ocr_metadata": metadata,
                    })

                document.close()

            valid_confidences = [score for score in page_confidences if score > 0]
            avg_confidence = (
                sum(valid_confidences) / len(valid_confidences)
                if valid_confidences
                else 0.0
            )

            return "\n\n".join(page_texts).strip(), avg_confidence, {
                "pages_count": len(page_metadata),
                "page_metadata": page_metadata,
            }

        except Exception as error:
            return "", 0.0, {
                "error": str(error),
                "page_metadata": page_metadata,
            }

    def _is_text_sufficient(self, text: str) -> bool:
        compact_text = "".join(char for char in text if char.isalnum())
        return len(compact_text) >= self.min_native_text_chars

    def _format_table(self, table: List[List[Any]]) -> List[str]:
        rows = []
        for row in table:
            cells = ["" if cell is None else str(cell).strip() for cell in row]
            rows.append(" | ".join(cells))
        return rows
