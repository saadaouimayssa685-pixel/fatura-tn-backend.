import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DocumentChunk:
    text: str
    page_number: Optional[int]
    chunk_index: int


class DocumentChunker:
    """Splits extracted document text into retrieval-friendly chunks."""

    def __init__(self, max_chars: int = 900, overlap_chars: int = 120):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def split(self, text: str) -> List[DocumentChunk]:
        pages = self._split_pages(text)
        chunks: List[DocumentChunk] = []

        for page_number, page_text in pages:
            for chunk_text in self._split_text(page_text):
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        page_number=page_number,
                        chunk_index=len(chunks),
                    )
                )

        return chunks

    def _split_pages(self, text: str):
        matches = list(re.finditer(r"---\s*Page\s+(\d+)\s*---", text, flags=re.IGNORECASE))
        if not matches:
            return [(None, text.strip())]

        pages = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            page_number = int(match.group(1))
            page_text = text[start:end].strip()
            if page_text:
                pages.append((page_number, page_text))

        return pages

    def _split_text(self, text: str) -> List[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        chunks: List[str] = []
        current = ""

        for paragraph in paragraphs:
            if not current:
                current = paragraph
                continue

            if len(current) + len(paragraph) + 2 <= self.max_chars:
                current = f"{current}\n\n{paragraph}"
            else:
                chunks.extend(self._split_long_chunk(current))
                current = paragraph

        if current:
            chunks.extend(self._split_long_chunk(current))

        return chunks

    def _split_long_chunk(self, text: str) -> List[str]:
        if len(text) <= self.max_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(0, end - self.overlap_chars)

        return [chunk for chunk in chunks if chunk]
