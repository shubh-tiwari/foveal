"""Deterministic page rendering, cached on disk so a page always has the same bytes."""

from __future__ import annotations

import base64
import hashlib
import threading
from pathlib import Path


class PageRenderer:
    def __init__(
        self,
        pdf: Path,
        cache_dir: Path = Path(".cache/pages"),
        dpi: int = 144,
        long_edge: int = 1568,
        jpeg_quality: int = 85,
    ):
        import pymupdf as fitz

        self.pdf = pdf
        self.dpi, self.long_edge, self.quality = dpi, long_edge, jpeg_quality
        key = hashlib.sha1(str(pdf.resolve()).encode()).hexdigest()[:10]
        self.dir = cache_dir / f"{pdf.stem[:40]}-{key}" / f"{dpi}dpi_{long_edge}_{jpeg_quality}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._doc = fitz.open(pdf)
        self.page_count = self._doc.page_count
        self._lock = threading.Lock()

    def toc(self) -> list[tuple[int, str, int]]:
        return [(lvl, title, page) for lvl, title, page in self._doc.get_toc()]

    def title(self) -> str:
        return (self._doc.metadata or {}).get("title") or self.pdf.stem

    def jpeg(self, page_no: int) -> bytes:
        """1-indexed page as JPEG bytes."""
        import pymupdf as fitz
        from PIL import Image

        path = self.dir / f"p{page_no:04d}.jpg"
        if path.exists():
            return path.read_bytes()
        with self._lock:
            if path.exists():
                return path.read_bytes()
            page = self._doc[page_no - 1]
            zoom = self.dpi / 72
            rect = page.rect
            longest = max(rect.width, rect.height) * zoom
            if longest > self.long_edge:
                zoom *= self.long_edge / longest
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            tmp = path.with_suffix(".tmp")
            im.save(tmp, "JPEG", quality=self.quality, optimize=True)
            tmp.replace(path)  # atomic, so concurrent readers never see a partial file
        return path.read_bytes()

    def image_block(self, page_no: int) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(self.jpeg(page_no)).decode(),
            },
        }

    def page_blocks(self, pages: list[int]) -> list[dict]:
        blocks: list[dict] = []
        for p in pages:
            blocks.append({"type": "text", "text": f"Page {p}:"})
            blocks.append(self.image_block(p))
        return blocks
