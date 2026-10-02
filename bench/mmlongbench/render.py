"""Page sources for the harness.

PageRenderer (baseline): pages go to the model as full images.
MemoryPages (foveal): pages go as L0 caption + L1 text; images only through look().
Both render with foveal.memory.render, so a page has identical bytes in either mode.
"""

from __future__ import annotations

import base64
import hashlib
import threading
from pathlib import Path
from typing import Any

from foveal.memory import Memory
from foveal.memory.render import render_pdf_page


class PageRenderer:
    images = True

    def __init__(
        self,
        pdf: Path,
        cache_dir: Path = Path(".cache/pages"),
        dpi: int = 144,
        long_edge: int = 1568,
        jpeg_quality: int = 85,
    ):
        import pymupdf

        self.pdf = pdf
        self.dpi, self.long_edge, self.quality = dpi, long_edge, jpeg_quality
        key = hashlib.sha1(str(pdf.resolve()).encode()).hexdigest()[:10]
        self.dir = cache_dir / f"{pdf.stem[:40]}-{key}" / f"{dpi}dpi_{long_edge}_{jpeg_quality}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._doc = pymupdf.open(pdf)
        self.page_count = self._doc.page_count
        self._lock = threading.Lock()

    def toc(self) -> list[tuple[int, str, int]]:
        return [(lvl, title, page) for lvl, title, page in self._doc.get_toc()]

    def title(self) -> str:
        return (self._doc.metadata or {}).get("title") or self.pdf.stem

    def jpeg(self, page_no: int) -> bytes:
        """1-indexed page as JPEG bytes, cached on disk."""
        path = self.dir / f"p{page_no:04d}.jpg"
        if path.exists():
            return path.read_bytes()
        with self._lock:
            if not path.exists():
                data = render_pdf_page(self._doc, page_no, self.dpi, self.long_edge, self.quality)
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(data)
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


class MemoryPages(PageRenderer):
    """Same document, served from a foveal Memory: text levels by default, pixels on demand."""

    images = False

    def __init__(self, pdf: Path, memory: Memory, **kw: Any):
        super().__init__(pdf, **kw)
        self.memory = memory
        self.assets = memory.ingest_pdf(pdf, dpi=self.dpi)

    def page_blocks(self, pages: list[int]) -> list[dict]:
        text = "\n\n".join(
            f"Page {p} {self.memory.describe(self.assets[p - 1].asset_id)}" for p in pages
        )
        return [{"type": "text", "text": text}]

    def look(self, page: int, region: Any = None, detail: str = "glimpse") -> list[dict]:
        view = self.memory.look(self.assets[page - 1].asset_id, region=region, detail=detail)
        where = (
            "whole page"
            if view.region is None
            else ("region " + ", ".join(f"{v:.2f}" for v in view.region))
        )
        return [
            {
                "type": "text",
                "text": f"Page {page}, {detail}, {where} ({view.width}x{view.height}px):",
            },
            view.block(),
        ]
