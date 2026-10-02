"""`Memory`: perceive each visual input once, keep cheap levels, zoom on demand.

mem = Memory(".foveal", captioner=Captioner(client))
pages = mem.ingest_pdf("report.pdf")            # L0-L3 for every page, once
print(mem.describe(pages[3].asset_id))          # L0 caption + L1 text
view = mem.look(pages[3].asset_id, region=(0.1, 0.5, 0.9, 0.8), detail="full")
"""

from __future__ import annotations

import base64
import contextvars
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from foveal.instrument.hashing import sha256_hex
from foveal.instrument.tokens import estimate_image_tokens
from foveal.memory.render import (
    Region,
    crop_image,
    render_pdf_page,
    thumbnail,
    validate_region,
)
from foveal.memory.store import Asset, Store

CAPTION_PROMPT = (
    "Describe this document page in one line of at most 30 words: what kind of page it is "
    "and its main visual elements (charts, tables, figures, photos), naming their titles or "
    "subjects. Do not transcribe body text. Reply with the line only."
)


class Captioner:
    """L0 captions from a cheap vision model. Calls go through whatever client is passed,
    so an InstrumentedAnthropic client logs (and bills) them like any other call."""

    def __init__(
        self,
        client: Any,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 120,
        image_tokens: int | None = 384,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.image_tokens = image_tokens  # downscale pages to this many visual tokens first

    def __call__(self, jpeg: bytes) -> str:
        if self.image_tokens:
            jpeg = thumbnail(jpeg, self.image_tokens, quality=85)
        block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(jpeg).decode(),
            },
        }
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "user", "content": [block, {"type": "text", "text": CAPTION_PROMPT}]}
            ],
        )
        text = " ".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return " ".join(text.split())


@dataclass
class View:
    """What `look()` returns: JPEG bytes plus their size and visual-token cost."""

    asset_id: str
    detail: str
    region: Region | None
    jpeg: bytes
    width: int
    height: int
    tokens: int

    def block(self) -> dict[str, Any]:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(self.jpeg).decode(),
            },
        }


def _size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as im:
        return im.size


class Memory:
    def __init__(
        self,
        root: str | Path = ".foveal",
        captioner: Captioner | None = None,
        model: str = "claude-sonnet-5-5",
        glimpse_tokens: int = 256,
        long_edge: int = 1568,
    ):
        self.store = Store(root)
        self.captioner = captioner
        self.model = model  # used for token estimates of what look() returns
        self.glimpse_tokens = glimpse_tokens
        self.long_edge = long_edge
        self._pdfs: dict[str, Any] = {}
        self._render_lock = threading.Lock()  # PyMuPDF documents are not thread-safe

    # -- ingest -----------------------------------------------------------

    def _finish(self, asset: Asset, l3: bytes) -> Asset:
        existing = self.store.get(asset.asset_id)
        if existing is not None:  # perceived before: nothing to pay
            if existing.caption is None and self.captioner is not None:
                self.store.update_levels(existing.asset_id, L0=self.captioner(l3))
                existing = self.store.get(asset.asset_id)
            return existing
        if asset.levels.pop("L1_needs_ocr", False):  # scanned page: OCR once, at first sight
            text = ocr(l3)
            if text and len(text) > len(asset.levels.get("L1") or ""):
                asset.levels["L1"], asset.levels["L1_source"] = text, "ocr"
        asset.levels["L3"] = str(self.store.write_blob(asset.asset_id, l3))
        l2 = thumbnail(l3, self.glimpse_tokens)
        asset.levels["L2"] = str(self.store.write_blob(asset.asset_id + "_l2", l2))
        if self.captioner is not None:
            asset.levels["L0"] = self.captioner(l3)
        self.store.put(asset)
        return asset

    def ingest_image(
        self, data: bytes, source: str = "", kind: str = "image", text: str | None = None
    ) -> Asset:
        sha = sha256_hex(data)
        w, h = _size(data)
        levels: dict[str, Any] = {}
        if text is None:
            text = ocr(data)
        if text is not None:
            levels["L1"] = text
        return self._finish(
            Asset(asset_id=sha, kind=kind, source=source, width=w, height=h, levels=levels), data
        )

    def ingest(self, source: str | Path | bytes, **kw: Any) -> Asset | list[Asset]:
        if isinstance(source, bytes):
            return self.ingest_image(source, **kw)
        path = Path(source)
        if path.suffix.lower() == ".pdf":
            return self.ingest_pdf(path, **kw)
        return self.ingest_image(path.read_bytes(), source=str(path), **kw)

    def _pdf(self, path: str) -> Any:
        import pymupdf

        if path not in self._pdfs:
            self._pdfs[path] = pymupdf.open(path)
        return self._pdfs[path]

    def ingest_pdf(
        self, path: str | Path, dpi: int = 144, workers: int = 8, min_text_chars: int = 40
    ) -> list[Asset]:
        """One `page` asset per page. L1 is the PDF text layer, or OCR when a page has
        almost no text layer (scanned pages) and the `ocr` extra is installed. Crops
        re-render from the PDF."""
        path = str(Path(path).resolve())
        doc = self._pdf(path)
        rendered = []
        for n in range(1, doc.page_count + 1):
            l3 = render_pdf_page(doc, n, dpi=dpi, long_edge=self.long_edge)
            text = doc[n - 1].get_text().strip()
            w, h = _size(l3)
            rendered.append(
                (
                    Asset(
                        asset_id=sha256_hex(l3),
                        kind="page",
                        source=f"{path}#page={n}",
                        width=w,
                        height=h,
                        levels={
                            "L1": text,
                            "L1_source": "pdf",
                            **({"L1_needs_ocr": True} if len(text) < min_text_chars else {}),
                        },
                        origin={"pdf": path, "page": n, "dpi": dpi},
                    ),
                    l3,
                )
            )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [
                pool.submit(contextvars.copy_context().run, self._finish, a, l3)
                for a, l3 in rendered
            ]
            return [f.result() for f in futs]

    # -- read -------------------------------------------------------------

    def get(self, asset_id: str) -> Asset:
        a = self.store.get(asset_id)
        if a is None:
            raise KeyError(f"unknown asset {asset_id}")
        return a

    def describe(
        self, asset_id: str, levels: tuple[str, ...] = ("L0", "L1"), max_text_chars: int = 4000
    ) -> str:
        """Text-only view of an asset: its L0 handle and (optionally) L1 structure."""
        a = self.get(asset_id)
        parts = [f"[asset {a.asset_id[:12]} | {a.kind} | {a.width}x{a.height}]"]
        if "L0" in levels:
            parts.append(f"Caption: {a.caption or '(none)'}")
        if "L1" in levels:
            text = (a.text or "").strip()
            if len(text) > max_text_chars:
                text = text[:max_text_chars] + " [...truncated]"
            parts.append(f"Text layer:\n{text}" if text else "Text layer: (no text)")
        return "\n".join(parts)

    def look(self, asset_id: str, region: Any = None, detail: str = "glimpse") -> View:
        """Page in a glimpse (L2, ~`glimpse_tokens`) or full detail (L3, optionally cropped)."""
        a = self.get(asset_id)
        reg = validate_region(region)
        if detail == "glimpse":
            if reg is None:
                data = Path(a.levels["L2"]).read_bytes()
            else:
                data = thumbnail(
                    crop_image(Path(a.levels["L3"]).read_bytes(), reg), self.glimpse_tokens
                )
        elif detail == "full":
            if reg is None:
                data = Path(a.levels["L3"]).read_bytes()
            elif "pdf" in a.origin:
                with self._render_lock:
                    data = render_pdf_page(
                        self._pdf(a.origin["pdf"]),
                        a.origin["page"],
                        long_edge=self.long_edge,
                        region=reg,
                    )
            else:
                data = crop_image(Path(a.levels["L3"]).read_bytes(), reg, self.long_edge)
        else:
            raise ValueError("detail must be 'glimpse' or 'full'")
        w, h = _size(data)
        return View(asset_id, detail, reg, data, w, h, estimate_image_tokens(w, h, self.model))

    def context(self, asset_ids: list[str]) -> str:
        """L0 handles for a set of assets (the budget-aware assembler comes in Phase 4)."""
        return "\n".join(self.describe(i, levels=("L0",)) for i in asset_ids)


def ocr(data: bytes) -> str | None:
    """L1 text for plain images via Tesseract, when the `ocr` extra is installed."""
    try:
        import pytesseract
    except ImportError:
        return None
    with Image.open(io.BytesIO(data)) as im:
        return pytesseract.image_to_string(im).strip() or None
