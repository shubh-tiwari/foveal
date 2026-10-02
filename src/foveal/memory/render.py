"""Deterministic rendering of PDF pages, thumbnails and crops to JPEG bytes."""

from __future__ import annotations

import io
import math
from typing import Any

from PIL import Image

from foveal.instrument.tokens import PATCH

Region = tuple[float, float, float, float]  # x0, y0, x1, y1 as fractions of width/height


def _jpeg(im: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def render_pdf_page(
    doc: Any,
    page_no: int,
    dpi: int = 144,
    long_edge: int = 1568,
    quality: int = 85,
    region: Region | None = None,
    max_dpi: int = 600,
) -> bytes:
    """Render a 1-indexed page (or a fractional region of it) to JPEG.

    A region is rendered at the zoom that fills `long_edge`, capped at `max_dpi`, so a crop
    of a small chart has real extra detail rather than upscaled pixels.
    """
    import pymupdf

    page = doc[page_no - 1]
    rect = page.rect
    clip = None
    if region is not None:
        x0, y0, x1, y1 = region
        clip = pymupdf.Rect(
            rect.x0 + x0 * rect.width,
            rect.y0 + y0 * rect.height,
            rect.x0 + x1 * rect.width,
            rect.y0 + y1 * rect.height,
        )
        target = clip
        zoom = min(long_edge / max(target.width, target.height), max_dpi / 72)
    else:
        target = rect
        zoom = dpi / 72
        longest = max(target.width, target.height) * zoom
        if longest > long_edge:
            zoom *= long_edge / longest
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return _jpeg(im, quality)


def fit_to_tokens(w: int, h: int, max_tokens: int) -> tuple[int, int]:
    """Largest size with the same aspect ratio costing at most `max_tokens` visual tokens."""
    scale = min(1.0, math.sqrt(max_tokens * PATCH * PATCH / (w * h)))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    while math.ceil(nw / PATCH) * math.ceil(nh / PATCH) > max_tokens:
        scale *= 0.98
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return nw, nh


def thumbnail(data: bytes, max_tokens: int = 256, quality: int = 80) -> bytes:
    with Image.open(io.BytesIO(data)) as im:
        size = fit_to_tokens(*im.size, max_tokens)
        return _jpeg(im.resize(size, Image.Resampling.LANCZOS), quality)


def crop_image(data: bytes, region: Region, long_edge: int = 1568, quality: int = 85) -> bytes:
    with Image.open(io.BytesIO(data)) as im:
        w, h = im.size
        x0, y0, x1, y1 = region
        box = (
            int(x0 * w),
            int(y0 * h),
            max(int(x0 * w) + 1, int(x1 * w)),
            max(int(y0 * h) + 1, int(y1 * h)),
        )
        c = im.crop(box)
        if max(c.size) > long_edge:
            s = long_edge / max(c.size)
            c = c.resize(
                (max(1, int(c.width * s)), max(1, int(c.height * s))), Image.Resampling.LANCZOS
            )
        return _jpeg(c, quality)


def validate_region(region: Any) -> Region | None:
    """Accept [x0, y0, x1, y1] fractions in [0, 1]; return None for the whole image."""
    if region is None:
        return None
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        raise ValueError("region must be [x0, y0, x1, y1] as fractions between 0 and 1")
    x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in region)
    if x1 - x0 < 0.02 or y1 - y0 < 0.02:
        raise ValueError("region is too small; x1 > x0 and y1 > y0 by at least 0.02")
    return (x0, y0, x1, y1)
