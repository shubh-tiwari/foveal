"""Diff engine for consecutive observations (screenshots, video frames).

`diff_frames(prev, curr)` finds what changed between two frames:
  * identical - nothing worth sending
  * partial   - a few changed regions (optionally after a vertical scroll)
  * full      - too much changed; send the whole frame

Scrolling is detected first: if `curr` is `prev` shifted vertically, only the newly
revealed strip and genuine changes count. Change detection is tile-based so JPEG noise
does not register as change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass
class FrameDiff:
    kind: str  # identical | partial | full
    scroll_dy: int = 0  # within the scroll band, curr row r shows prev row r + scroll_dy
    scroll_band: tuple[int, int] | None = None  # rows [y0, y1) that scrolled (None = all)
    regions: list[Box] = field(default_factory=list)
    changed_frac: float = 0.0


def _gray(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), dtype=np.int16)


def detect_scroll(
    prev: np.ndarray, curr: np.ndarray, min_overlap: float = 0.3, noise: float = 2.0
) -> int:
    """Vertical shift dy such that curr[r] ~= prev[r + dy], or 0 if none is clearly better."""
    h = prev.shape[0]
    # Row signatures: mean and spread of each row (spread keeps blank rows from matching anything)
    sp = np.stack([prev.mean(1), prev.std(1)], 1)
    sc = np.stack([curr.mean(1), curr.std(1)], 1)
    lo = int(h * min_overlap)

    def err_of(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        # Primary: 75th percentile of per-row error, so a toast or changed widget covering a
        # minority of rows does not hide an otherwise clean scroll. Secondary: mean error,
        # which separates the exact shift from near-misses that only blank rows support.
        e = np.abs(a - b).sum(axis=1)
        return float(np.percentile(e, 75)), float(e.mean())

    best_dy, best = 0, err_of(sp, sc)
    if best[0] < noise and best[1] < noise:
        return 0
    for dy in range(-(h - lo), h - lo + 1):
        if dy == 0:
            continue
        if dy > 0:
            a, b = sp[dy:], sc[: h - dy]
        else:
            a, b = sp[: h + dy], sc[-dy:]
        if (a[:, 1] > 1).mean() < 0.2:  # overlap is mostly blank rows: no evidence
            continue
        err = err_of(a, b)
        if (round(err[0], 3), err[1]) < (round(best[0], 3), best[1]):
            best_dy, best = dy, err
    return best_dy if best[0] < noise else 0


def fixed_bands(prev: np.ndarray, curr: np.ndarray, tol: float = 4.0) -> tuple[int, int]:
    """Rows [y0, y1) between an unchanged top band (sticky header) and bottom band (footer)."""
    same = np.abs(prev - curr).mean(axis=1) < tol
    h = len(same)
    if same.all():
        return 0, h
    y0 = int(np.argmax(~same))
    y1 = h - int(np.argmax(~same[::-1]))
    return y0, y1


def _apply_scroll(
    prev: np.ndarray, dy: int, band: tuple[int, int] | None
) -> tuple[np.ndarray, np.ndarray]:
    """prev with rows in `band` scrolled by dy, plus a mask of rows with no prev content."""
    h = prev.shape[0]
    y0, y1 = band or (0, h)
    out = prev.copy()
    revealed = np.zeros(h, dtype=bool)
    if dy:
        out[y0:y1], revealed[y0:y1] = _shift(prev[y0:y1], dy)
    return out, revealed


def _shift(prev: np.ndarray, dy: int) -> tuple[np.ndarray, np.ndarray]:
    """prev aligned to curr's coordinates, plus a mask of rows with no prev content."""
    h = prev.shape[0]
    out = np.zeros_like(prev)
    revealed = np.zeros(h, dtype=bool)
    if dy > 0:
        out[: h - dy] = prev[dy:]
        revealed[h - dy :] = True
    elif dy < 0:
        out[-dy:] = prev[: h + dy]
        revealed[:-dy] = True
    else:
        out = prev.copy()
    return out, revealed


def diff_frames(
    prev: Image.Image,
    curr: Image.Image,
    tile: int = 16,
    pix_thresh: int = 24,
    min_pixels: int = 3,
    merge_gap: int = 2,
    full_frac: float = 0.6,
    scroll: bool = True,
) -> FrameDiff:
    if prev.size != curr.size:
        return FrameDiff("full", changed_frac=1.0)
    p, c = _gray(prev), _gray(curr)
    h, w = c.shape
    dy, band = 0, None
    if scroll:
        y0, y1 = fixed_bands(p, c)
        if y1 - y0 >= 0.3 * h:  # scroll inside the band between sticky header and footer
            dy = detect_scroll(p[y0:y1], c[y0:y1])
            band = (y0, y1) if dy and (y0, y1) != (0, h) else None
    aligned, revealed = _apply_scroll(p, dy, band)
    changed = np.abs(c - aligned) > pix_thresh
    changed[revealed] = True
    th, tw = -(-h // tile), -(-w // tile)
    pad = np.zeros((th * tile, tw * tile), dtype=bool)
    pad[:h, :w] = changed
    counts = pad.reshape(th, tile, tw, tile).sum(axis=(1, 3))
    mask = counts >= min_pixels
    if not mask.any():
        return FrameDiff("identical", scroll_dy=dy, scroll_band=band)
    mask = ndimage.binary_dilation(mask, iterations=merge_gap)  # join nearby changes
    labels, _ = ndimage.label(mask)
    boxes = []
    for sl in ndimage.find_objects(labels):
        y0, y1 = sl[0].start * tile, min(h, sl[0].stop * tile)
        x0, x1 = sl[1].start * tile, min(w, sl[1].stop * tile)
        boxes.append(Box(x0, y0, x1 - x0, y1 - y0))
    boxes = _merge(boxes)
    frac = _union_area(boxes, w, h) / (w * h)
    if frac >= full_frac:
        return FrameDiff("full", scroll_dy=dy, scroll_band=band, changed_frac=frac)
    return FrameDiff("partial", scroll_dy=dy, scroll_band=band, regions=boxes, changed_frac=frac)


def _merge(boxes: list[Box]) -> list[Box]:
    """Merge overlapping boxes until stable."""
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        out: list[Box] = []
        while boxes:
            a = boxes.pop()
            for i, b in enumerate(out):
                if not (a.x + a.w < b.x or b.x + b.w < a.x or a.y + a.h < b.y or b.y + b.h < a.y):
                    x0, y0 = min(a.x, b.x), min(a.y, b.y)
                    x1, y1 = max(a.x + a.w, b.x + b.w), max(a.y + a.h, b.y + b.h)
                    out[i] = Box(x0, y0, x1 - x0, y1 - y0)
                    merged = True
                    break
            else:
                out.append(a)
        boxes = out
    return sorted(boxes, key=lambda b: (b.y, b.x))


def _union_area(boxes: list[Box], w: int, h: int) -> int:
    m = np.zeros((h, w), dtype=bool)
    for b in boxes:
        m[b.y : b.y + b.h, b.x : b.x + b.w] = True
    return int(m.sum())


def reconstruct(prev: Image.Image, curr: Image.Image, d: FrameDiff) -> Image.Image:
    """What a receiver holding `prev` sees after applying the diff (for fidelity checks)."""
    if d.kind == "full":
        return curr.copy()
    arr = np.asarray(prev.convert("RGB"))
    out, _ = _apply_scroll(arr, d.scroll_dy, d.scroll_band)
    img = Image.fromarray(out)
    for b in d.regions:
        img.paste(curr.crop((b.x, b.y, b.x + b.w, b.y + b.h)), (b.x, b.y))
    return img


def describe(d: FrameDiff, since: str = "the previous frame") -> str:
    """Text that accompanies the crops in the model's context."""
    if d.kind == "identical":
        s = f"Unchanged since {since}"
        return s + (f" except scrolled by {d.scroll_dy}px." if d.scroll_dy else ".")
    if d.kind == "full":
        return "Changed substantially; full frame follows."
    parts = [f"Same as {since}"]
    if d.scroll_dy:
        where = f" (rows {d.scroll_band[0]}-{d.scroll_band[1]})" if d.scroll_band else ""
        parts.append(f"scrolled {'down' if d.scroll_dy > 0 else 'up'} {abs(d.scroll_dy)}px{where}")
    regions = "; ".join(f"({b.x},{b.y},{b.w}x{b.h})" for b in d.regions)
    return ", ".join(parts) + f", except {len(d.regions)} changed region(s): {regions}."
