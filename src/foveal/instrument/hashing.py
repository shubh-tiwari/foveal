"""Content identity for image blocks: exact (sha256) and perceptual (pHash)."""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass

import imagehash
from PIL import Image


@dataclass
class ImageInfo:
    sha256: str
    width: int | None
    height: int | None
    phash: str | None


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect_image_bytes(data: bytes) -> ImageInfo:
    """Hash and measure raw image bytes. Undecodable bytes still get a sha256."""
    digest = sha256_hex(data)
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            ph = str(imagehash.phash(im.convert("RGB")))
    except Exception:
        return ImageInfo(digest, None, None, None)
    return ImageInfo(digest, w, h, ph)


def decode_base64(data: str) -> bytes:
    return base64.b64decode(data)


def phash_distance(a: str, b: str) -> int:
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)
