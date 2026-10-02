import base64
import io

import pytest
from PIL import Image, ImageDraw


def make_png(w=300, h=200, seed=0) -> bytes:
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    box = [10 + seed * 7, 10, 120 + seed * 11, 90 + seed * 5]
    d.rectangle(box, fill=(seed * 40 % 255, 80, 160))
    d.text((20, 120), f"page {seed}", fill="black")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def image_block(png: bytes) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(png).decode(),
        },
    }


@pytest.fixture
def png_factory():
    return make_png


@pytest.fixture
def block_factory():
    return image_block
