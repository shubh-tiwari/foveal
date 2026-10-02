from foveal.instrument.hashing import inspect_image_bytes, phash_distance
from foveal.instrument.tokens import estimate_image_tokens, resized_dims


def test_doc_examples_standard_tier():
    m = "claude-haiku-4-5"
    assert estimate_image_tokens(200, 200, m) == 64
    assert estimate_image_tokens(1000, 1000, m) == 1296
    assert estimate_image_tokens(1092, 1092, m) == 1521
    assert estimate_image_tokens(1920, 1080, m) <= 1568


def test_doc_examples_high_res_tier():
    m = "claude-opus-5-5"
    assert estimate_image_tokens(1920, 1080, m) == 2691
    assert estimate_image_tokens(2000, 1500, m) == 3888
    assert resized_dims(3840, 2160, m)[0] <= 2576
    assert estimate_image_tokens(3840, 2160, m) <= 4784


def test_hash_stable_and_phash_close(png_factory):
    a = inspect_image_bytes(png_factory(seed=1))
    b = inspect_image_bytes(png_factory(seed=1))
    c = inspect_image_bytes(png_factory(seed=5))
    assert a.sha256 == b.sha256 and (a.width, a.height) == (300, 200)
    assert a.sha256 != c.sha256
    assert phash_distance(a.phash, b.phash) == 0
