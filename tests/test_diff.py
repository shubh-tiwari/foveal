import numpy as np
from PIL import Image, ImageDraw

from foveal.diff import detect_scroll, diff_frames, reconstruct


def page(h=720, w=1280, seed=0, extra=None):
    """A text-like page: rows of varied dark bars on white."""
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (w, 3000), "white")
    d = ImageDraw.Draw(im)
    for y in range(20, 3000, 28):
        x = 40
        while x < w - 100:
            ln = int(rng.integers(20, 120))
            d.rectangle([x, y, x + ln, y + 10], fill=(30, 30, 30))
            x += ln + int(rng.integers(8, 30))
    if extra:
        extra(d)
    return im


def view(im, top, h=720):
    return im.crop((0, top, im.width, top + h))


def test_identical_and_small_change():
    full = page()
    a = view(full, 0)
    assert diff_frames(a, a.copy()).kind == "identical"
    changed = full.copy()
    ImageDraw.Draw(changed).rectangle([600, 300, 700, 330], fill=(200, 0, 0))
    d = diff_frames(a, view(changed, 0))
    assert d.kind == "partial" and len(d.regions) == 1 and d.changed_frac < 0.05
    b = d.regions[0]
    assert b.x <= 600 and b.x + b.w >= 700 and b.y <= 300 and b.y + b.h >= 330


def test_scroll_detected_and_only_new_strip_sent():
    full = page(seed=3)
    a, b = view(full, 100), view(full, 400)  # scrolled down 300px
    assert (
        detect_scroll(
            np.asarray(a.convert("L"), dtype=np.int16), np.asarray(b.convert("L"), dtype=np.int16)
        )
        == 300
    )
    d = diff_frames(a, b)
    assert d.kind == "partial" and d.scroll_dy == 300
    assert 0.35 < d.changed_frac < 0.5  # the 300px revealed strip of 720
    rec = np.asarray(reconstruct(a, b, d), dtype=np.int16)
    assert np.abs(rec - np.asarray(b, dtype=np.int16)).max() < 30


def test_new_page_is_full():
    d = diff_frames(view(page(seed=1), 0), view(page(seed=9), 0))
    assert d.kind == "full"


def test_replay_policies_on_scrolling_trajectory():
    from bench.replay.mind2web import Step, Trajectory
    from bench.replay.simulate import replay

    full = page(seed=5)
    steps = [Step(i, "scroll", view(full, 200 * i), (100, 100, 50, 20)) for i in range(5)]
    r = replay(Trajectory("t", "site", "task", steps))
    tok = r["tokens"]
    assert tok["foveal_diff"] < 0.6 * tok["full_history"]  # only new strips are sent
    assert tok["last_1"] < tok["last_3"] < tok["full_history"]
    assert r["scrolls"] == 4 and all(r["target_ok"])
    assert max(r["fidelity_bad_px"]) == 0
