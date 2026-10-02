from PIL import ImageDraw

from bench.replay.agent_eval import history_blocks, parse, prompt
from bench.replay.mind2web import Step, Trajectory
from tests.test_diff import page, view


def _traj():
    full = page(seed=4)
    frames = [view(full, 0), view(full, 0).copy(), view(full, 300)]
    ImageDraw.Draw(frames[1]).rectangle([100, 100, 300, 140], fill=(0, 0, 0))
    opts = [{"box": (10, 10, 50, 20), "tag": "button", "desc": 'id="go"'}]
    steps = [Step(i, f"act {i}", f, None, opts, 0, "CLICK") for i, f in enumerate(frames)]
    return Trajectory("t", "site", "do it", steps)


def _images(blocks):
    return [b for b in blocks if b["type"] == "image"]


def test_policies_send_expected_images():
    t = _traj()
    assert len(_images(history_blocks(t.steps, 2, "full_history"))) == 3
    assert len(_images(history_blocks(t.steps, 2, "last_3"))) == 3
    fov = history_blocks(t.steps, 2, "foveal_diff")
    texts = " ".join(b["text"] for b in fov if b["type"] == "text")
    assert "scrolled down 300px" in texts and "changed region" in texts
    msgs = prompt(t, 2, "foveal_diff")[0]["content"]
    cached = [b for b in msgs if "cache_control" in b]
    assert len(cached) == 1 and msgs.index(cached[0]) == len(msgs) - 2  # before the question


def test_parse():
    assert parse("C TYPE: New York") == (2, "TYPE")
    assert parse("Answer: B CLICK") == (1, "CLICK")
    assert parse("no idea") == (None, "")
