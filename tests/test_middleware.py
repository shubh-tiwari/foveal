import base64
import io

from PIL import ImageDraw

from foveal import Memory
from foveal.middleware import Rewriter, run_look_tool, wrap
from tests.test_diff import page, view


def b64(im, fmt="PNG"):
    buf = io.BytesIO()
    im.save(buf, fmt)
    return base64.b64encode(buf.getvalue()).decode()


def shot(im):
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": b64(im)},
    }


def screenshot_conversation():
    """A computer-use style loop: each tool_result returns a screenshot."""
    full = page(seed=41)
    f1 = view(full, 0)
    f2 = f1.copy()
    ImageDraw.Draw(f2).rectangle([300, 300, 500, 340], fill=(0, 0, 0))  # typed into a field
    f3 = view(full, 0)  # back to the first screen exactly
    msgs = [{"role": "user", "content": [{"type": "text", "text": "book a flight"}]}]
    for i, f in enumerate([f1, f2, f3]):
        msgs.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"t{i}", "name": "computer", "input": {}}],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": [shot(f)]}],
            }
        )
    return msgs


def test_rewrite_dedup_and_diff(tmp_path):
    rw = Rewriter(Memory(tmp_path / "s"))
    msgs = screenshot_conversation()
    out, st = rw.rewrite(msgs)
    assert st.images == 3 and st.diffed == 1 and st.deduplicated == 1
    assert st.tokens_after < 0.45 * st.tokens_before
    second = out[4]["content"][0]["content"]
    assert second[0]["type"] == "text" and "changed region" in second[0]["text"]
    assert any(b["type"] == "image" for b in second)  # the crop
    third = out[6]["content"][0]["content"]
    assert [b["type"] for b in third] == ["text"] and "identical" in third[0]["text"]
    assert msgs[4]["content"][0]["content"][0]["type"] == "image"  # input untouched


def test_rewrite_is_prefix_stable(tmp_path):
    """Calling again with one more turn must not change any earlier rewritten message."""
    rw = Rewriter(Memory(tmp_path / "s"))
    msgs = screenshot_conversation()
    a, _ = rw.rewrite(msgs[:5])
    b, _ = rw.rewrite(msgs)
    assert b[:5] == a


def test_openai_format_and_wrapper(tmp_path):
    f = view(page(seed=42), 0)
    url = "data:image/png;base64," + b64(f)
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "what is this?"},
            ],
        },
        {"role": "assistant", "content": "a page"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]},
    ]

    class Completions:
        def create(self, **kw):
            self.sent = kw["messages"]
            return "ok"

    class FakeOpenAI:
        def __init__(self):
            self.chat = type("C", (), {"completions": Completions()})()

    raw = FakeOpenAI()
    client = wrap(raw, Memory(tmp_path / "s"), fmt="openai")
    assert client.chat.completions.create(model="m", messages=msgs) == "ok"
    sent = raw.chat.completions.sent
    assert sent[0]["content"][0]["type"] == "image_url"
    assert sent[2]["content"][0]["type"] == "text" and "identical" in sent[2]["content"][0]["text"]
    assert client.last_stats.deduplicated == 1


def test_look_tool_pages_detail_back(tmp_path):
    mem = Memory(tmp_path / "s")
    rw = Rewriter(mem)
    out, _ = rw.rewrite(screenshot_conversation())
    note = out[6]["content"][0]["content"][0]["text"]
    asset = note.split("asset ")[1][:12]
    blocks = run_look_tool(mem, {"asset": asset, "detail": "full"})
    assert blocks[-1]["type"] == "image"
