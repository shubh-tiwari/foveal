from types import SimpleNamespace

from foveal.instrument import InstrumentedAnthropic, JsonlSink, Tracer
from foveal.instrument.metrics import compute, gate
from foveal.instrument.records import read_jsonl


class FakeMessages:
    def create(self, **kw):
        return SimpleNamespace(
            model=kw["model"],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                model_dump=lambda **_: {
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            ),
        )

    def count_tokens(self, **kw):
        return SimpleNamespace(input_tokens=1)


class FakeClient:
    messages = FakeMessages()


def test_resend_history_is_two_thirds_repeated(tmp_path, png_factory, block_factory):
    log = tmp_path / "r.jsonl"
    tracer = Tracer("r", JsonlSink(log))
    client = InstrumentedAnthropic(FakeClient(), tracer)
    img = block_factory(png_factory(seed=1))
    history = [{"role": "user", "content": [img, {"type": "text", "text": "q"}]}]
    with tracer.scope(task_id="t1", agent_id="orch", doc_id="d"):
        for _ in range(3):  # same image resent on every call
            client.messages.create(model="claude-opus-5-5", max_tokens=1, messages=history)
            history = history + [
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "more"},
            ]
    recs = read_jsonl(log)
    assert [r["agent_step"] for r in recs] == [1, 2, 3]
    s = compute(recs)
    assert abs(s["reperception_rate"] - 2 / 3) < 1e-9
    assert s["same_agent_share"] == s["reperception_rate"]
    assert gate(s, 0.4)[0]


def test_cross_agent_and_tool_result_nesting(tmp_path, png_factory, block_factory):
    log = tmp_path / "r.jsonl"
    tracer = Tracer("r", JsonlSink(log))
    client = InstrumentedAnthropic(FakeClient(), tracer)
    a, b = block_factory(png_factory(seed=1)), block_factory(png_factory(seed=2))
    nested = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "x", "content": [a, b]}],
        }
    ]
    with tracer.scope(task_id="t1", doc_id="d"):
        with tracer.scope(agent_id="reader-1"):
            client.messages.create(model="claude-opus-5-5", max_tokens=1, messages=nested)
        with tracer.scope(agent_id="reader-2"):
            client.messages.create(
                model="claude-opus-5-5", max_tokens=1, messages=[{"role": "user", "content": [a]}]
            )
    recs = read_jsonl(log)
    assert recs[0]["images"][0]["path"] == "0.content.0.content.0"
    s = compute(recs)
    t = s["tasks"][0]
    assert t.tokens["cross_agent"] == recs[1]["images"][0]["est_tokens"]
    assert t.tokens["same_agent"] == 0
    assert t.unique_images == 2


def test_cross_task_same_doc(tmp_path, png_factory, block_factory):
    log = tmp_path / "r.jsonl"
    tracer = Tracer("r", JsonlSink(log))
    client = InstrumentedAnthropic(FakeClient(), tracer)
    a = block_factory(png_factory(seed=3))
    for tid in ("t1", "t2"):
        with tracer.scope(task_id=tid, agent_id="orch", doc_id="d"):
            client.messages.create(
                model="claude-opus-5-5", max_tokens=1, messages=[{"role": "user", "content": [a]}]
            )
    s = compute(read_jsonl(log))
    assert s["reperception_rate"] == 0
    assert s["cross_task_share_of_first_seen"] == 0.5
