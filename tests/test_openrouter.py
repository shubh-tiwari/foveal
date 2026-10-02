import json
from types import SimpleNamespace

from bench.openrouter import from_openai, to_openai


def test_request_conversion_moves_tool_images_to_a_user_message():
    img = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"},
    }
    msgs = [
        {"role": "user", "content": [img, {"type": "text", "text": "what is on page 3?"}]},
        {
            "role": "assistant",
            "content": [
                SimpleNamespace(type="text", text="looking"),
                SimpleNamespace(type="tool_use", id="c1", name="look", input={"page": 3}),
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "content": [{"type": "text", "text": "Page 3:"}, img],
                }
            ],
        },
    ]
    out = to_openai("be brief", msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "user"]
    assert out[1]["content"][0]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"page": 3}
    assert out[3]["tool_call_id"] == "c1" and "attached in the next message" in out[3]["content"]
    assert out[4]["content"][1]["type"] == "image_url"


def test_response_conversion_and_reasoning_replay():
    data = {
        "model": "google/gemini-3.8-flash",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_details": [{"type": "reasoning.encrypted"}],
                    "tool_calls": [
                        {
                            "id": "c9",
                            "type": "function",
                            "function": {"name": "look", "arguments": '{"page": 2}'},
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "cost": 0.0012,
            "prompt_tokens_details": {"cached_tokens": 600},
        },
    }
    r = from_openai(data, "google/gemini-3.8-flash")
    assert r.stop_reason == "tool_use"
    use = next(b for b in r.content if b.type == "tool_use")
    assert (use.name, use.input) == ("look", {"page": 2})
    u = r.usage.model_dump()
    assert (u["input_tokens"], u["cache_read_input_tokens"], u["cost_usd"]) == (400, 600, 0.0012)
    # the next request replays the provider message unmodified, reasoning_details included
    replay = to_openai(None, [{"role": "assistant", "content": r.content}])
    assert replay[0]["reasoning_details"] == [{"type": "reasoning.encrypted"}]
