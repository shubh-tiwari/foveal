"""Next-action prediction on Mind2Web under different screenshot-history policies.

For each step the model gets the task, its previous actions, screenshots under a policy,
and up to 8 labelled candidate elements on the current screen; it answers with the option
letter and the operation. Same steps, same options, same model for every policy, so the
only variable is how screenshot history is sent:

  full_history  every screenshot so far, full
  last_3        the last 3 screenshots, full
  foveal_diff   first screenshot full, then each later one as a diff (text + changed crops)

Calls go through the foveal instrumentation (tokens, cache, cost per policy).

    uv run python -m bench.replay.agent_eval --dry-run
    uv run python -m bench.replay.agent_eval --steps 150 --max-cost-usd 6   # live, needs budget
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image

from bench.replay.mind2web import Step, Trajectory, load
from foveal.diff import describe, diff_frames
from foveal.instrument import InstrumentedAnthropic, JsonlSink, Tracer

LETTERS = "ABCDEFGH"
POLICIES = ("full_history", "last_3", "foveal_diff")

SYSTEM = """You are a web agent completing a task on a website. You see screenshots of the \
browser viewport (1280x720 px) from earlier steps and the current step, the actions you \
already took, and candidate elements on the current screen given as letter, tag, \
attributes and pixel box (x, y, width, height). Choose the element for the next action.

Reply on one line: the letter, then the operation: CLICK, TYPE: <text> or SELECT: <option>. \
Example: C TYPE: New York"""


def _jpeg_block(im: Image.Image) -> dict:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=85)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.b64encode(buf.getvalue()).decode(),
        },
    }


def history_blocks(steps: list[Step], k: int, policy: str) -> list[dict]:
    """Screenshot content for step k (0-based) under a policy."""
    blocks: list[dict] = []
    if policy == "full_history":
        for i in range(k + 1):
            blocks += [
                {"type": "text", "text": f"Screenshot at step {i + 1}:"},
                _jpeg_block(steps[i].frame),
            ]
    elif policy == "last_3":
        for i in range(max(0, k - 2), k + 1):
            blocks += [
                {"type": "text", "text": f"Screenshot at step {i + 1}:"},
                _jpeg_block(steps[i].frame),
            ]
    elif policy == "foveal_diff":
        blocks += [{"type": "text", "text": "Screenshot at step 1:"}, _jpeg_block(steps[0].frame)]
        for i in range(1, k + 1):
            d = diff_frames(steps[i - 1].frame, steps[i].frame)
            note = describe(d, since=f"step {i}")
            blocks.append({"type": "text", "text": f"Screenshot at step {i + 1}: {note}"})
            if d.kind == "full":
                blocks.append(_jpeg_block(steps[i].frame))
            else:
                for b in d.regions:
                    blocks.append({"type": "text", "text": f"Region ({b.x},{b.y},{b.w}x{b.h}):"})
                    blocks.append(
                        _jpeg_block(steps[i].frame.crop((b.x, b.y, b.x + b.w, b.y + b.h)))
                    )
    else:
        raise ValueError(policy)
    return blocks


def prompt(t: Trajectory, k: int, policy: str) -> list[dict]:
    s = t.steps[k]
    prev = "\n".join(f"{i + 1}. {st.action}" for i, st in enumerate(t.steps[:k])) or "(none)"
    opts = "\n".join(
        f"{LETTERS[i]}. <{o['tag']}> {o['desc']} box={o['box']}" for i, o in enumerate(s.options)
    )
    history = history_blocks(t.steps, k, policy)
    # Cache breakpoint after the screenshots: the next step of an append-only policy starts
    # with exactly this prefix, so it is read from cache; a sliding window never matches.
    history[-1] = {**history[-1], "cache_control": {"type": "ephemeral"}}
    return [
        {
            "role": "user",
            "content": [
                *history,
                {
                    "type": "text",
                    "text": f"Task: {t.task}\n\nPrevious actions:\n{prev}\n\n"
                    f"Candidates on the current screen:\n{opts}",
                },
            ],
        }
    ]


def parse(text: str) -> tuple[int | None, str]:
    m = re.search(r"\b([A-H])\b\s*[.:)-]?\s*(CLICK|TYPE|SELECT)?", text.strip())
    if not m:
        return None, ""
    return LETTERS.index(m.group(1)), (m.group(2) or "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150, help="steps to evaluate (all policies)")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-cost-usd", type=float, default=6.0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S") + "-m2w" + (
        "-dry" if args.dry_run else ""
    )
    log = Path("runs") / f"{run_id}.jsonl"
    if args.dry_run:
        from bench.replay.fake import FakeActionClient

        raw: Any = FakeActionClient()
        trajs = load(1, 3, with_candidates=True)[:4]
    else:
        import anthropic
        from dotenv import load_dotenv

        load_dotenv()
        raw = anthropic.Anthropic()
        trajs = load(2, 3, with_candidates=True)
    tracer = Tracer(run_id, JsonlSink(log))
    client = InstrumentedAnthropic(raw, tracer)
    work = [(t, k) for t in trajs for k in range(len(t.steps)) if t.steps[k].answer is not None]
    work = work[: args.steps]
    policies = args.policies.split(",")
    print(f"run {run_id}: {len(work)} steps x {len(policies)} policies -> {log}")
    for t, k in work:  # step-major, so a budget stop leaves policies equally covered
        for policy in policies:
            if tracer.total_cost_usd >= args.max_cost_usd:
                print(f"budget reached (${tracer.total_cost_usd:.2f}); stopping")
                return 0
            with tracer.scope(
                task_id=f"{t.annotation_id[:8]}#{k}@{policy}",
                agent_id=policy,
                policy=policy,
                traj=t.annotation_id,
            ):
                resp = client.messages.create(
                    model=args.model,
                    max_tokens=4000,
                    system=SYSTEM,
                    messages=prompt(t, k, policy),
                    thinking={"type": "adaptive"},
                    output_config={"effort": args.effort},
                )
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                pick, op = parse(text)
                s = t.steps[k]
                tracer.log_event(
                    type="task",
                    policy=policy,
                    step=k,
                    answer=s.answer,
                    pick=pick,
                    op_gold=s.op,
                    op=op,
                    reply=text[:200],
                    element_ok=pick == s.answer,
                    score=float(pick == s.answer and op == s.op),
                )
    print(
        f"done, ${tracer.total_cost_usd:.2f}. summarize with:\n"
        f"  uv run python -m bench.replay.agent_eval_report {log}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
