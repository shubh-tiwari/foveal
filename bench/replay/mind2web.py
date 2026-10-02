"""Web-agent trajectories from Multimodal-Mind2Web, replayed as viewport screenshots.

Mind2Web stores a full-page screenshot per action. An agent sees a viewport, so each step
becomes a 1280x720 crop scrolled to show that step's target element (or kept at the last
scroll position when the target is unknown).
"""

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

REPO = "osunlp/Multimodal-Mind2Web"
SHARDS = [
    "data/test_domain-00000-of-00011-26c55c12cbbcdc8e.parquet",
    "data/test_domain-00001-of-00011-93dadb8d3ca8a3e9.parquet",
]
VIEW_W, VIEW_H = 1280, 720


@dataclass
class Step:
    index: int
    action: str
    frame: Image.Image  # viewport the agent saw before acting
    target: tuple[int, int, int, int] | None  # target element box in viewport coords
    options: list[dict] = field(default_factory=list)  # candidate elements (with_candidates)
    answer: int | None = None  # index of the correct option
    op: str = ""  # CLICK | TYPE | SELECT


@dataclass
class Trajectory:
    annotation_id: str
    website: str
    task: str
    steps: list[Step] = field(default_factory=list)


def _target_box(pos_candidates: list[str]) -> tuple[float, float, float, float] | None:
    for c in pos_candidates or []:
        try:
            attrs = json.loads(json.loads(c)["attributes"])
            x, y, w, h = (float(v) for v in attrs["bounding_box_rect"].split(","))
            if w > 0 and h > 0:
                return x, y, w, h
        except (KeyError, ValueError, TypeError):
            continue
    return None


def _viewport(page: Image.Image, box, last_top: int) -> tuple[Image.Image, int, tuple | None]:
    page = page.convert("RGB")
    if page.width != VIEW_W:  # a few pages are 1318px wide; scale to the viewport width
        s = VIEW_W / page.width
        page = page.resize((VIEW_W, round(page.height * s)))
        if box is not None:
            box = tuple(v * s for v in box)
    top = last_top
    if box is not None:
        x, y, w, h = box
        top = int(y + h / 2 - VIEW_H / 2)
    top = max(0, min(top, max(0, page.height - VIEW_H)))
    frame = Image.new("RGB", (VIEW_W, VIEW_H), "white")
    frame.paste(page.crop((0, top, VIEW_W, min(page.height, top + VIEW_H))), (0, 0))
    target = None
    if box is not None:
        x, y, w, h = box
        target = (int(x), int(y - top), int(w), int(h))
    return frame, top, target


INTERACTIVE = {"a", "button", "input", "select", "textarea", "label", "option", "li", "span", "img"}
_DESC_KEYS = (
    "role",
    "aria_label",
    "aria-label",
    "name",
    "id",
    "type",
    "placeholder",
    "value",
    "title",
    "alt",
    "href",
)


def _candidate(c: str, top: int, scale: float) -> dict | None:
    try:
        c = json.loads(c)
        attrs = json.loads(c["attributes"])
        x, y, w, h = (float(v) * scale for v in attrs["bounding_box_rect"].split(","))
    except (KeyError, ValueError, TypeError):
        return None
    y -= top
    if w < 4 or h < 4 or w * h > 0.25 * VIEW_W * VIEW_H:
        return None
    if x < 0 or y < 0 or x + w > VIEW_W or y + h > VIEW_H:  # must be fully visible
        return None
    bits = [f'{k}="{str(attrs[k])[:40]}"' for k in _DESC_KEYS if attrs.get(k)]
    return {
        "box": (int(x), int(y), int(w), int(h)),
        "tag": c.get("tag", "?"),
        "desc": " ".join(bits[:4]),
    }


def _options(
    r: dict, top: int, scale: float, rng: random.Random, n: int = 8
) -> tuple[list[dict], int | None]:
    pos = next(
        (o for o in (_candidate(c, top, scale) for c in r["pos_candidates"] or []) if o), None
    )
    if pos is None:
        return [], None
    negs = [
        o
        for o in (_candidate(c, top, scale) for c in r["neg_candidates"] or [])
        if o and o["tag"] in INTERACTIVE and o["box"] != pos["box"]
    ]
    opts = [pos, *rng.sample(negs, min(n - 1, len(negs)))]
    rng.shuffle(opts)
    return opts, opts.index(pos)


def load(
    n_shards: int = 1, min_steps: int = 3, with_candidates: bool = False, seed: int = 0
) -> list[Trajectory]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    trajs: dict[str, Trajectory] = {}
    rows: list[dict] = []
    for name in SHARDS[:n_shards]:
        path = Path(hf_hub_download(REPO, name, repo_type="dataset"))
        cols = [
            "annotation_id",
            "website",
            "confirmed_task",
            "target_action_index",
            "target_action_reprs",
            "pos_candidates",
            "screenshot",
        ] + (["neg_candidates", "operation"] if with_candidates else [])
        rows += pq.read_table(path, columns=cols).to_pylist()
    rows.sort(key=lambda r: (r["annotation_id"], int(r["target_action_index"])))
    tops: dict[str, int] = {}
    rng = random.Random(seed)
    for r in rows:
        aid = r["annotation_id"]
        shot = r["screenshot"] or {}
        if not shot.get("bytes"):  # a few rows have no screenshot
            continue
        t = trajs.setdefault(aid, Trajectory(aid, r["website"], r["confirmed_task"]))
        page = Image.open(io.BytesIO(shot["bytes"]))
        scale = VIEW_W / page.width
        frame, top, target = _viewport(page, _target_box(r["pos_candidates"]), tops.get(aid, 0))
        tops[aid] = top
        step = Step(int(r["target_action_index"]), r["target_action_reprs"], frame, target)
        if with_candidates:
            step.options, step.answer = _options(r, top, scale, rng)
            step.op = json.loads(r["operation"] or "{}").get("op", "")
        t.steps.append(step)
    return [t for t in trajs.values() if len(t.steps) >= min_steps]
