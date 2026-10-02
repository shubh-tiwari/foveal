"""Web-agent trajectories from Multimodal-Mind2Web, replayed as viewport screenshots.

Mind2Web stores a full-page screenshot per action. An agent sees a viewport, so each step
becomes a 1280x720 crop scrolled to show that step's target element (or kept at the last
scroll position when the target is unknown).
"""

from __future__ import annotations

import io
import json
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
        page = page.resize((VIEW_W, round(page.height * VIEW_W / page.width)))
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


def load(n_shards: int = 1, min_steps: int = 3) -> list[Trajectory]:
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
        ]
        rows += pq.read_table(path, columns=cols).to_pylist()
    rows.sort(key=lambda r: (r["annotation_id"], int(r["target_action_index"])))
    tops: dict[str, int] = {}
    for r in rows:
        aid = r["annotation_id"]
        shot = r["screenshot"] or {}
        if not shot.get("bytes"):  # a few rows have no screenshot
            continue
        t = trajs.setdefault(aid, Trajectory(aid, r["website"], r["confirmed_task"]))
        page = Image.open(io.BytesIO(shot["bytes"]))
        frame, top, target = _viewport(page, _target_box(r["pos_candidates"]), tops.get(aid, 0))
        tops[aid] = top
        t.steps.append(Step(int(r["target_action_index"]), r["target_action_reprs"], frame, target))
    return [t for t in trajs.values() if len(t.steps) >= min_steps]
