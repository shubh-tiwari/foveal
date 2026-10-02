"""Rule-based answer scoring, adapted from MMLongBench-Doc's answer formats.

The official eval first extracts a short answer with an LLM; here the orchestrator is
asked for a short answer directly, and these rules compare it to the gold answer.
"""

from __future__ import annotations

import ast
import re
from difflib import SequenceMatcher

_UNANSWERABLE = (
    "not answerable",
    "cannot be answered",
    "not in the document",
    "no information",
    "not mentioned",
    "unanswerable",
    "not provided",
    "cannot be determined",
    "fail to answer",
)


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\"'`*_]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .")


def _numbers(s: str) -> list[float]:
    out = []
    for m in re.findall(r"-?\d[\d,]*\.?\d*", s):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _str_match(gold: str, pred: str) -> bool:
    g, p = _norm(gold), _norm(pred)
    if not g:
        return False
    if g == p or g in p or (len(p) >= 3 and p in g):
        return True
    return SequenceMatcher(None, g, p).ratio() >= 0.8


def _num_match(gold: float, preds: list[float], rel: float = 0.01) -> bool:
    for v in preds:
        if gold == v or abs(gold - v) <= rel * max(abs(gold), 1e-9):
            return True
        if abs(gold) < 1 and abs(gold * 100 - v) <= rel * abs(gold * 100):  # 0.25 vs 25%
            return True
    return False


def score(gold: str, pred: str | None, fmt: str, question: str = "") -> float:
    if pred is None:
        return 0.0
    p = pred.strip()
    unanswerable = any(k in p.lower() for k in _UNANSWERABLE)
    if fmt == "None":
        return float(unanswerable)
    if unanswerable:
        return 0.0
    if fmt == "Int":
        nums = _numbers(p)
        try:
            return float(int(float(gold.replace(",", ""))) in [int(n) for n in nums if n == int(n)])
        except ValueError:
            return float(_str_match(gold, p))
    if fmt == "Float":
        g = _numbers(gold)
        return float(bool(g) and _num_match(g[0], _numbers(p)))
    if fmt == "Str" and gold.strip().startswith("["):  # some Str golds are list literals
        fmt = "List"
    if fmt == "List":
        try:
            items = ast.literal_eval(gold)
            items = items if isinstance(items, (list, tuple)) else [gold]
        except (ValueError, SyntaxError):
            items = [x for x in re.split(r",\s*", gold.strip("[]")) if x]
        ok = all(
            _num_match(float(i), _numbers(p))
            if isinstance(i, (int, float))
            else _str_match(str(i), p) or _norm(str(i)) in _norm(p)
            for i in items
        )
        return float(ok)
    return float(_str_match(gold, p))
