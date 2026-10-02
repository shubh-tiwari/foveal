"""Phase 0 metrics computed from a run's JSONL log.

Every image block in every call is classified, per task, as:
  first_seen   - the first time this exact image (sha256) was sent in the task
  same_agent   - re-sent by an agent that already sent it (usually history resend)
  cross_agent  - first time *this* agent sends it, but another agent already did
Re-perception rate = (same_agent + cross_agent) / all image tokens.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from foveal.instrument.hashing import phash_distance
from foveal.instrument.pricing import prices_for

CATEGORIES = ("first_seen", "same_agent", "cross_agent")


def _tok(img: dict[str, Any]) -> int:
    t = img.get("tokens")
    return t if t is not None else (img.get("est_tokens") or 0)


@dataclass
class TaskMetrics:
    task_id: str
    doc_id: str | None
    n_calls: int = 0
    tokens: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CATEGORIES, 0))
    near_dup_tokens: int = 0  # first_seen tokens whose pHash is within the threshold
    cross_task_tokens: int = 0  # first_seen tokens already seen in an earlier task (same doc)
    keep_last_n_tokens: int = 0
    repeated_cached_tokens: int = 0  # repeated tokens plausibly billed as cache reads
    repeated_cost_no_cache: float = 0.0  # USD, priced per call's own model
    repeated_cost_est: float = 0.0
    cumulative: list[tuple[int, dict[str, int]]] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_s: float = 0.0
    unique_images: int = 0
    score: float | None = None

    @property
    def total(self) -> int:
        return sum(self.tokens.values())

    @property
    def repeated(self) -> int:
        return self.tokens["same_agent"] + self.tokens["cross_agent"]

    @property
    def reperception_rate(self) -> float:
        return self.repeated / self.total if self.total else 0.0


def _keep_last_n(images: list[dict[str, Any]], n: int) -> int:
    """Image tokens if the harness only kept images from the last `n` image-bearing messages."""
    by_msg: dict[int, int] = defaultdict(int)
    for img in images:
        by_msg[img["msg_index"]] += _tok(img)
    keep = sorted(by_msg)[-n:] if n > 0 else []
    return sum(by_msg[i] for i in keep)


def compute(
    records: list[dict[str, Any]], keep_last: int = 3, phash_threshold: int = 4
) -> dict[str, Any]:
    calls = sorted((r for r in records if r.get("type") == "call"), key=lambda r: r["call_index"])
    results = {r["task_id"]: r for r in records if r.get("type") == "task"}

    tasks: dict[str, TaskMetrics] = {}
    seen_in_task: dict[str, dict[str, set[str]]] = defaultdict(dict)  # task -> sha -> agents
    phashes_in_task: dict[str, list[str]] = defaultdict(list)
    task_order: list[str] = []
    per_agent: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(CATEGORIES, 0))

    for call in calls:
        tid = call.get("task_id") or "_untasked"
        doc = (call.get("meta") or {}).get("doc_id")
        if tid not in tasks:
            tasks[tid] = TaskMetrics(task_id=tid, doc_id=doc)
            task_order.append(tid)
        tm = tasks[tid]
        tm.n_calls += 1
        tm.cost_usd += (call.get("meta") or {}).get("cost_usd", 0.0)
        tm.latency_s += call.get("latency_s", 0.0)
        agent = call.get("agent_id") or "_agent"
        role = agent.split("-")[0]
        imgs = [i for i in call.get("images", []) if i.get("kind") == "image"]
        seen = seen_in_task[tid]
        call_repeated = 0
        for img in imgs:
            t = _tok(img)
            sha = img["sha256"]
            if sha not in seen:
                cat = "first_seen"
                seen[sha] = {agent}
                ph = img.get("phash")
                if ph and any(
                    phash_distance(ph, p) <= phash_threshold for p in phashes_in_task[tid]
                ):
                    tm.near_dup_tokens += t
                if ph:
                    phashes_in_task[tid].append(ph)
            elif agent in seen[sha]:
                cat = "same_agent"
            else:
                cat = "cross_agent"
                seen[sha].add(agent)
            tm.tokens[cat] += t
            per_agent[role][cat] += t
            if cat != "first_seen":
                call_repeated += t
        cache_read = (call.get("usage") or {}).get("cache_read_input_tokens") or 0
        cached_here = min(call_repeated, cache_read)
        tm.repeated_cached_tokens += cached_here
        inp, _, cread = prices_for(call.get("model", ""))
        tm.repeated_cost_no_cache += call_repeated * inp / 1e6
        tm.repeated_cost_est += (cached_here * cread + (call_repeated - cached_here) * inp) / 1e6
        tm.keep_last_n_tokens += _keep_last_n(imgs, keep_last)
        tm.cumulative.append((call["call_index"], dict(tm.tokens)))

    for tid in task_order:
        tm = tasks[tid]
        tm.unique_images = len(seen_in_task[tid])
        if tid in results:
            tm.score = results[tid].get("score")

    # Cross-task: recompute in task order so "earlier task on same doc" is well defined.
    doc_seen: dict[str, set[str]] = defaultdict(set)
    first_tokens: dict[tuple[str, str], int] = {}
    for call in calls:
        tid = call.get("task_id") or "_untasked"
        for img in call.get("images", []):
            if img.get("kind") == "image":
                first_tokens.setdefault((tid, img["sha256"]), _tok(img))
    for tid in task_order:
        tm = tasks[tid]
        shas = seen_in_task[tid]
        if tm.doc_id:
            tm.cross_task_tokens = sum(
                first_tokens[(tid, s)] for s in shas if s in doc_seen[tm.doc_id]
            )
            doc_seen[tm.doc_id].update(shas)

    return summarize(list(tasks.values()), per_agent, calls, keep_last)


def summarize(
    tasks: list[TaskMetrics],
    per_agent: dict[str, dict[str, int]],
    calls: list[dict[str, Any]],
    keep_last: int,
) -> dict[str, Any]:
    total = sum(t.total for t in tasks)
    repeated = sum(t.repeated for t in tasks)
    first = sum(t.tokens["first_seen"] for t in tasks)
    keep_n = sum(t.keep_last_n_tokens for t in tasks)
    cached = sum(t.repeated_cached_tokens for t in tasks)
    scored = [t.score for t in tasks if t.score is not None]
    usage_tot: dict[str, int] = defaultdict(int)
    for c in calls:
        for k in (
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "output_tokens",
        ):
            usage_tot[k] += (c.get("usage") or {}).get(k) or 0
    return {
        "n_tasks": len(tasks),
        "n_calls": len(calls),
        "image_tokens_total": total,
        "image_tokens_per_task_median": median([t.total for t in tasks]) if tasks else 0,
        "image_tokens_per_task_mean": total / len(tasks) if tasks else 0,
        "reperception_rate": repeated / total if total else 0.0,
        "same_agent_share": sum(t.tokens["same_agent"] for t in tasks) / total if total else 0.0,
        "cross_agent_share": sum(t.tokens["cross_agent"] for t in tasks) / total if total else 0.0,
        "near_dup_share_of_first_seen": (
            sum(t.near_dup_tokens for t in tasks) / first if first else 0.0
        ),
        "cross_task_share_of_first_seen": (
            sum(t.cross_task_tokens for t in tasks) / first if first else 0.0
        ),
        f"keep_last_{keep_last}_tokens": keep_n,
        f"reperception_under_keep_last_{keep_last}": (keep_n - first) / keep_n if keep_n else 0.0,
        "repeated_tokens_cached_share": cached / repeated if repeated else 0.0,
        "repeated_cost_usd_no_cache": sum(t.repeated_cost_no_cache for t in tasks),
        "repeated_cost_usd_est_actual": sum(t.repeated_cost_est for t in tasks),
        "success_rate": sum(scored) / len(scored) if scored else None,
        "cost_usd_total": sum(t.cost_usd for t in tasks),
        "latency_s_per_call_mean": (
            sum(c.get("latency_s", 0) for c in calls) / len(calls) if calls else 0.0
        ),
        "usage_totals": dict(usage_tot),
        "per_agent_role": {k: dict(v) for k, v in per_agent.items()},
        "tasks": tasks,
    }


def gate(summary: dict[str, Any], threshold: float = 0.40) -> tuple[bool, str]:
    rate = summary["reperception_rate"]
    ok = rate >= threshold
    verdict = "PASS" if ok else "FAIL"
    return ok, f"{verdict}: re-perception {rate:.1%} vs gate {threshold:.0%}"
