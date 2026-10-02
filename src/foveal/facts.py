"""Shared fact memory: what agents learned from assets, with provenance and freshness.

A Fact records a claim, the asset (and stream version) it came from, the pixel region
that supports it, the level it was seen at, a confidence and its author. Other agents
recall facts instead of re-perceiving the asset, and can verify one by zooming into the
cited region.

Invalidation is region-aware: when a stream gets a new version, a fact whose region is
untouched by the change carries forward (moved by any scroll); a fact whose region changed,
or that has no region, becomes stale. Subscribers are notified of changes as events.

Two interchangeable backends: SQLite (default; WAL, one file shared by processes on a
machine, events polled) and Redis (shared across machines; events also published).
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

Box = tuple[int, int, int, int]  # x, y, w, h in pixels of the asset


@dataclass
class Fact:
    claim: str
    asset_id: str
    author: str
    region: Box | None = None
    level_seen: str = "L3"  # L1 | L2 | L3
    confidence: float = 1.0
    source: str | None = None  # stream, for versioned assets (screenshots)
    asset_version: int | None = None
    status: str = "fresh"  # fresh | stale
    fact_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: float = field(default_factory=time.time)
    stale_reason: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str | bytes) -> Fact:
        d = json.loads(s)
        if d.get("region") is not None:
            d["region"] = tuple(d["region"])
        return cls(**d)


@dataclass
class Event:
    event_id: int
    kind: str  # changed | stale
    source: str
    version: int
    subscriber: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Subscription:
    agent_id: str
    source: str
    region: Box | None = None


class FactBackend(Protocol):
    def put(self, fact: Fact) -> None: ...
    def get(self, fact_id: str) -> Fact | None: ...
    def query(
        self, asset_id: str | None = None, source: str | None = None, status: str | None = None
    ) -> list[Fact]: ...
    def subscribe(self, sub: Subscription) -> None: ...
    def subscriptions(self, source: str) -> list[Subscription]: ...
    def emit(self, event: Event) -> Event: ...
    def poll(self, agent_id: str, after: int = 0) -> list[Event]: ...


# -- invalidation ---------------------------------------------------------------


def _overlaps(a: Box, b: Box) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def carry_forward(
    region: Box | None, diff: dict[str, Any], frame_h: int | None = None
) -> tuple[Box | None, str | None]:
    """Where a fact's region is in the new version, or (None, reason) if it is stale.

    `diff` is the stored diff of the new version against the fact's version.
    """
    kind = diff.get("kind")
    if kind == "identical" and not diff.get("scroll_dy"):
        return region, None
    if region is None:
        return None, f"asset changed ({kind}) and the fact has no region"
    if kind == "full":
        return None, "asset changed substantially"
    x, y, w, h = region
    dy = diff.get("scroll_dy") or 0
    band = diff.get("scroll_band")
    if dy:
        y0, y1 = band or (0, frame_h or 10**9)
        if y0 <= y and y + h <= y1:  # region scrolled with the content
            y -= dy
            if y < y0 or y + h > y1:
                return None, "region scrolled out of view"
    moved = (x, y, w, h)
    for r in diff.get("regions", []):
        if _overlaps(moved, tuple(r)):
            return None, "region changed"
    return moved, None


def region_touched(region: Box | None, diff: dict[str, Any], frame_h: int | None = None) -> bool:
    """Whether a fixed screen area shows something different in the new version.
    (Subscriptions watch screen areas; facts follow their content, see carry_forward.)"""
    kind = diff.get("kind")
    if kind == "identical" and not diff.get("scroll_dy"):
        return False
    if region is None or kind == "full":
        return True
    x, y, w, h = region
    if diff.get("scroll_dy"):
        y0, y1 = diff.get("scroll_band") or (0, frame_h or 10**9)
        if y < y1 and y + h > y0:  # part of the area is in the scrolled band
            return True
    return any(_overlaps(region, tuple(r)) for r in diff.get("regions", []))


# -- recall ---------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def rank(query: str, facts: list[Fact], k: int = 5, embed: Any = None) -> list[tuple[float, Fact]]:
    """BM25 over claims (plus cosine similarity when an `embed` callable is given)."""
    if not facts:
        return []
    docs = [tokens(f.claim) for f in facts]
    q = tokens(query)
    n = len(docs)
    avg = sum(map(len, docs)) / n or 1
    df = Counter(t for d in docs for t in set(d))
    scored = []
    qv = embed(query) if embed else None
    for f, d in zip(facts, docs, strict=True):
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avg))
        if qv is not None:
            fv = embed(f.claim)
            dot = sum(a * b for a, b in zip(qv, fv, strict=True))
            norm = math.sqrt(sum(a * a for a in qv) * sum(b * b for b in fv)) or 1
            s += 3.0 * dot / norm
        s *= 0.5 + 0.5 * f.confidence
        if s > 0:
            scored.append((s, f))
    scored.sort(key=lambda x: -x[0])
    return scored[:k]


# -- SQLite backend -------------------------------------------------------------

_FACTS_SQL = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id TEXT PRIMARY KEY, asset_id TEXT, source TEXT, status TEXT, body TEXT
);
CREATE INDEX IF NOT EXISTS facts_asset ON facts(asset_id);
CREATE INDEX IF NOT EXISTS facts_source ON facts(source);
CREATE TABLE IF NOT EXISTS subscriptions (
    agent_id TEXT, source TEXT, region TEXT, PRIMARY KEY (agent_id, source, region)
);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, source TEXT, version INTEGER,
    subscriber TEXT, data TEXT, created_at REAL
);
"""


class SQLiteFacts:
    """Facts, subscriptions and an append-only event log in the store's SQLite file.
    Other processes on the machine open the same file and `poll` for events."""

    def __init__(self, path: str | Path):
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_FACTS_SQL)
        self._lock = threading.Lock()

    def put(self, fact: Fact) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO facts VALUES (?, ?, ?, ?, ?)",
                (fact.fact_id, fact.asset_id, fact.source, fact.status, fact.to_json()),
            )
            self._db.commit()

    def get(self, fact_id: str) -> Fact | None:
        with self._lock:
            row = self._db.execute("SELECT body FROM facts WHERE fact_id=?", (fact_id,)).fetchone()
        return Fact.from_json(row[0]) if row else None

    def query(
        self, asset_id: str | None = None, source: str | None = None, status: str | None = None
    ) -> list[Fact]:
        where, args = [], []
        for col, val in (("asset_id", asset_id), ("source", source), ("status", status)):
            if val is not None:
                where.append(f"{col}=?")
                args.append(val)
        sql = "SELECT body FROM facts" + (" WHERE " + " AND ".join(where) if where else "")
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [Fact.from_json(r[0]) for r in rows]

    def subscribe(self, sub: Subscription) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO subscriptions VALUES (?, ?, ?)",
                (sub.agent_id, sub.source, json.dumps(sub.region)),
            )
            self._db.commit()

    def subscriptions(self, source: str) -> list[Subscription]:
        with self._lock:
            rows = self._db.execute(
                "SELECT agent_id, source, region FROM subscriptions WHERE source=?", (source,)
            ).fetchall()
        out = []
        for a, s, r in rows:
            reg = json.loads(r)
            out.append(Subscription(a, s, tuple(reg) if reg else None))
        return out

    def emit(self, event: Event) -> Event:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO events (kind, source, version, subscriber, data, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.kind,
                    event.source,
                    event.version,
                    event.subscriber,
                    json.dumps(event.data),
                    time.time(),
                ),
            )
            self._db.commit()
            event.event_id = cur.lastrowid
        return event

    def poll(self, agent_id: str, after: int = 0) -> list[Event]:
        with self._lock:
            rows = self._db.execute(
                "SELECT event_id, kind, source, version, subscriber, data FROM events "
                "WHERE event_id > ? AND (subscriber = ? OR subscriber IS NULL) "
                "ORDER BY event_id",
                (after, agent_id),
            ).fetchall()
        return [Event(r[0], r[1], r[2], r[3], r[4], json.loads(r[5])) for r in rows]


# -- Redis backend --------------------------------------------------------------


class RedisFacts:
    """Same interface on Redis: facts as JSON strings with set indexes, events in a stream
    (durable, pollable) and also published on a channel for push subscribers."""

    def __init__(self, url: str = "redis://localhost:6379/0", prefix: str = "foveal"):
        import redis

        self.r = redis.Redis.from_url(url)
        self.p = prefix

    def _k(self, *parts: str) -> str:
        return ":".join((self.p, *parts))

    def put(self, fact: Fact) -> None:
        old = self.get(fact.fact_id)
        pipe = self.r.pipeline()
        if old is not None:
            pipe.srem(self._k("status", old.status), fact.fact_id)
        pipe.set(self._k("fact", fact.fact_id), fact.to_json())
        pipe.sadd(self._k("asset", fact.asset_id), fact.fact_id)
        if fact.source:
            pipe.sadd(self._k("source", fact.source), fact.fact_id)
        pipe.sadd(self._k("status", fact.status), fact.fact_id)
        pipe.sadd(self._k("all"), fact.fact_id)
        pipe.execute()

    def get(self, fact_id: str) -> Fact | None:
        raw = self.r.get(self._k("fact", fact_id))
        return Fact.from_json(raw) if raw else None

    def query(
        self, asset_id: str | None = None, source: str | None = None, status: str | None = None
    ) -> list[Fact]:
        keys = [self._k("asset", asset_id)] if asset_id else []
        keys += [self._k("source", source)] if source else []
        keys += [self._k("status", status)] if status else []
        ids = self.r.sinter(keys) if keys else self.r.smembers(self._k("all"))
        if not ids:
            return []
        raws = self.r.mget([self._k("fact", i.decode()) for i in ids])
        return [Fact.from_json(x) for x in raws if x]

    def subscribe(self, sub: Subscription) -> None:
        self.r.sadd(
            self._k("subs", sub.source),
            json.dumps({"agent_id": sub.agent_id, "region": sub.region}),
        )

    def subscriptions(self, source: str) -> list[Subscription]:
        out = []
        for raw in self.r.smembers(self._k("subs", source)):
            d = json.loads(raw)
            out.append(
                Subscription(d["agent_id"], source, tuple(d["region"]) if d["region"] else None)
            )
        return out

    def emit(self, event: Event) -> Event:
        payload = {
            "kind": event.kind,
            "source": event.source,
            "version": event.version,
            "subscriber": event.subscriber or "",
            "data": json.dumps(event.data),
        }
        sid = self.r.xadd(self._k("events"), payload)
        ms, seq = sid.decode().split("-")
        event.event_id = int(ms) * 1000 + int(seq)
        self.r.publish(self._k("channel"), json.dumps({**payload, "id": sid.decode()}))
        return event

    def poll(self, agent_id: str, after: int = 0) -> list[Event]:
        start = f"{after // 1000}-{after % 1000 + 1}" if after else "0-0"
        out = []
        for sid, f in self.r.xrange(self._k("events"), min=start):
            ms, seq = sid.decode().split("-")
            sub = f[b"subscriber"].decode() or None
            if sub not in (None, agent_id):
                continue
            out.append(
                Event(
                    int(ms) * 1000 + int(seq),
                    f[b"kind"].decode(),
                    f[b"source"].decode(),
                    int(f[b"version"]),
                    sub,
                    json.loads(f[b"data"]),
                )
            )
        return out
