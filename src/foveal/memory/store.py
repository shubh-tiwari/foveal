"""Content-addressed asset store: SQLite metadata plus image files on disk."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Asset:
    asset_id: str  # sha256 of the L3 bytes
    kind: str  # image | page | screenshot | frame
    source: str  # file path, URL, window id ...
    version: int = 1
    parent_version: int | None = None
    width: int | None = None
    height: int | None = None
    levels: dict[str, Any] = field(default_factory=dict)  # L0 caption, L1 text, L2/L3 paths
    origin: dict[str, Any] = field(default_factory=dict)  # how to re-render, e.g. pdf + page
    changed_regions: list[Any] = field(default_factory=list)  # Phase 2 (diff engine)

    @property
    def caption(self) -> str | None:
        return self.levels.get("L0")

    @property
    def text(self) -> str | None:
        return self.levels.get("L1")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY, kind TEXT, source TEXT, version INTEGER, parent_version INTEGER,
    width INTEGER, height INTEGER, levels TEXT, origin TEXT, changed_regions TEXT,
    created_at REAL
)
"""

# A stream (window, tab, camera) is a `source`; each new observation of it is a version.
# Content stays deduplicated in `assets`; versions point at it.
_VERSIONS = """
CREATE TABLE IF NOT EXISTS versions (
    source TEXT, version INTEGER, asset_id TEXT, parent_version INTEGER, diff TEXT,
    created_at REAL, PRIMARY KEY (source, version)
)
"""


@dataclass
class VersionRow:
    source: str
    version: int
    asset_id: str
    parent_version: int | None
    diff: dict[str, Any]


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "blobs").mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.root / "foveal.sqlite", check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")  # readers in other processes don't block
        self._db.execute(_SCHEMA)
        self._db.execute(_VERSIONS)
        self._lock = threading.Lock()

    def blob_path(self, sha: str, suffix: str = ".jpg") -> Path:
        d = self.root / "blobs" / sha[:2]
        d.mkdir(exist_ok=True)
        return d / f"{sha}{suffix}"

    def write_blob(self, sha: str, data: bytes, suffix: str = ".jpg") -> Path:
        path = self.blob_path(sha, suffix)
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return path

    def get(self, asset_id: str) -> Asset | None:
        with self._lock:
            row = self._db.execute(
                "SELECT asset_id, kind, source, version, parent_version, width, height, levels,"
                " origin, changed_regions FROM assets WHERE asset_id=?",
                (asset_id,),
            ).fetchone()
        if row is None:
            return None
        return Asset(
            asset_id=row[0],
            kind=row[1],
            source=row[2],
            version=row[3],
            parent_version=row[4],
            width=row[5],
            height=row[6],
            levels=json.loads(row[7]),
            origin=json.loads(row[8]),
            changed_regions=json.loads(row[9]),
        )

    def put(self, a: Asset) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    a.asset_id,
                    a.kind,
                    a.source,
                    a.version,
                    a.parent_version,
                    a.width,
                    a.height,
                    json.dumps(a.levels),
                    json.dumps(a.origin),
                    json.dumps(a.changed_regions),
                    time.time(),
                ),
            )
            self._db.commit()

    def update_levels(self, asset_id: str, **levels: Any) -> None:
        a = self.get(asset_id)
        if a is None:
            raise KeyError(asset_id)
        a.levels.update(levels)
        self.put(a)

    def find_prefix(self, prefix: str) -> str | None:
        """Full asset id for a unique id prefix (as shown to models), else None."""
        if len(prefix) < 6:
            return None
        with self._lock:
            rows = self._db.execute(
                "SELECT asset_id FROM assets WHERE substr(asset_id, 1, ?) = ? LIMIT 2",
                (len(prefix), prefix),
            ).fetchall()
        return rows[0][0] if len(rows) == 1 else None

    # -- versions -----------------------------------------------------------

    def add_version(self, source: str, asset_id: str, diff: dict[str, Any]) -> VersionRow:
        with self._lock:
            row = self._db.execute(
                "SELECT MAX(version) FROM versions WHERE source=?", (source,)
            ).fetchone()
            parent = row[0]
            version = (parent or 0) + 1
            self._db.execute(
                "INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?)",
                (source, version, asset_id, parent, json.dumps(diff), time.time()),
            )
            self._db.commit()
        return VersionRow(source, version, asset_id, parent, diff)

    def _version_rows(self, sql: str, args: tuple) -> list[VersionRow]:
        with self._lock:
            rows = self._db.execute(
                "SELECT source, version, asset_id, parent_version, diff FROM versions " + sql, args
            ).fetchall()
        return [VersionRow(r[0], r[1], r[2], r[3], json.loads(r[4])) for r in rows]

    def latest_version(self, source: str) -> VersionRow | None:
        rows = self._version_rows("WHERE source=? ORDER BY version DESC LIMIT 1", (source,))
        return rows[0] if rows else None

    def get_version(self, source: str, version: int) -> VersionRow | None:
        rows = self._version_rows("WHERE source=? AND version=?", (source, version))
        return rows[0] if rows else None

    def versions(self, source: str) -> list[VersionRow]:
        return self._version_rows("WHERE source=? ORDER BY version", (source,))

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
