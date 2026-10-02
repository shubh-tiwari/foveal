"""foveal as an MCP server: any MCP client (Claude Code, Claude Desktop, IDEs, custom agents)
gets perceive-once memory as tools.

    foveal-mcp --store .foveal                 # stdio
    foveal-mcp --store .foveal --redis-url redis://localhost:6379/0   # facts shared via Redis

Tools: ingest, ingest_frame, describe, look, diff_since, recall_facts, write_fact,
verify_fact, subscribe, poll. Asset ids may be given as unique prefixes (12 characters, as
shown in tool output).
"""

from __future__ import annotations

import argparse
import base64
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from foveal.memory import Memory

ALLOWED = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf"}

INSTRUCTIONS = """foveal keeps images, screenshots and document pages so you see each one once.
ingest a file (or ingest_frame a screenshot of a window/tab) to get an asset id, read its
caption and text with describe, and look at pixels only when needed (glimpse first, then
full on a region). For screens, diff_since shows only what changed. Record what you learn
with write_fact; check recall_facts before re-reading something another agent may have read."""


def build_server(memory: Memory) -> Any:
    from mcp.server.mcpserver import Image, MCPServer

    server = MCPServer("foveal", instructions=INSTRUCTIONS)

    def resolve(asset_id: str) -> str:
        if len(asset_id) == 64:
            return asset_id
        full = memory.store.find_prefix(asset_id)
        if full is None:
            raise ValueError(f"no unique asset starts with {asset_id!r}")
        return full

    def image(view: Any) -> Any:
        return Image(data=view.jpeg, format="jpeg")

    @server.tool()
    def ingest(path: str) -> str:
        """Store an image or PDF once (deduplicated by content). Returns asset ids."""
        p = Path(path).expanduser()
        if p.suffix.lower() not in ALLOWED or not p.is_file():
            raise ValueError(f"expected an existing image or PDF file, got {path!r}")
        out = memory.ingest(p)
        assets = out if isinstance(out, list) else [out]
        return "\n".join(
            f"{a.asset_id[:12]} {a.kind} {a.width}x{a.height} {a.source}" for a in assets
        )

    @server.tool()
    def ingest_frame(data_base64: str, source: str) -> str:
        """Record a new screenshot of `source` (a window, tab or camera). Unchanged frames
        create no version; changed ones become version N+1 and invalidate stale facts."""
        a = memory.ingest_frame(base64.b64decode(data_base64), source)
        d = a.origin.get("diff", {})
        return (
            f"{a.asset_id[:12]} {source} version {a.version} "
            f"({d.get('kind', 'new')}, {len(a.changed_regions)} changed regions)"
        )

    @server.tool()
    def describe(asset_id: str, include_text: bool = True) -> str:
        """Caption (L0) and, optionally, text layer or OCR (L1) of an asset. No image tokens."""
        levels = ("L0", "L1") if include_text else ("L0",)
        return memory.describe(resolve(asset_id), levels=levels)

    @server.tool(structured_output=False)  # mixed text + image content
    def look(
        asset_id: str, detail: str = "glimpse", region: list[float] | None = None
    ) -> list[Any]:
        """See an asset: detail 'glimpse' (~256 tokens) or 'full'; region [x0, y0, x1, y1]
        as fractions zooms in (PDF pages are re-rendered at higher resolution)."""
        v = memory.look(resolve(asset_id), region=region, detail=detail)
        return [f"{v.asset_id[:12]} {detail} {v.width}x{v.height} (~{v.tokens} tokens)", image(v)]

    @server.tool(structured_output=False)  # mixed text + image content
    def diff_since(source: str, version: int) -> list[Any]:
        """What changed in `source` since `version`: a description plus crops."""
        dv = memory.diff_since(source, version)
        out: list[Any] = [dv.text]
        for v in dv.views:
            if v.region is not None:
                out.append("region " + ", ".join(f"{x:.2f}" for x in v.region))
            out.append(image(v))
        return out

    @server.tool()
    def recall_facts(
        query: str,
        asset_id: str | None = None,
        source: str | None = None,
        include_stale: bool = False,
        k: int = 5,
    ) -> str:
        """Facts other agents recorded, relevant to `query`, with provenance."""
        facts = memory.recall_facts(
            query,
            asset_id=resolve(asset_id) if asset_id else None,
            source=source,
            include_stale=include_stale,
            k=k,
        )
        return (
            json.dumps([{**asdict(f), "asset_id": f.asset_id[:12]} for f in facts], indent=1)
            or "[]"
        )

    @server.tool()
    def write_fact(
        asset_id: str,
        claim: str,
        region: list[int] | None = None,
        confidence: float = 1.0,
        author: str = "agent",
        source: str | None = None,
        level_seen: str = "L3",
    ) -> str:
        """Record what you learned from an asset. region is a pixel box [x, y, w, h]."""
        f = memory.write_fact(
            resolve(asset_id),
            claim,
            region=region,
            confidence=confidence,
            author=author,
            level_seen=level_seen,
            source=source,
        )
        return f"fact {f.fact_id}"

    @server.tool(structured_output=False)  # mixed text + image content
    def verify_fact(fact_id: str) -> list[Any]:
        """Zoom into the region a fact cites, to check it."""
        v = memory.verify_fact(fact_id)
        return [f"evidence for fact {fact_id}", image(v)]

    @server.tool()
    def subscribe(agent_id: str, source: str, region: list[int] | None = None) -> str:
        """Get `changed` events (via poll) when `source`, or a pixel region of it, changes."""
        memory.subscribe(agent_id, source, region)
        return f"{agent_id} subscribed to {source}" + (f" region {region}" if region else "")

    @server.tool()
    def poll(agent_id: str, after: int = 0) -> str:
        """Events for `agent_id` newer than event id `after`: changed screens, stale facts."""
        return json.dumps([asdict(e) for e in memory.poll(agent_id, after)], indent=1)

    return server


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="foveal-mcp")
    ap.add_argument("--store", default=".foveal")
    ap.add_argument("--redis-url", default=None)
    ap.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    args = ap.parse_args(argv)
    facts = None
    if args.redis_url:
        from foveal.facts import RedisFacts

        facts = RedisFacts(args.redis_url)
    build_server(Memory(args.store, facts=facts)).run(args.transport)


if __name__ == "__main__":
    main()
