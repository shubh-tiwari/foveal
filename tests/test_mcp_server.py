import asyncio
import base64
import io

import pytest

from foveal import Memory
from tests.test_diff import page, view

mcp = pytest.importorskip("mcp")


def _png(im):
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _text(result):
    return "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")


def test_tools_end_to_end(tmp_path):
    from foveal.mcp_server import build_server

    server = build_server(Memory(tmp_path / "s"))

    async def run():
        names = {t.name for t in await server.list_tools()}
        assert {"ingest", "look", "diff_since", "recall_facts", "write_fact", "poll"} <= names
        full = page(seed=51)
        r1 = await server.call_tool(
            "ingest_frame", {"data_base64": _png(view(full, 0)), "source": "tab:1"}
        )
        asset = _text(r1).split()[0]
        await server.call_tool("subscribe", {"agent_id": "orch", "source": "tab:1"})
        f = await server.call_tool(
            "write_fact",
            {
                "asset_id": asset,
                "claim": "header says Expedia",
                "region": [0, 0, 400, 60],
                "source": "tab:1",
            },
        )
        assert "fact" in _text(f)
        look = await server.call_tool("look", {"asset_id": asset, "detail": "glimpse"})
        assert any(getattr(c, "type", "") == "image" for c in look.content)
        r2 = await server.call_tool(
            "ingest_frame", {"data_base64": _png(view(full, 200)), "source": "tab:1"}
        )
        assert "version 2" in _text(r2)
        d = await server.call_tool("diff_since", {"source": "tab:1", "version": 1})
        assert "scrolled down 200px" in _text(d)
        ev = await server.call_tool("poll", {"agent_id": "orch"})
        assert '"changed"' in _text(ev)
        facts = await server.call_tool("recall_facts", {"query": "header", "include_stale": True})
        assert "Expedia" in _text(facts)

    asyncio.run(run())
