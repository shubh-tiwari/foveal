import multiprocessing as mp
import shutil
import socket
import subprocess
import time

import pytest
from PIL import ImageDraw

from foveal import Memory
from foveal.facts import RedisFacts, carry_forward
from tests.test_diff import page, view


@pytest.fixture(scope="module")
def redis_url():
    exe = shutil.which("redis-server")
    if exe is None:
        pytest.skip("redis-server not installed")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [exe, "--port", str(port), "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"redis://127.0.0.1:{port}/0"
    import redis

    for _ in range(50):
        try:
            redis.Redis.from_url(url).ping()
            break
        except redis.ConnectionError:
            time.sleep(0.1)
    yield url
    proc.terminate()
    proc.wait()


@pytest.fixture(params=["sqlite", "redis"])
def mem(request, tmp_path):
    if request.param == "sqlite":
        return Memory(tmp_path / "store")
    url = request.getfixturevalue("redis_url")
    return Memory(tmp_path / "store", facts=RedisFacts(url, prefix=f"t{time.time_ns()}"))


def test_write_recall_verify(mem, png_factory):
    a = mem.ingest(png_factory(w=600, h=400, seed=1), text="invoice", use_ocr=False)
    f = mem.write_fact(
        a.asset_id,
        "invoice total is 42,300 INR",
        region=(40, 300, 200, 40),
        author="reader-1",
        confidence=0.9,
    )
    mem.write_fact(a.asset_id, "invoice date is 4 April 2023", author="reader-2")
    hits = mem.recall_facts("what is the invoice total?")
    assert hits[0].fact_id == f.fact_id and hits[0].author == "reader-1"
    view_ = mem.verify_fact(f.fact_id)
    assert view_.region == pytest.approx((40 / 600, 300 / 400, 240 / 600, 340 / 400))


def test_region_aware_invalidation_and_events(mem):
    full = page(seed=21)
    src = "tab:checkout"
    a1 = mem.ingest_frame(view(full, 0), src)
    total = mem.write_fact(
        a1.asset_id,
        "order total shown is $120",
        region=(100, 400, 300, 40),
        author="reader-1",
        source=src,
    )
    vague = mem.write_fact(a1.asset_id, "page is the checkout form", author="reader-2", source=src)
    mem.subscribe("orch", src, region=(100, 400, 300, 40))

    f2 = view(full, 0).copy()  # a change elsewhere on the screen
    ImageDraw.Draw(f2).rectangle([900, 100, 1100, 140], fill=(0, 140, 60))
    mem.ingest_frame(f2, src)
    kept = mem.facts.get(total.fact_id)
    assert kept.status == "fresh" and kept.asset_version == 2 and kept.region == (100, 400, 300, 40)
    assert mem.facts.get(vague.fact_id).status == "stale"  # no region: any change invalidates
    assert [e.kind for e in mem.poll("orch")] == []  # subscribed region untouched

    f3 = view(full, 120).copy()  # scroll by 120px; the toast moves with the page
    ImageDraw.Draw(f3).rectangle([900, -20, 1100, 20], fill=(0, 140, 60))
    mem.ingest_frame(f3, src)
    moved = mem.facts.get(total.fact_id)
    assert moved.status == "fresh" and moved.region == (100, 280, 300, 40)

    f4 = f3.copy()  # now the total itself changes
    ImageDraw.Draw(f4).rectangle([120, 285, 380, 315], fill=(200, 0, 0))
    mem.ingest_frame(f4, src)
    gone = mem.facts.get(total.fact_id)
    assert gone.status == "stale" and gone.stale_reason == "region changed"
    events = mem.poll("reader-1")
    assert any(e.kind == "stale" and e.data["fact_id"] == total.fact_id for e in events)
    assert mem.recall_facts("order total") == []
    assert mem.recall_facts("order total", include_stale=True)[0].fact_id == total.fact_id
    changed = [e for e in mem.poll("orch") if e.kind == "changed"]
    assert changed and all(e.subscriber == "orch" for e in changed)


def test_carry_forward_rules():
    assert carry_forward((0, 0, 10, 10), {"kind": "full"})[1] == "asset changed substantially"
    assert (
        carry_forward((0, 500, 10, 10), {"kind": "partial", "scroll_dy": 600, "regions": []}, 720)[
            1
        ]
        == "region scrolled out of view"
    )
    assert carry_forward(
        (0, 50, 10, 10),
        {"kind": "partial", "scroll_dy": 100, "scroll_band": [80, 720], "regions": []},
        720,
    ) == ((0, 50, 10, 10), None)  # inside the sticky header: stays put


def _other_process(root, q):
    m = Memory(root)
    m.subscribe("watcher", "window:A")
    q.put("subscribed")
    deadline = time.time() + 10
    while time.time() < deadline:
        ev = m.poll("watcher")
        if ev:
            q.put(ev[0].kind)
            return
        time.sleep(0.05)
    q.put("timeout")


def test_sqlite_events_cross_process(tmp_path):
    root = tmp_path / "shared"
    mem = Memory(root)
    full = page(seed=31)
    mem.ingest_frame(view(full, 0), "window:A")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_other_process, args=(root, q))
    p.start()
    assert q.get(timeout=20) == "subscribed"
    mem.ingest_frame(view(full, 200), "window:A")
    assert q.get(timeout=20) == "changed"
    p.join(5)
