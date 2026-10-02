import numpy as np
from PIL import ImageDraw

from foveal import Memory
from tests.test_diff import page, view


def test_versions_and_diff_since(tmp_path):
    mem = Memory(tmp_path / "store")
    full = page(seed=11)
    f1 = view(full, 0)
    a1 = mem.ingest_frame(f1, source="tab:checkout")
    assert (a1.version, a1.parent_version) == (1, None)
    assert mem.ingest_frame(f1.copy(), source="tab:checkout").version == 1  # unchanged

    f2 = f1.copy()
    ImageDraw.Draw(f2).rectangle([900, 500, 1100, 560], fill=(0, 140, 60))  # a toast appears
    a2 = mem.ingest_frame(f2, source="tab:checkout")
    assert (a2.version, a2.parent_version) == (2, 1)
    assert a2.origin["diff"]["kind"] == "partial" and len(a2.changed_regions) == 1

    f3 = view(full, 250)  # scrolled
    mem.ingest_frame(f3, source="tab:checkout")
    assert [a.version for a in mem.history("tab:checkout")] == [1, 2, 3]
    assert mem.latest("tab:checkout").origin["diff"]["scroll_dy"] == 250

    dv = mem.diff_since("tab:checkout", 2)
    assert dv.from_version == 2 and dv.to_version == 3
    assert "scrolled down 250px" in dv.text
    full_tokens = mem.look(mem.latest("tab:checkout").asset_id, detail="full").tokens
    assert 0 < dv.tokens < full_tokens  # only the revealed strip and the old toast area
    assert dv.blocks()[0]["type"] == "text" and dv.blocks()[-1]["type"] == "image"

    same = mem.diff_since("tab:checkout", 3)
    assert same.diff.kind == "identical" and same.views == []


def test_streams_are_independent_and_content_deduplicated(tmp_path):
    mem = Memory(tmp_path / "store")
    f = view(page(seed=12), 0)
    a = mem.ingest_frame(f, source="window:A")
    b = mem.ingest_frame(f, source="window:B")
    assert a.asset_id == b.asset_id and mem.store.count() == 1
    assert mem.latest("window:B").version == 1
    rec = np.asarray(f)
    assert rec.shape == (720, 1280, 3)
