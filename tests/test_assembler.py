from bench.mmlongbench.data import synthetic_tasks
from foveal import Memory
from foveal.assembler import Assembler, CacheModel, Item


class Cap:
    def __call__(self, jpeg):
        return "a page with a blue bar chart"


def _pages(tmp_path):
    pdf = synthetic_tasks(tmp_path / "pdf", n_docs=1, per_doc=1, pages=6)[0].pdf_path
    mem = Memory(tmp_path / "s", captioner=Cap())
    return mem, [a.asset_id for a in mem.ingest_pdf(pdf)]


def test_budget_and_relevance(tmp_path):
    mem, ids = _pages(tmp_path)
    tight = Assembler(mem, budget_tokens=600).build(ids, "what is the value on page 4?")
    assert tight.tokens <= 600 and "L3" not in tight.levels.values()
    roomy = Assembler(mem, budget_tokens=20000, recent=0).build(ids, "value on page 4")
    # the page whose text mentions "page 4" ranks first and gets the most detail
    best = max(ids, key=lambda i: "page 4" in (mem.get(i).text or ""))
    assert roomy.levels[best] in ("L2", "L3")
    assert any(b["type"] == "image" for b in roomy.blocks)


def test_cache_model_compacts_only_when_it_pays():
    cm = CacheModel("claude-sonnet-5-5")
    assert not cm.should_compact(100_000, 90_000, calls_left=2)  # small saving, few calls
    assert cm.should_compact(100_000, 10_000, calls_left=50)  # big saving, long horizon
    assert not cm.should_compact(100_000, 10_000, calls_left=0)


def test_plan_compaction(tmp_path):
    mem, ids = _pages(tmp_path)
    asm = Assembler(mem)
    items = [Item(a, k, "L3", asm.level_tokens(a, "L3")) for k, a in enumerate(ids)]
    assert asm.plan_compaction(items, calls_left=1) is None
    plan = asm.plan_compaction(items, calls_left=40, keep_recent=2)
    assert plan is not None and set(plan) == set(ids[:-2]) and set(plan.values()) == {"L0"}
