import pytest

from bench.mmlongbench.data import synthetic_tasks
from foveal import Memory
from foveal.memory.render import fit_to_tokens


class FakeCaptioner:
    def __init__(self):
        self.calls = 0

    def __call__(self, jpeg):
        self.calls += 1
        return f"caption {self.calls}"


@pytest.fixture
def pdf(tmp_path):
    return synthetic_tasks(tmp_path / "pdf", n_docs=1, per_doc=1, pages=4)[0].pdf_path


def test_ingest_pdf_once(tmp_path, pdf):
    cap = FakeCaptioner()
    mem = Memory(tmp_path / "store", captioner=cap)
    pages = mem.ingest_pdf(pdf)
    assert len(pages) == 4 and cap.calls == 4
    assert "page 2" in pages[1].text and pages[1].caption.startswith("caption")
    again = mem.ingest_pdf(pdf)  # same bytes -> no new perception
    assert cap.calls == 4 and [a.asset_id for a in again] == [a.asset_id for a in pages]
    assert mem.store.count() == 4


def test_look_levels(tmp_path, pdf):
    mem = Memory(tmp_path / "store")
    a = mem.ingest_pdf(pdf)[0]
    glimpse = mem.look(a.asset_id)
    assert glimpse.tokens <= 256
    full = mem.look(a.asset_id, detail="full")
    assert (full.width, full.height) == (a.width, a.height)
    crop = mem.look(a.asset_id, region=[0.0, 0.0, 0.5, 0.25], detail="full")
    # a zoomed crop is re-rendered from the PDF: it fills the long edge, unlike a pixel crop
    assert max(crop.width, crop.height) > 0.5 * a.width
    assert mem.look(a.asset_id, region=[0, 0, 0.5, 0.25], detail="full").jpeg == crop.jpeg
    with pytest.raises(ValueError):
        mem.look(a.asset_id, region=[0.5, 0.5, 0.5, 0.6], detail="full")


def test_describe_and_image_ingest(tmp_path, png_factory):
    mem = Memory(tmp_path / "store")
    a = mem.ingest(png_factory(seed=2), text="hello")
    assert mem.ingest(png_factory(seed=2), text="hello").asset_id == a.asset_id
    d = mem.describe(a.asset_id)
    assert "hello" in d and a.asset_id[:12] in d


def test_fit_to_tokens():
    w, h = fit_to_tokens(1109, 1568, 256)
    assert -(-w // 28) * -(-h // 28) <= 256 and w / h == pytest.approx(1109 / 1568, rel=0.03)
