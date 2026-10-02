"""MMLongBench-Doc tasks: questions grouped by document, PDFs fetched from the HF dataset."""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass
from pathlib import Path

REPO = "yubo2333/MMLongBench-Doc"


@dataclass
class Task:
    task_id: str
    doc_id: str
    question: str
    answer: str
    answer_format: str
    evidence_pages: str
    pdf_path: Path


def _download(filename: str, retries: int = 3) -> Path:
    from huggingface_hub import hf_hub_download

    for attempt in range(retries):
        try:
            return Path(
                hf_hub_download(REPO, filename, repo_type="dataset", force_download=attempt > 0)
            )
        except OSError:
            if attempt == retries - 1:
                raise
    raise AssertionError("unreachable")


def load_rows() -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(_download("data/train-00000-of-00001.parquet")).to_pylist()


def page_count(pdf: Path) -> int:
    import pymupdf as fitz

    with fitz.open(pdf) as doc:
        return doc.page_count


def text_chars_per_page(pdf: Path) -> float:
    """Average text-layer characters per page; near zero means a scanned document."""
    import pymupdf as fitz

    with fitz.open(pdf) as doc:
        return sum(len(p.get_text().strip()) for p in doc) / max(1, doc.page_count)


def _evidence(r: dict) -> set[str]:
    try:
        return {s.split(" ")[0] for s in ast.literal_eval(r["evidence_sources"] or "[]")}
    except (ValueError, SyntaxError):
        return set()


def select_tasks(
    n_docs: int,
    per_doc: int,
    min_pages: int = 10,
    max_pages: int = 60,
    seed: int = 0,
    evidence: set[str] | None = None,
    scanned_only: bool = False,
    exclude_docs: set[str] | None = None,
) -> list[Task]:
    """Pick `n_docs` documents within the page range and `per_doc` questions from each.

    Slices: `evidence` keeps only questions whose evidence includes one of these sources
    (Chart, Table, Figure, Pure-text, Generalized-text); `scanned_only` keeps documents with
    almost no text layer; `exclude_docs` skips documents used in earlier runs.
    """
    rows = load_rows()
    if evidence:
        rows = [r for r in rows if _evidence(r) & evidence]
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        if exclude_docs and r["doc_id"] in exclude_docs:
            continue
        by_doc.setdefault(r["doc_id"], []).append(r)
    rng = random.Random(seed)
    doc_ids = sorted(d for d, qs in by_doc.items() if len(qs) >= per_doc)
    rng.shuffle(doc_ids)
    tasks: list[Task] = []
    picked = 0
    for doc_id in doc_ids:
        if picked >= n_docs:
            break
        try:
            pdf = _download(f"documents/{doc_id}")
        except OSError as e:
            print(f"skipping {doc_id}: {e}")
            continue
        if not (min_pages <= page_count(pdf) <= max_pages):
            continue
        if scanned_only and text_chars_per_page(pdf) > 100:
            continue
        qs = sorted(by_doc[doc_id], key=lambda r: r["question"])
        for i, r in enumerate(rng.sample(qs, per_doc)):
            tasks.append(
                Task(
                    task_id=f"{Path(doc_id).stem[:40]}#q{i}",
                    doc_id=doc_id,
                    question=r["question"],
                    answer=r["answer"],
                    answer_format=r["answer_format"],
                    evidence_pages=r["evidence_pages"],
                    pdf_path=pdf,
                )
            )
        picked += 1
    return tasks


def synthetic_tasks(
    out_dir: Path, n_docs: int = 2, per_doc: int = 2, pages: int = 12
) -> list[Task]:
    """Offline PDFs for --dry-run and tests."""
    import pymupdf as fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for d in range(n_docs):
        pdf = out_dir / f"synthetic_{d}.pdf"
        if not pdf.exists():
            doc = fitz.open()
            for p in range(pages):
                page = doc.new_page()
                page.insert_text((72, 72), f"Synthetic document {d}, page {p + 1}", fontsize=18)
                page.insert_text((72, 120), f"The value on this page is {d * 100 + p}.")
                page.draw_rect(
                    fitz.Rect(72, 160, 72 + 30 * (p + 1) % 400, 260),
                    color=(0.2, 0.3, 0.7),
                    fill=(0.6, 0.7, 0.9),
                )
            doc.set_toc([[1, "Intro", 1], [1, "Body", 3], [1, "Appendix", pages - 1]])
            doc.save(pdf)
        for q in range(per_doc):
            tasks.append(
                Task(
                    task_id=f"synthetic_{d}#q{q}",
                    doc_id=pdf.name,
                    question=f"What is the value on page {q + 2}?",
                    answer=str(d * 100 + q + 1),
                    answer_format="Int",
                    evidence_pages=f"[{q + 2}]",
                    pdf_path=pdf,
                )
            )
    return tasks
