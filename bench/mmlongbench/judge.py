"""LLM answer judge, applied identically to every run being compared.

MMLongBench-Doc's official eval also uses a model (to extract a short answer before rule
matching). Rule matching alone misjudges correct paraphrases such as "the document mentions
no cooler" for a "Not answerable" gold. Verdicts are cached on disk by (question, gold, pred).
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

JUDGE_PROMPT = """You grade answers to questions about a document.

Question: {question}
Gold answer ({fmt}): {gold}
Predicted answer: {pred}

Rules:
- Correct if the prediction gives the same answer as the gold, ignoring wording, units and \\
formatting. Extra explanation is fine if the answer itself is right.
- Numbers must match (small rounding differences are fine). For lists, every gold item must be \\
present.
- If the gold is "Not answerable", the prediction is correct only if it says the document \\
does not contain or support an answer.
- An answer that hedges between options, or answers a different question, is incorrect.

Reply with exactly one word: CORRECT or INCORRECT."""


class Judge:
    """Haiku judges every answer; when its verdict disagrees with the rule-based scorer, a
    stronger model (Sonnet 5.5) decides. The same procedure applies to every run."""

    def __init__(
        self,
        client: Any,
        model: str = "claude-haiku-4-5",
        cache_path: str | Path = ".cache/judge.json",
        adjudicator: str | None = "claude-sonnet-5-5",
    ):
        self.client = client
        self.model = model
        self.adjudicator = adjudicator
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, float] = (
            json.loads(self.path.read_text()) if self.path.exists() else {}
        )
        self._lock = threading.Lock()

    def _verdict(self, model: str, gold: str, pred: str, fmt: str, question: str) -> float:
        key = hashlib.sha256(json.dumps([model, question, gold, pred, fmt]).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        kw: dict[str, Any] = {"max_tokens": 5}
        if model != "claude-haiku-4-5":  # thinking models need room; keep effort low
            kw = {"max_tokens": 2000, "output_config": {"effort": "low"}}
        resp = self.client.messages.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=question, gold=gold, fmt=fmt, pred=pred
                    ),
                }
            ],
            **kw,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        verdict = 1.0 if text.strip().upper().startswith("CORRECT") else 0.0
        with self._lock:
            self.cache[key] = verdict
            self.path.write_text(json.dumps(self.cache, indent=0))
        return verdict

    def __call__(self, gold: str, pred: str | None, fmt: str, question: str = "") -> float:
        if pred is None:
            return 0.0
        v = self._verdict(self.model, gold, pred, fmt, question)
        if self.adjudicator:
            from bench.mmlongbench.score import score

            if score(gold, pred, fmt) != v:  # the two scorers disagree: ask a stronger judge
                v = self._verdict(self.adjudicator, gold, pred, fmt, question)
        return v
