from bench.mmlongbench.run import main
from bench.mmlongbench.score import score
from foveal.instrument.metrics import compute
from foveal.instrument.records import read_jsonl


def test_score_formats():
    assert score("42", "The answer is 42.", "Int") == 1
    assert score("42", "41", "Int") == 0
    assert score("12.5%", "12.5", "Float") == 1
    assert score("Not answerable", "Not answerable", "None") == 1
    assert score("Not answerable", "7", "None") == 0
    assert score("['red', 'blue']", '["Blue", "Red"]', "List") == 1
    assert score("['red', 'blue']", '["red"]', "List") == 0
    assert score('["Stage 5"]', "Only Stage 5, with heaters", "Str") == 1
    assert score("Some college or more", "some college or more", "Str") == 1


def test_dry_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["--dry-run", "--run-id", "t", "--docs", "1", "--per-doc", "1"]) == 0
    recs = read_jsonl(tmp_path / "runs" / "t.jsonl")
    calls = [r for r in recs if r["type"] == "call"]
    agents = {c["agent_id"] for c in calls}
    assert {"orch", "reader-1", "reader-2"} <= agents
    assert all(c["task_id"] for c in calls)
    s = compute(recs)
    assert s["n_tasks"] == 1 and s["cross_agent_share"] > 0 and s["same_agent_share"] > 0
    # count_tokens ground truth (fake) agrees with the estimate for every image
    imgs = [i for c in calls for i in c["images"]]
    assert imgs and all(i["tokens"] == i["est_tokens"] for i in imgs)


def test_memory_mode_dry_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = ["--dry-run", "--docs", "1", "--per-doc", "2", "--store", str(tmp_path / "st")]
    assert main([*args, "--run-id", "img"]) == 0
    assert main([*args, "--run-id", "mem", "--mode", "memory"]) == 0
    mem = read_jsonl(tmp_path / "runs" / "mem.jsonl")
    s = compute(mem)
    assert s["ingest_calls"] == 12  # one caption per page, once for both tasks
    assert s["ingest_image_tokens"] > 0
    looks = [
        i
        for c in mem
        if c["type"] == "call" and not c["task_id"].startswith("ingest:")
        for i in c["images"]
    ]
    assert looks  # pixels arrive only through look()
    base = compute(read_jsonl(tmp_path / "runs" / "img.jsonl"))
    assert s["image_tokens_total"] < base["image_tokens_total"]
