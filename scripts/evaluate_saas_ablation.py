"""Run same-corpus SaaS retrieval ablations. No generation API calls."""

import hashlib
import json
import math
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from technical_support.retrieval_experiments import RetrievalExperiment  # noqa: E402

cases = json.loads((root / "data/evaluation/saas_rag_cases.json").read_text(encoding="utf-8"))
engine = RetrievalExperiment(rerank=True)
report = {
    "corpus_sha256": hashlib.sha256(
        (root / "data/policies/saas_support_guide.md").read_bytes()
    ).hexdigest(),
    "cases_sha256": hashlib.sha256(
        (root / "data/evaluation/saas_rag_cases.json").read_bytes()
    ).hexdigest(),
    "variants": {},
}
for method in ["lexical", "dense", "hybrid", "bm25", "rrf", "rerank"]:
    engine.search("401令牌失效", method)
    details = []
    times = []
    hit = 0
    reciprocal = 0
    rejected = 0
    for case in cases:
        started = time.perf_counter()
        hits = engine.search(case["question"], method)
        times.append((time.perf_counter() - started) * 1000)
        rank = next(
            (
                n
                for n, h in enumerate(hits, 1)
                if h["title"] == case.get("expected_title")
                and case.get("expected_phrase", "") in h["content"]
            ),
            None,
        )
        if case["kind"] == "answerable" and rank:
            hit += 1
            reciprocal += 1 / rank
        if case["kind"] == "unrelated" and not hits:
            rejected += 1
        details.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "rank": rank,
                "titles": [h["title"] for h in hits],
            }
        )
    positive = sum(c["kind"] == "answerable" for c in cases)
    negative = sum(c["kind"] == "unrelated" for c in cases)
    metrics = {
        "recall_at_5": hit / positive,
        "mrr_at_5": reciprocal / positive,
        "unrelated_rejection_accuracy": rejected / negative,
        "p95_ms": sorted(times)[math.ceil(0.95 * len(times)) - 1],
    }
    report["variants"][method] = {"metrics": metrics, "details": details}
    (root / "data/evaluation/saas_ablation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(method, metrics, flush=True)
