"""Run the real project's Hybrid RAG on labelled SaaS cases; no LLM calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from customer_support.policy import POLICY_PATH, load_policy_sections  # noqa: E402
from technical_support.evaluation import evaluate_cases, validate_cases  # noqa: E402

CASES_PATH = Path("data/evaluation/saas_rag_cases.json")
REPORT_PATH = Path("data/evaluation/saas_rag_report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        parser.error("cases 顶层必须是数组")
    sections = load_policy_sections()
    titles = [section.title for section in sections]
    if len(titles) != len(set(titles)):
        parser.error("文档章节标题重复，无法按标题验收")
    validate_cases(cases, {section.title: section.content for section in sections})

    # Import after label validation: loading the real index can be expensive.
    from customer_support.rag import get_policy_knowledge_base

    knowledge_base = get_policy_knowledge_base()
    chunk_count = knowledge_base.ensure_index()
    print(f"知识库：{len(sections)} 章节，{chunk_count} 个切片")
    # Warm up retrieval separately so p50/p95 exclude first model load.
    first_positive = next(case for case in cases if case["kind"] == "answerable")
    knowledge_base.search(first_positive["question"], top_k=3, method="hybrid")
    result = evaluate_cases(
        cases,
        search=lambda question: knowledge_base.search(question, top_k=3, method="hybrid"),
        expected_source=POLICY_PATH.as_posix(),
    )
    result["corpus_sha256"] = hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()
    result["cases_sha256"] = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    result["method"] = "hybrid"
    result["chunk_count"] = chunk_count
    report = args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    positive = result["positive"]
    print(f"可回答 {positive['count']}：Hit@1 {positive['hit_at_1']:.1%}，"
          f"Hit@2 {positive['hit_at_2']:.1%}，Hit@3 {positive['hit_at_3']:.1%}，"
          f"MRR@3 {positive['mrr_at_3']:.1%}")
    print(f"无关 {result['unrelated']['count']}：无命中率 {result['unrelated']['no_hit_rate']:.1%}")
    print(f"缺失事实 {result['missing_fact']['count']}：需要检查 Agent 是否正确拒答")
    print(f"检索延时 P50 {result['latency_ms']['p50']:.2f} ms，"
          f"P95 {result['latency_ms']['p95']:.2f} ms")
    failed = [d for d in result["details"] if d["kind"] == "answerable" and d["evidence_rank"] != 1]
    for case in failed[:10]:
        titles = [h["title"] for h in case["retrieved"]]
        print(f"待检查 {case['id']}：{case['question']} -> {titles}")
    print(f"详细报告：{report}")


if __name__ == "__main__":
    main()
