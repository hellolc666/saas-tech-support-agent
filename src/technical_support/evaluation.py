"""Pure evaluation logic for the synthetic SaaS RAG corpus."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from statistics import median
from typing import Any

KINDS = {"answerable", "unrelated", "missing_fact"}


def validate_cases(cases: list[dict[str, Any]], sections: dict[str, str]) -> None:
    """Catch broken hand labels before doing expensive embedding retrieval."""
    if not cases:
        raise ValueError("评测集不能为空")
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for case in cases:
        identifier = case.get("id")
        question = case.get("question")
        kind = case.get("kind")
        category = case.get("category")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"评测 ID 缺失或重复：{identifier}")
        seen.add(identifier)
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{identifier} 缺少问题")
        if kind not in KINDS or not isinstance(category, str) or not category:
            raise ValueError(f"{identifier} 的 kind/category 无效")
        counts[kind] += 1
        title = case.get("expected_title")
        phrase = case.get("expected_phrase")
        if kind == "answerable":
            if not isinstance(title, str) or title not in sections:
                raise ValueError(f"{identifier} 的目标章节不存在：{title}")
            if not isinstance(phrase, str) or not phrase or phrase not in sections[title]:
                raise ValueError(f"{identifier} 的证据短语不在章节中：{phrase}")
        elif title is not None or phrase is not None:
            raise ValueError(f"{identifier} 为非答案用例，不能设置目标章节或证据短语")
    if any(counts[k] == 0 for k in KINDS):
        raise ValueError("必须同时包含可回答、无关和未公开事实三种用例")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(fraction * len(ordered)) - 1)], 2)


def evaluate_cases(
    cases: list[dict[str, Any]],
    search: Callable[[str], list[dict[str, Any]]],
    expected_source: str,
) -> dict[str, Any]:
    """Score chunk evidence, unrelated rejection, and review-only missing facts."""
    if not cases:
        raise ValueError("评测集不能为空")
    positive = 0
    hit1 = hit2 = hit3 = unrelated = rejected = 0
    reciprocal = 0.0
    categories: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "hit1": 0})
    details: list[dict[str, Any]] = []
    elapsed: list[float] = []
    for case in cases:
        started = time.perf_counter()
        hits = search(case["question"])
        elapsed.append((time.perf_counter() - started) * 1000)
        if not isinstance(hits, list):
            raise TypeError("search 应返回命中文档列表")
        hits = hits[:3]
        kind = case["kind"]
        detail: dict[str, Any] = {
            "id": case["id"], "kind": kind, "category": case["category"],
            "question": case["question"], "expected_title": case["expected_title"],
            "retrieved": [
                {"title": h.get("title"), "source": h.get("source"),
                 "chunk_id": h.get("chunk_id"), "excerpt": str(h.get("content", ""))[:220]}
                for h in hits
            ],
        }
        if kind == "answerable":
            positive += 1
            categories[case["category"]]["total"] += 1
            rank = next((i for i, h in enumerate(hits, 1)
                         if h.get("title") == case["expected_title"]
                         and h.get("source") == expected_source
                         and case["expected_phrase"] in str(h.get("content", ""))), None)
            detail["evidence_rank"] = rank
            if rank is not None:
                reciprocal += 1 / rank
                hit3 += 1
                hit2 += rank <= 2
                hit1 += rank == 1
                categories[case["category"]]["hit1"] += rank == 1
        elif kind == "unrelated":
            unrelated += 1
            detail["no_hit"] = not hits
            rejected += not hits
        else:
            detail["review_required"] = True
            # A related excerpt can be relevant context yet contain no answer.
            # Whether the Agent refuses to invent the missing fact needs review.
        details.append(detail)
    return {
        "case_count": len(cases),
        "positive": {
            "count": positive, "hit_at_1": round(hit1 / positive, 4) if positive else None,
            "hit_at_2": round(hit2 / positive, 4) if positive else None,
            "hit_at_3": round(hit3 / positive, 4) if positive else None,
            "mrr_at_3": round(reciprocal / positive, 4) if positive else None,
            "categories": dict(sorted(categories.items())),
        },
        "unrelated": {
            "count": unrelated,
            "no_hit_rate": round(rejected / unrelated, 4) if unrelated else None,
        },
        "missing_fact": {"count": sum(c["kind"] == "missing_fact" for c in cases),
                         "manual_review_required": True},
        "latency_ms": {"p50": round(median(elapsed), 2), "p95": _percentile(elapsed, 0.95)},
        "details": details,
    }
