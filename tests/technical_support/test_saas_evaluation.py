"""Test label integrity and scoring semantics without downloading an embedding model."""

import json
from pathlib import Path

import pytest

from customer_support.policy import load_policy_sections
from technical_support.evaluation import evaluate_cases, validate_cases

CASES_PATH = Path("data/evaluation/saas_rag_cases.json")
DOC_PATH = Path("data/policies/saas_support_guide.md")
SOURCE = DOC_PATH.as_posix()


def test_real_case_labels_cover_all_document_sections():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    sections = load_policy_sections(DOC_PATH)
    labels = {section.title: section.content for section in sections}
    validate_cases(cases, labels)
    assert len(labels) >= 30
    assert {c["expected_title"] for c in cases if c["kind"] == "answerable"} == set(labels)
    assert len([c for c in cases if c["kind"] == "answerable"]) >= 50
    assert len([c for c in cases if c["kind"] == "missing_fact"]) >= 5


def test_invalid_label_is_rejected_before_index_build():
    cases = [{"id": "x", "kind": "answerable", "category": "a",
              "question": "hello", "expected_title": "401", "expected_phrase": "不存在"},
             {"id": "y", "kind": "unrelated", "category": "b", "question": "weather",
              "expected_title": None, "expected_phrase": None},
             {"id": "z", "kind": "missing_fact", "category": "c", "question": "price",
              "expected_title": None, "expected_phrase": None}]
    with pytest.raises(ValueError, match="证据短语"):
        validate_cases(cases, {"401": "真实证据"})


def test_missing_fact_keeps_related_document_for_manual_review():
    cases = [
        {"id": "p1", "kind": "answerable", "category": "errors", "question": "first",
         "expected_title": "401", "expected_phrase": "令牌过期"},
        {"id": "p2", "kind": "answerable", "category": "errors", "question": "second",
         "expected_title": "401", "expected_phrase": "令牌过期"},
        {"id": "n1", "kind": "unrelated", "category": "unrelated", "question": "weather",
         "expected_title": None, "expected_phrase": None},
        {"id": "m1", "kind": "missing_fact", "category": "missing_fact", "question": "price",
         "expected_title": None, "expected_phrase": None},
    ]
    good = {"title": "401", "content": "检查令牌过期", "source": SOURCE, "chunk_id": "good"}
    wrong = {"title": "403", "content": "访问权限", "source": SOURCE, "chunk_id": "wrong"}
    results = {"first": [good], "second": [wrong, good], "weather": [], "price": [good]}
    report = evaluate_cases(cases, lambda question: results[question], SOURCE)
    assert report["positive"]["hit_at_1"] == 0.5
    assert report["positive"]["hit_at_2"] == 1.0
    assert report["positive"]["mrr_at_3"] == 0.75
    assert report["unrelated"]["no_hit_rate"] == 1.0
    assert report["missing_fact"]["manual_review_required"]
    assert report["details"][-1]["retrieved"] and report["details"][-1]["review_required"]


def test_evidence_phrase_must_be_present_in_retrieved_chunk():
    cases = [{"id": "p", "kind": "answerable", "category": "auth", "question": "question",
              "expected_title": "401", "expected_phrase": "重新登录"}]
    hit = {"title": "401", "content": "没有恢复步骤", "source": SOURCE, "chunk_id": "x"}
    report = evaluate_cases(cases, lambda question: [hit], SOURCE)
    assert report["positive"]["hit_at_1"] == 0.0
    assert report["details"][0]["evidence_rank"] is None
