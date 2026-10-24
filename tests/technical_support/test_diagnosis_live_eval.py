"""Pure parsing/aggregation tests; no network requests or paid Qwen calls."""

from scripts.evaluate_saas_diagnosis_live import (
    analyze_events,
    evidence_path_ok,
    extract_final_code,
    parse_sse_line,
    redact,
    summarize,
)


def tool_event(view, incident_id="F01", app_id="APP_DEMO_001"):
    return {
        "type": "message", "content": {
            "type": "tool", "content": __import__("json").dumps({
                "ok": True, "incident_id": incident_id, "app_id": app_id, "view": view,
            }),
        },
    }


def test_extract_explicit_final_conclusion_not_candidate_list():
    assert extract_final_code("候选 QUOTA_EXHAUSTED、BURST_RATE_LIMIT\n根因代码：BURST_RATE_LIMIT") == "BURST_RATE_LIMIT"
    assert extract_final_code("QUOTA_EXHAUSTED 和 BURST_RATE_LIMIT 需要继续调查") is None
    assert extract_final_code("根因代码：BOGUS_CODE") is None


def test_requires_ordered_successful_evidence():
    assert evidence_path_ok("F01", [
        {"ok": True, "incident_id": "F01", "app_id": "APP_DEMO_001", "view": "overview"},
        {"ok": True, "incident_id": "F01", "app_id": "APP_DEMO_001", "view": "traffic"},
    ])
    assert not evidence_path_ok("F01", [
        {"ok": True, "incident_id": "F01", "app_id": "APP_DEMO_001", "view": "traffic"},
        {"ok": True, "incident_id": "F01", "app_id": "APP_DEMO_001", "view": "overview"},
    ])
    assert not evidence_path_ok("F05", [
        {"ok": True, "incident_id": "F05", "app_id": "APP_DEMO_001", "view": "overview"},
        {"ok": True, "incident_id": "F05", "app_id": "APP_DEMO_001", "view": "auth"},
    ])


def test_score_real_stream_and_no_missing_evidence_shortcut():
    events = [tool_event("overview"), tool_event("traffic"),
              {"type": "message", "content": {"type": "ai", "content": "根因代码：QUOTA_EXHAUSTED", "tool_calls": []}},
              {"type": "done"}]
    result = analyze_events("F01", events, 9.2)
    assert result["judgment"] == "correct" and result["evidence_path_ok"]
    without_evidence = analyze_events("F01", events[2:], 2.1)
    assert without_evidence["judgment"] == "incorrect"
    ambiguous = analyze_events("F01", [tool_event("overview"), tool_event("traffic"),
        {"type": "message", "content": {"type": "ai", "content": "疑似配额用尽", "tool_calls": []}},
        {"type": "done"}], 3.4)
    assert ambiguous["judgment"] == "manual_review"
    assert summarize([result, ambiguous])["needs_manual_review"] == 1


def test_sse_parser_and_redaction():
    assert parse_sse_line('data: {"type":"token","content":"hello"}')["content"] == "hello"
    assert parse_sse_line("data: [DONE]") == {"type": "done"}
    assert "DEMO-ALPHA-4821" not in redact("DEMO-ALPHA-4821")
    assert "sk-example" not in redact("sk-example")
