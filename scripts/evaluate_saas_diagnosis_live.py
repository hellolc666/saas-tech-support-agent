"""Run a real Qwen / FastAPI SSE diagnosis benchmark on synthetic SaaS incidents.

This is a model evaluation, NOT a unit test, not a deterministic-rule baseline,
and not evidence of performance on production incidents. It incurs Qwen API costs.

Usage:
  python scripts/evaluate_saas_diagnosis_live.py --cases F01,F02
  python scripts/evaluate_saas_diagnosis_live.py --base-url http://127.0.0.1:18085
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Ground truth is imported only in the evaluation process; it is never
# injected into the user question, model context, or read-only diagnostic tools.
from technical_support.incident_scenarios import GROUND_TRUTH, SCENARIOS  # noqa: E402

RESULT_PATH = ROOT / "data" / "evaluation" / "saas-diagnosis-agent-live.json"
VALID_CAUSES = set(GROUND_TRUTH.values()) | {"UNDETERMINED", "CONFLICTING_EVIDENCE"}
CODE_PATTERN = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?(?:最终)?根因代码(?:\*\*)?\s*[:：]\s*`?([A-Z][A-Z0-9_]+)`?"
)
DEMO_CODES = {"APP_DEMO_001": "DEMO-ALPHA-4821", "APP_DEMO_002": "DEMO-BETA-7394"}


def redact(text: str) -> str:
    """Never write credentials to benchmark artifacts."""
    text = re.sub(r"DEMO-(?:ALPHA|BETA)-\d{4}", "[DEMO_CODE]", text, flags=re.I)
    text = re.sub(r"\bsk-[\w.-]+", "[API_KEY]", text, flags=re.I)
    return re.sub(r"Bearer\s+[\w.-]+", "Bearer [REDACTED]", text, flags=re.I)


def extract_final_code(answer: str) -> str | None:
    """Only score an explicit final code, never guesses found in hypotheses lists."""
    candidates = [s for s in CODE_PATTERN.findall(answer) if s in VALID_CAUSES]
    return candidates[-1] if candidates and len(set(candidates)) == 1 else None


def required_detail(case_id: str) -> str:
    case = SCENARIOS[case_id]
    return (
        "export" if case["category"] == "export" else
        {429: "traffic", 401: "auth", 503: "dependency"}.get(case["http_status"], "")
    )


def evidence_path_ok(case_id: str, observations: list[dict]) -> bool:
    """Successful tool results must include overview followed by the needed detail."""
    successes = [
        row["view"] for row in observations
        if row.get("ok") is True
        and row.get("incident_id") == case_id
        and row.get("app_id") == SCENARIOS[case_id]["app_id"]
    ]
    expected = required_detail(case_id)
    return bool(expected and "overview" in successes and expected in successes
                and successes.index("overview") < successes.index(expected))


def parse_sse_line(line: str) -> dict | None:
    if not line.startswith("data: "):
        return None
    content = line[6:]
    if content == "[DONE]":
        return {"type": "done"}
    data = json.loads(content)
    return data if isinstance(data, dict) else None


def analyze_events(case_id: str, events: list[dict], elapsed_s: float) -> dict:
    tool_calls, observations, answers, errors = [], [], [], []
    token_usages = []
    finished = False
    for event in events:
        kind = event.get("type")
        if kind == "done":
            finished = True
        elif kind == "error":
            errors.append(str(event.get("content", "Stream error"))[:160])
        elif kind == "message" and isinstance(event.get("content"), dict):
            message = event["content"]
            if message.get("type") == "ai":
                for call in message.get("tool_calls") or []:
                    if isinstance(call, dict):
                        args = call.get("args") or {}
                        tool_calls.append({
                            "name": call.get("name"),
                            "view": args.get("view") if isinstance(args, dict) else None,
                        })
                if not message.get("tool_calls") and isinstance(message.get("content"), str):
                    answers.append(message["content"])
                metadata = message.get("response_metadata") or {}
                usage = metadata.get("token_usage") or metadata.get("usage")
                if isinstance(usage, dict):
                    token_usages.append(usage)
            elif message.get("type") == "tool":
                try:
                    payload = json.loads(message.get("content", ""))
                except (ValueError, TypeError):
                    continue
                if isinstance(payload, dict) and "incident_id" in payload and "view" in payload:
                    observations.append({
                        "incident_id": payload.get("incident_id"),
                        "app_id": payload.get("app_id"),
                        "view": payload.get("view"),
                        "ok": payload.get("ok"),
                    })
    answer = redact(answers[-1]) if answers else ""
    predicted = extract_final_code(answer)
    evidence_ok = evidence_path_ok(case_id, observations)
    expected = GROUND_TRUTH[case_id]
    if errors or not finished:
        judgment = "error"
    elif predicted is None:
        judgment = "manual_review"
    elif predicted == expected and evidence_ok:
        judgment = "correct"
    else:
        judgment = "incorrect"
    return {
        "case_id": case_id,
        "expected_root_cause": expected,
        "predicted_root_cause": predicted,
        "judgment": judgment,
        "evidence_path_ok": evidence_ok,
        "tool_calls": tool_calls,
        "evidence_observations": observations,
        "duration_s": round(elapsed_s, 2),
        "answer": answer[:6000],
        "sse_completed": finished,
        "errors": errors,
        "model_usage_metadata": token_usages or None,
    }


def run_case(client: httpx.Client, base_url: str, agent: str, case_id: str, timeout: int) -> dict:
    app_id = SCENARIOS[case_id]["app_id"]
    prompt = (
        f"请诊断应用 {app_id} 的合成历史故障 {case_id}，"
        f"演示验证码 {DEMO_CODES[app_id]}。请通过工具取证，"
        "对候选根因逐一核实；结尾单独一行写‘根因代码：CODE’，"
        "其中 CODE 为验证后的英文根因代码；证据不足则写 UNDETERMINED。"
    )
    started = time.perf_counter()
    events: list[dict] = []
    try:
        with client.stream(
            "POST", f"{base_url.rstrip('/')}/{agent}/stream",
            json={"message": prompt, "thread_id": str(uuid4()), "stream_tokens": False},
            timeout=httpx.Timeout(connect=15, read=timeout, write=20, pool=15),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                event = parse_sse_line(line)
                if event is not None:
                    events.append(event)
    except (httpx.HTTPError, json.JSONDecodeError) as error:
        events.append({"type": "error", "content": type(error).__name__})
    return analyze_events(case_id, events, time.perf_counter() - started)


def summarize(details: list[dict]) -> dict:
    times = sorted(row["duration_s"] for row in details)
    def percentile(p: float) -> float | None:
        if not times:
            return None
        index = p * (len(times) - 1)
        lo = int(index)
        hi = min(lo + 1, len(times) - 1)
        return round(times[lo] + (times[hi] - times[lo]) * (index - lo), 2)

    total = len(details)
    return {
        "cases": total,
        "correct_strict": sum(x["judgment"] == "correct" for x in details),
        "strict_accuracy_lower_bound": round(sum(x["judgment"] == "correct" for x in details) / total, 4) if total else None,
        "needs_manual_review": sum(x["judgment"] == "manual_review" for x in details),
        "incorrect": sum(x["judgment"] == "incorrect" for x in details),
        "errors": sum(x["judgment"] == "error" for x in details),
        "evidence_path_rate": round(sum(x["evidence_path_ok"] for x in details) / total, 4) if total else None,
        "mean_tool_calls": round(statistics.mean(len(x["tool_calls"]) for x in details), 2) if total else None,
        "latency_p50_s": percentile(0.50),
        "latency_p95_s": percentile(0.95),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--agent", default="saas-support")
    parser.add_argument("--cases", default="F01,F02,F03,F04,F05,F06,F07,F08")
    parser.add_argument("--timeout", type=int, default=360, help="SSE idle/read timeout seconds")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    case_ids = [item.strip().upper() for item in args.cases.split(",") if item.strip()]
    if not case_ids or any(case not in GROUND_TRUTH for case in case_ids):
        parser.error("--cases must be comma-separated F01..F08")
    load_dotenv(ROOT / ".env")
    auth_secret = os.getenv("AUTH_SECRET", "")
    headers = {"Authorization": f"Bearer {auth_secret}"} if auth_secret else {}
    # Explicitly bypass VPN/system proxies for loopback requests.
    with httpx.Client(headers=headers, trust_env=False) as client:
        try:
            client.get(f"{args.base_url.rstrip('/')}/health", timeout=8).raise_for_status()
        except httpx.HTTPError as error:
            print(f"Backend not reachable: {type(error).__name__}. Start FastAPI first.", file=sys.stderr)
            return 2
        details = []
        for case in case_ids:
            print(f"Running {case} (real Qwen API calls) ...", flush=True)
            result = run_case(client, args.base_url, args.agent, case, args.timeout)
            details.append(result)
            print(f"{case}: {result['judgment']} | path={result['evidence_path_ok']} | "
                  f"tools={len(result['tool_calls'])} | {result['duration_s']}s", flush=True)
    report = {
        "scope": "Real SaaS Agent via FastAPI/SSE, 8 synthetic fixed cases; not production performance",
        "created_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "agent": args.agent,
        "metrics": summarize(details),
        "details": details,
        "grading_note": "Only explicit root-cause code in the final answer plus ordered, successful evidence queries counts as correct. Missing code needs manual review. Token usage is reported only if present in SSE message metadata; null means unavailable, not zero.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")
    return 0 if not any(x["judgment"] == "error" for x in details) else 1


if __name__ == "__main__":
    raise SystemExit(main())
