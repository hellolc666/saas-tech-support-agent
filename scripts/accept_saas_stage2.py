"""Paid Qwen smoke acceptance against a running SaaS FastAPI service.

Run explicitly after starting the service in qwen mode. Makes multiple model calls.
"""

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

root = Path(__file__).resolve().parents[1]
load_dotenv(root / ".env")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-url", default="http://127.0.0.1:8080")
parser.add_argument("--report", type=Path, default=root / "data/evaluation/saas_stage2_report.json")
args = parser.parse_args()
out = args.report
out.parent.mkdir(parents=True, exist_ok=True)
headers = (
    {"Authorization": "Bearer " + os.environ["AUTH_SECRET"]} if os.getenv("AUTH_SECRET") else {}
)
results = []
with httpx.Client(base_url=args.base_url, headers=headers, trust_env=False, timeout=180) as c:
    c.get("/info").raise_for_status()
    workspace = c.post("/saas/workspace")
    workspace.raise_for_status()
    profile_token = workspace.json()["token"]
    c.headers["X-SaaS-Token"] = profile_token

    def run(name, turns, expected_tool=None, terms=()):
        tid = str(uuid.uuid4())
        answers = []
        start = time.monotonic()
        try:
            for text in turns:
                r = c.post(
                    "/saas-support/invoke",
                    json={
                        "message": text,
                        "thread_id": tid,
                        "agent_config": {"profile_token": profile_token},
                    },
                )
                r.raise_for_status()
                answers.append(r.json()["content"])
            h = c.post("/history", json={"thread_id": tid})
            h.raise_for_status()
            messages = h.json()["messages"]
            calls = [t for m in messages for t in m.get("tool_calls", [])]
            tool_results = [m["content"] for m in messages if m["type"] == "tool"]
            names = [t["name"] for t in calls]
            passed = (not expected_tool or expected_tool in names) and all(
                t in answers[-1] for t in terms
            )
            record = {
                "name": name,
                "passed": passed,
                "answers": answers,
                "tool_names": names,
                "tool_results": tool_results,
                "elapsed_s": round(time.monotonic() - start, 2),
            }
        except Exception as e:
            record = {
                "name": name,
                "passed": False,
                "error_type": type(e).__name__,
                "elapsed_s": round(time.monotonic() - start, 2),
            }
        results.append(record)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            name, record["passed"], record.get("tool_names", record.get("error_type")), flush=True
        )

    run(
        "401 knowledge",
        ["API接口返回401怎么办？"],
        "search_technical_docs",
        ("401", "saas_support_guide.md"),
    )
    run(
        "5001 export",
        ["为什么 APP_DEMO_001 导出5001条失败？演示验证码 DEMO-ALPHA-4821"],
        "check_export_eligibility",
        ("5000",),
    )
    run(
        "multi-turn export",
        ["为什么导出5001条失败？", "应用 APP_DEMO_001", "演示验证码 DEMO-ALPHA-4821"],
        "check_export_eligibility",
        ("5000",),
    )
    run(
        "identity denial",
        ["查询 APP_DEMO_001 配额，演示验证码 DEMO-BETA-7394"],
        "get_api_usage",
        ("不匹配",),
    )
    run(
        "version and permission",
        ["APP_DEMO_002 为什么不能导出100条？演示验证码 DEMO-BETA-7394"],
        "check_export_eligibility",
        ("1.0",),
    )
    # Exercise actual SSE transport with the real model and real retrieval.
    try:
        r = c.post(
            "/saas-support/stream",
            json={
                "message": "401登录失效怎么办？",
                "thread_id": str(uuid.uuid4()),
                "stream_tokens": True,
                "agent_config": {"profile_token": profile_token},
            },
        )
        r.raise_for_status()
        events = [
            json.loads(line[6:])
            for line in r.text.splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"
        ]
        passed = (
            "[DONE]" in r.text
            and any("saas_support_guide.md" in json.dumps(e, ensure_ascii=False) for e in events)
            and not any(e.get("type") == "error" for e in events)
        )
        results.append(
            {
                "name": "SSE Qwen retrieval",
                "passed": passed,
                "event_types": sorted({e["type"] for e in events}),
            }
        )
    except Exception as e:
        results.append(
            {"name": "SSE Qwen retrieval", "passed": False, "error_type": type(e).__name__}
        )
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SSE", results[-1]["passed"], flush=True)


raise SystemExit(0 if all(r["passed"] for r in results) else 1)
