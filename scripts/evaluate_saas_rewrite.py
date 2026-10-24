"""Paid contextual rewrite comparison: fixed cases, same production retriever."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.language_models import FakeMessagesListChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from core import settings  # noqa: E402
from technical_support.knowledge_agent import build_technical_support_graph  # noqa: E402
from technical_support.retrieval import get_support_retriever  # noqa: E402

CASES = [
    ("APP_DEMO_001导出失败", "那5001条呢？", "导出失败与导出数量限制"),
    ("接口返回401，提示令牌过期", "这个怎么恢复？", "401 登录失效与令牌过期"),
    ("Webhook签名校验失败", "该怎么核对？", "Webhook 签名校验"),
    ("CSV任务导入提示格式错误", "应该检查哪些列？", "CSV 任务导入失败"),
    ("应用请求返回429", "要等多久才能重试？", "429 请求过多与配额"),
    ("成员邀请链接已经失效", "能重新发送吗？", "成员邀请与邀请失效"),
]


class AnswerStub(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


async def main():
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    planner = ChatOpenAI(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_BASE_URL,
        model=settings.DASHSCOPE_MODEL,
        temperature=0,
        timeout=35,
        extra_body={"enable_thinking": False},
    ).bind(response_format={"type": "json_object"})
    graph = build_technical_support_graph(
        AnswerStub(responses=[AIMessage(content="仅评估查询重写")]), planner=planner
    )
    retriever = get_support_retriever()
    details = []
    for index, (previous, current, title) in enumerate(CASES):
        started = time.perf_counter()
        state = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content=previous),
                    AIMessage(content="请补充具体情况"),
                    HumanMessage(content=current),
                ]
            },
            {"configurable": {"thread_id": f"rewrite-eval-{index}"}},
        )
        understanding = state["understanding"]
        assert not understanding.get("planner_error"), understanding
        rewritten = understanding["independent_query"]
        original_hits = retriever.search(current, top_k=5)
        rewritten_hits = retriever.search(rewritten, top_k=5)
        ranks = []
        for hits in [original_hits, rewritten_hits]:
            ranks.append(next((i for i, h in enumerate(hits, 1) if h["title"] == title), None))
        details.append(
            {
                "context": previous,
                "query": current,
                "rewritten": rewritten,
                "expected_title": title,
                "original_rank": ranks[0],
                "rewritten_rank": ranks[1],
                "elapsed_s": round(time.perf_counter() - started, 2),
            }
        )
        print(index, ranks, flush=True)
    report = {
        "case_count": len(details),
        "scope": "Six fixed contextual retrieval examples; not end-to-end answer accuracy.",
        "original_hit_at_5": sum(x["original_rank"] is not None for x in details) / len(details),
        "rewritten_hit_at_5": sum(x["rewritten_rank"] is not None for x in details) / len(details),
        "original_mrr_at_5": sum(
            1 / x["original_rank"] if x["original_rank"] else 0 for x in details
        )
        / len(details),
        "rewritten_mrr_at_5": sum(
            1 / x["rewritten_rank"] if x["rewritten_rank"] else 0 for x in details
        )
        / len(details),
        "details": details,
    }
    (ROOT / "data/evaluation/saas-rewrite-comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    asyncio.run(main())
