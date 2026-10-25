"""Load the SaaS support graph through the existing service configuration."""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from agents.lazy_agent import LazyLoadingAgent
from core import settings
from technical_support.knowledge_agent import (
    build_technical_support_graph,
    build_technical_support_mock,
)

logger = logging.getLogger(__name__)


class SaaSSupportAgent(LazyLoadingAgent):
    async def load(self) -> None:
        mode = settings.CUSTOMER_SUPPORT_MODE.lower()
        configured = bool(
            settings.DASHSCOPE_API_KEY and settings.DASHSCOPE_BASE_URL and settings.DASHSCOPE_MODEL
        )
        if mode not in {"auto", "mock", "qwen"}:
            raise ValueError("CUSTOMER_SUPPORT_MODE must be auto, mock, or qwen")
        use_qwen = mode == "qwen" or (mode == "auto" and configured)
        if use_qwen and not configured:
            raise ValueError("Qwen mode requires all DASHSCOPE settings")
        if use_qwen:
            model = ChatOpenAI(
                api_key=settings.DASHSCOPE_API_KEY,
                base_url=settings.DASHSCOPE_BASE_URL,
                model=settings.DASHSCOPE_MODEL,
                temperature=0,
                streaming=True,
                tags=["skip_stream"],
                timeout=60,
                max_retries=2,
                extra_body={"enable_thinking": False},
            )
            structured_model = model.bind(response_format={"type": "json_object"})
            self._graph = build_technical_support_graph(
                model,
                planner=structured_model,
                memory_extractor=structured_model,
            )
            logger.info("SaaS support mode: Qwen (documents, diagnostics and approved tickets)")
        else:
            self._graph = build_technical_support_mock()
            logger.info(
                "SaaS support mode: offline mock (documents and deterministic business workflow)"
            )
        self._loaded = True


saas_support_agent = SaaSSupportAgent()
