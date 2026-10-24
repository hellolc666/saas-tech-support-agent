from types import SimpleNamespace

from langchain_core.documents import Document

from technical_support import retrieval


def test_fusion_then_cross_encoder_changes_final_order(monkeypatch):
    docs = [
        Document(
            page_content="401登录令牌过期重新认证",
            metadata={"chunk_id": "auth", "title": "401", "source": "guide.md"},
        ),
        Document(
            page_content="401会话刷新",
            metadata={"chunk_id": "session", "title": "会话", "source": "guide.md"},
        ),
        Document(
            page_content="火星天气",
            metadata={"chunk_id": "irrelevant", "title": "天气", "source": "guide.md"},
        ),
    ]

    def search(query, **kwargs):
        assert kwargs["method"] == "dense" and not kwargs["apply_relevance_threshold"]
        return [
            {"chunk_id": key, "dense_relevance": score}
            for key, score in [("session", 0.9), ("auth", 0.8), ("irrelevant", 0.1)]
        ]

    monkeypatch.setattr(
        retrieval,
        "get_policy_knowledge_base",
        lambda: SimpleNamespace(ensure_index=lambda: None, _chunks=docs, search=search),
    )
    engine = retrieval.SupportRetriever()
    observed = []

    def rerank(query, passages):
        observed.extend(passages)
        return [10 if "过期" in text else -10 for text in passages]

    engine.reranker = SimpleNamespace(rerank=rerank)
    hits = engine.search("401登录", top_k=2)
    assert len(observed) == 2 and "火星天气" not in observed
    assert hits[0]["chunk_id"] == "auth"
    assert hits[0]["rerank_score"] == 10
    assert hits[0]["retrieval_pipeline"] == "bm25+dense+rrf+bge-reranker"
    assert hits[0]["fusion_score"] > 0
    assert engine.search(" ") == []


def test_public_policy_entry_uses_full_pipeline(monkeypatch):
    from customer_support.policy import search_policy

    calls = []

    def search(query, **kwargs):
        calls.append(kwargs)
        return [{"title": "401", "content": "登录", "source": "guide.md", "score": 1}]

    monkeypatch.setattr(retrieval, "get_support_retriever", lambda: SimpleNamespace(search=search))
    assert search_policy("401")["references"]
    assert calls == [{"top_k": 2, "method": "rerank"}]
