"""Comparable BM25 + dense RRF and optional multilingual Cross-Encoder retrieval."""

import math
import re
import threading
import time
from collections import Counter
from functools import lru_cache

from customer_support.rag import _lexical_score, get_policy_knowledge_base
from technical_support.trace import emit


def tokens(text):
    units = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower())
    return [
        token
        for unit in units
        for token in (
            [unit]
            if unit.isascii() or len(unit) == 1
            else [unit[i : i + 2] for i in range(len(unit) - 1)]
        )
    ]


class SupportRetriever:
    def __init__(self, rerank=False):
        self._lock = threading.Lock()
        self.kb = get_policy_knowledge_base()
        self.kb.ensure_index()
        self.docs = self.kb._chunks
        self.counts = [
            Counter(tokens(d.metadata["title"] + " " + d.page_content)) for d in self.docs
        ]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / len(self.lengths)
        self.df = Counter(t for c in self.counts for t in c)
        self.reranker = None
        if rerank:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self.reranker = TextCrossEncoder(
                model_name="BAAI/bge-reranker-base", cache_dir="data/models", threads=2
            )

    def bm25(self, query):
        scores = []
        for i, counts in enumerate(self.counts):
            score = 0.0
            for term in set(tokens(query)):
                tf = counts[term]
                idf = math.log(1 + (len(self.docs) - self.df[term] + 0.5) / (self.df[term] + 0.5))
                score += (
                    idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * self.lengths[i] / self.average))
                )
            if score > 0:
                scores.append((i, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)

    def search(self, query, method="rerank", top_k=5):
        # Serialize shared ONNX/Chroma access across concurrent tool threads.
        with self._lock:
            return self._search(query, method, top_k)

    def _search(self, query, method, top_k):
        if method not in {"hybrid", "dense", "lexical", "bm25", "rrf", "rerank"}:
            raise ValueError("Unknown retrieval method")
        if not query.strip() or top_k <= 0:
            return []
        rerank_scores = {}
        if method in {"hybrid", "dense", "lexical"}:
            return self.kb.search(query, top_k=top_k, method=method)
        started = time.perf_counter()
        dense = self.kb.search(
            query, top_k=len(self.docs), method="dense", apply_relevance_threshold=False
        )
        emit(
            "dense_retrieval",
            duration_ms=(time.perf_counter() - started) * 1000,
            ranking=[{"id": h["chunk_id"], "score": h["dense_relevance"]} for h in dense],
        )
        by_id = {d.metadata["chunk_id"]: i for i, d in enumerate(self.docs)}
        relevance = {by_id[h["chunk_id"]]: h["dense_relevance"] for h in dense}
        started = time.perf_counter()
        sparse = self.bm25(query)
        emit(
            "bm25_retrieval",
            duration_ms=(time.perf_counter() - started) * 1000,
            ranking=[
                {"id": self.docs[i].metadata["chunk_id"], "score": score} for i, score in sparse
            ],
        )
        started = time.perf_counter()
        scores = {}
        if method == "bm25":
            scores = dict(sparse)
        else:
            for ranking in [[i for i, _ in sparse], [by_id[h["chunk_id"]] for h in dense]]:
                for rank, i in enumerate(ranking, 1):
                    scores[i] = scores.get(i, 0) + 1 / (60 + rank)
        selected = sorted(scores, key=scores.get, reverse=True)
        emit(
            "rrf_fusion",
            duration_ms=(time.perf_counter() - started) * 1000,
            ranking=[
                {"id": self.docs[i].metadata["chunk_id"], "score": scores[i]} for i in selected
            ],
        )
        # Identical evidence gate across experimental variants; never tune on test labels.
        selected = [
            i
            for i in selected
            if relevance.get(i, 0) >= 0.5 or _lexical_score(query, self.docs[i]) >= 0.3
        ]
        if method == "rerank" and selected:
            selected = selected[:15]
            if self.reranker is None:
                raise RuntimeError("BGE reranker is required for rerank mode")
            started = time.perf_counter()
            values = list(
                self.reranker.rerank(query, [self.docs[i].page_content for i in selected])
            )
            if len(values) != len(selected):
                raise RuntimeError("Reranker returned an incomplete candidate ranking")
            rerank_scores = dict(zip(selected, map(float, values)))
            selected = [i for _, i in sorted(zip(values, selected), reverse=True)]
            emit(
                "bge_rerank",
                duration_ms=(time.perf_counter() - started) * 1000,
                ranking=[
                    {"id": self.docs[i].metadata["chunk_id"], "score": rerank_scores[i]}
                    for i in selected
                ],
            )
        return [
            {
                **self.docs[i].metadata,
                "content": self.docs[i].page_content,
                "score": rerank_scores.get(i, scores[i]),
                "fusion_score": scores[i],
                "rerank_score": rerank_scores.get(i),
                "retrieval_pipeline": "bm25+dense+rrf+bge-reranker"
                if method == "rerank"
                else method,
            }
            for i in selected[:top_k]
        ]


# Keep the evaluation script compatible with the shared production implementation.
RetrievalExperiment = SupportRetriever


@lru_cache(maxsize=1)
def get_support_retriever():
    return SupportRetriever(rerank=True)
