from sentence_transformers import CrossEncoder
from typing import List, Tuple


class CrossReranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.m = CrossEncoder(model)

    def rerank(self, query: str, candidates: List[Tuple[int, str]], top_k: int = 8):
        if not candidates:
            return []
        pairs = [[query, txt] for _, txt in candidates]
        scores = self.m.predict(pairs)
        rank = sorted(zip(candidates, scores),
                      key=lambda x: x[1], reverse=True)[:top_k]
        return [(doc_id, float(s)) for ((doc_id, _), s) in rank]
