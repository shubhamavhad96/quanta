from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorReRanker:
    """
    Re-rank BM25 candidates with cosine similarity on MiniLM embeddings.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", weight: float = 0.35):
        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self.weight = float(weight)

    def _load(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        self._load()
        embs = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True)
        return embs.astype("float32")

    def rerank(self, query: str, candidates: List[Tuple[int, float]], fetch_text_fn) -> List[Tuple[int, float]]:
        if not candidates or self.weight <= 0:
            return candidates

        q_emb = self.embed([query])[0]

        doc_ids = [d for d, _ in candidates]
        texts = []
        for doc_id in doc_ids:
            title, body = fetch_text_fn(doc_id)
            texts.append((title + " " + body[:500]).strip())
        d_embs = self.embed(texts)

        sims = (d_embs @ q_emb)

        bm25 = np.array([s for _, s in candidates], dtype="float32")
        if bm25.size > 0:
            bmin, bmax = float(bm25.min()), float(bm25.max())
            bm25n = (bm25 - bmin) / (bmax - bmin + 1e-6)
        else:
            bm25n = bm25
        simn = (sims - sims.min()) / (sims.max() - sims.min() + 1e-6)

        final = (1.0 - self.weight) * bm25n + self.weight * simn
        pairs = list(zip(doc_ids, final.tolist()))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs
