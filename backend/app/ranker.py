import math
from typing import Dict, List, Tuple
import sqlite3
from .utils import tokenize


class BM25Ranker:
    def __init__(self, conn: sqlite3.Connection, k1: float = 1.4, b: float = 0.75):
        self.conn = conn
        self.k1 = k1
        self.b = b

    def _stats(self) -> Tuple[int, float]:
        cur = self.conn.cursor()
        N = cur.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        if N == 0:
            return 0, 0.0
        total_len = cur.execute(
            "SELECT SUM(length) FROM docs").fetchone()[0] or 0
        avgdl = total_len / float(N) if N else 0.0
        return N, avgdl

    def _idf(self, N: int, df: int) -> float:
        # BM25 idf with +1 inside log to avoid negative values on very frequent terms
        return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        q_terms = tokenize(query)
        if not q_terms:
            return []

        cur = self.conn.cursor()
        N, avgdl = self._stats()
        if N == 0:
            return []

        scores: Dict[int, float] = {}
        for t in q_terms:
            row = cur.execute(
                "SELECT df FROM terms WHERE term = ?", (t,)).fetchone()
            if not row:
                continue
            df = row[0]
            idf = self._idf(N, df)

            for doc_id, tf in cur.execute("SELECT doc_id, tf FROM postings WHERE term = ?", (t,)):
                dl = cur.execute(
                    "SELECT length FROM docs WHERE id = ?", (doc_id,)).fetchone()[0]
                denom = tf + self.k1 * \
                    (1 - self.b + self.b * (dl / max(1.0, avgdl)))
                score_add = idf * ((tf * (self.k1 + 1)) / max(1e-9, denom))
                scores[doc_id] = scores.get(doc_id, 0.0) + score_add

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return ranked

    def search_page(self, query: str, page: int = 1, per_page: int = 10):
        ranked = self.search(query, k=10_000)
        start = max(0, (page - 1) * per_page)
        end = start + per_page
        return ranked[start:end], len(ranked)


def get_postings_with_pos(conn: sqlite3.Connection, term: str):
    cur = conn.cursor()
    return [
        (doc_id, tf, list(map(int, positions.split(","))))
        for (doc_id, tf, positions) in cur.execute(
            "SELECT doc_id, tf, positions FROM postings WHERE term= ?", (term,)
        )
    ]
