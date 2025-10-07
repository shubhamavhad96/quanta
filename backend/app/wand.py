import heapq
import math
from typing import Dict, List, Tuple
import sqlite3


class BMWScorer:
    """
    Lite Block-Max WAND over BM25 using precomputed term_blocks(max_tf).
    """

    def __init__(self, conn: sqlite3.Connection, k1: float = 1.4, b: float = 0.75, block_size: int = 128):
        self.conn = conn
        self.k1 = k1
        self.b = b
        self.block_size = block_size
        c = self.conn.cursor()
        self.N = c.execute("SELECT COUNT(*) FROM docs").fetchone()[0] or 1
        total_len = c.execute(
            "SELECT SUM(length) FROM docs").fetchone()[0] or 0
        self.avgdl = (total_len / self.N) if self.N else 0.0

    def _idf(self, df: int) -> float:
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1e-9)

    def _bm25_ub(self, idf: float, max_tf: int) -> float:
        k1 = self.k1
        b = self.b
        tf = float(max_tf)
        denom = tf + k1 * (1 - b + b * 1.0)
        return idf * ((tf * (k1 + 1)) / max(1e-9, denom))

    def _load_blocks(self, term: str) -> List[Tuple[int, int, int]]:
        c = self.conn.cursor()
        return list(c.execute("SELECT start_doc_id, end_doc_id, max_tf FROM term_blocks WHERE term=? ORDER BY start_doc_id", (term,)))

    def _load_df(self, term: str) -> int:
        c = self.conn.cursor()
        r = c.execute("SELECT df FROM terms WHERE term=?", (term,)).fetchone()
        return r[0] if r else 0

    def _postings_in_range(self, term: str, lo: int, hi: int) -> List[Tuple[int, int]]:
        c = self.conn.cursor()
        return list(c.execute(
            "SELECT doc_id, tf FROM postings WHERE term=? AND doc_id BETWEEN ? AND ? ORDER BY doc_id",
            (term, lo, hi)
        ))

    def search(self, q_terms: List[str], k: int = 50) -> List[Tuple[int, float]]:
        terms = [t for t in q_terms if t]
        if not terms:
            return []
        blocks = {t: self._load_blocks(t) for t in terms}
        idfs = {t: self._idf(self._load_df(t)) for t in terms}
        ptr = {t: 0 for t in terms}
        top: List[Tuple[float, int]] = []
        theta = 0.0
        c = self.conn.cursor()
        while True:
            active = [t for t in terms if ptr[t] < len(blocks[t])]
            if not active:
                break
            sum_ub = 0.0
            pivot_lo = 10**12
            pivot_hi = -1
            for t in active:
                (start, end, max_tf) = blocks[t][ptr[t]]
                pivot_lo = min(pivot_lo, start)
                pivot_hi = max(pivot_hi, end)
                sum_ub += self._bm25_ub(idfs[t], max_tf)
            if sum_ub <= theta and top:
                advance_t = min(active, key=lambda x: blocks[x][ptr[x]][1])
                ptr[advance_t] += 1
                continue
            per_term = {}
            doc_ids = set()
            for t in active:
                ps = self._postings_in_range(t, pivot_lo, pivot_hi)
                per_term[t] = ps
                for d, _ in ps:
                    doc_ids.add(d)
            for d in sorted(doc_ids):
                dl_row = c.execute(
                    "SELECT length FROM docs WHERE id=?", (d,)).fetchone()
                if not dl_row:
                    continue
                dl = float(dl_row[0] or 0.0)
                score = 0.0
                for t in active:
                    tf = 0
                    for (dd, tt) in per_term[t]:
                        if dd == d:
                            tf = tt
                            break
                    if tf == 0:
                        continue
                    k1 = self.k1
                    b = self.b
                    denom = tf + k1 * (1 - b + b * (dl / max(1.0, self.avgdl)))
                    score += idfs[t] * ((tf * (k1 + 1)) / max(1e-9, denom))
                if score > 0.0:
                    if len(top) < k:
                        heapq.heappush(top, (score, d))
                        theta = top[0][0]
                    elif score > top[0][0]:
                        heapq.heapreplace(top, (score, d))
                        theta = top[0][0]
            advance_t = min(active, key=lambda x: blocks[x][ptr[x]][1])
            ptr[advance_t] += 1
        top.sort(reverse=True)
        return [(d, s) for (s, d) in top]
