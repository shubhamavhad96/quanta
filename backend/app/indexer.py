import os
import math
import sqlite3
import time
import mimetypes
import tldextract
from collections import Counter
from typing import Dict
from simhash import Simhash
from langdetect import detect, DetectorFactory
from .utils import tokenize, html_to_text, make_snippet

DetectorFactory.seed = 0

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS docs(
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE,
  title TEXT,
  length INTEGER,
  last_modified INTEGER,
  crawled_at INTEGER,
  domain TEXT,
  lang TEXT,
  mime TEXT
);
CREATE INDEX IF NOT EXISTS idx_docs_domain ON docs(domain);
CREATE INDEX IF NOT EXISTS idx_docs_lang ON docs(lang);
CREATE INDEX IF NOT EXISTS idx_docs_mime ON docs(mime);
CREATE TABLE IF NOT EXISTS terms(term TEXT PRIMARY KEY, df INTEGER);
CREATE TABLE IF NOT EXISTS postings(term TEXT, doc_id INTEGER, tf INTEGER, positions TEXT, PRIMARY KEY(term, doc_id));
CREATE TABLE IF NOT EXISTS vectors(doc_id INTEGER PRIMARY KEY, vec BLOB);
CREATE TABLE IF NOT EXISTS term_blocks(
  term TEXT,
  start_doc_id INTEGER,
  end_doc_id INTEGER,
  max_tf INTEGER,
  PRIMARY KEY(term, start_doc_id)
);
CREATE TABLE IF NOT EXISTS signatures(
  doc_id INTEGER PRIMARY KEY,
  simhash INTEGER,
  band0 INTEGER,
  band1 INTEGER,
  band2 INTEGER,
  band3 INTEGER
);
CREATE INDEX IF NOT EXISTS idx_sig_band0 ON signatures(band0);
CREATE INDEX IF NOT EXISTS idx_sig_band1 ON signatures(band1);
CREATE INDEX IF NOT EXISTS idx_sig_band2 ON signatures(band2);
CREATE INDEX IF NOT EXISTS idx_sig_band3 ON signatures(band3);
CREATE TABLE IF NOT EXISTS duplicates(
  path TEXT PRIMARY KEY,
  dup_of_doc_id INTEGER,
  reason TEXT
);
-- virtual AI docs
CREATE TABLE IF NOT EXISTS virtual_docs(
  id INTEGER PRIMARY KEY,
  slug TEXT UNIQUE,
  title TEXT,
  body TEXT,
  created_at INTEGER
);
"""


def _simhash64(tokens) -> int:
    return Simhash(tokens, f=64).value


def _bands(sig: int):
    return (sig & 0xFFFF, (sig >> 16) & 0xFFFF, (sig >> 32) & 0xFFFF, (sig >> 48) & 0xFFFF)


def _hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _guess_mime(path: str, is_html: bool) -> str:
    if is_html:
        return "text/html"
    m, _ = mimetypes.guess_type(path)
    return m or "text/plain"


def _guess_domain_from_path(path: str) -> str:
    return "local"


class InvertedIndex:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA synchronous=OFF;")
        self.conn.executescript(SCHEMA)

    def close(self): self.conn.close()

    def _dup_candidates(self, band0: int, band1: int, band2: int, band3: int):
        cur = self.conn.cursor()
        rows = []
        for i, band in enumerate((band0, band1, band2, band3)):
            rows += cur.execute(
                f"SELECT doc_id, simhash FROM signatures WHERE band{i} = ?",
                (band,)
            ).fetchall()
        seen, out = set(), []
        for d, s in rows:
            if d in seen:
                continue
            seen.add(d)
            out.append((d, s))
        return out

    def _insert_signature(self, doc_id: int, sig: int):
        b0, b1, b2, b3 = _bands(sig)
        cur = self.conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO signatures(doc_id, simhash, band0, band1, band2, band3)
                       VALUES(?,?,?,?,?,?)""",
            (doc_id, sig, b0, b1, b2, b3)
        )
        self.conn.commit()

    def _index_doc(self, doc_id: int, path: str, text: str, given_title: str | None = None, last_modified: int | None = None, crawled_at: int | None = None):
        tokens = tokenize(text)
        tf = Counter(tokens)
        pos_map = {}
        for i, tok in enumerate(tokens):
            pos_map.setdefault(tok, []).append(i)
        length = sum(tf.values())
        title = given_title or os.path.basename(path)
        is_html = path.lower().endswith((".html", ".htm"))
        mime = _guess_mime(path, is_html)
        domain = _guess_domain_from_path(path)
        try:
            lang = detect(text) if text and len(text) > 40 else "unk"
        except Exception:
            lang = "unk"

        cur = self.conn.cursor()
        cur.execute(
            """INSERT OR IGNORE INTO docs(id, path, title, length, last_modified, crawled_at, domain, lang, mime)
                       VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, path, title, length, last_modified or 0,
             crawled_at or 0, domain, lang, mime)
        )
        for term, count in tf.items():
            cur.execute(
                """INSERT OR REPLACE INTO postings(term, doc_id, tf, positions)
                   VALUES(?, ?, ?, ?)""",
                (term, doc_id, count, ",".join(map(str, pos_map[term])))
            )
            cur.execute(
                "INSERT OR IGNORE INTO terms(term, df) VALUES(?, ?)", (term, 0)
            )
            cur.execute("UPDATE terms SET df = df + 1 WHERE term = ?", (term,))
        self.conn.commit()

    def build_from_folder(self, folder: str):
        cur = self.conn.cursor()
        doc_id = 1 + \
            (cur.execute(
                "SELECT COALESCE(MAX(id),0) FROM docs").fetchone()[0] or 0)
        for root, _, files in os.walk(folder):
            for fn in files:
                if not fn.lower().endswith((".txt", ".html", ".htm")):
                    continue
                path = os.path.join(root, fn)
                if cur.execute("SELECT 1 FROM docs WHERE path= ?", (path,)).fetchone():
                    continue
                try:
                    with open(path, "r", errors="ignore") as f:
                        raw = f.read()
                except Exception:
                    continue
                if fn.lower().endswith((".html", ".htm")):
                    title, text = html_to_text(raw)
                else:
                    title, text = os.path.basename(path), raw
                tokens = tokenize(text)
                sig = _simhash64(tokens)
                b0, b1, b2, b3 = _bands(sig)
                cands = self._dup_candidates(b0, b1, b2, b3)
                is_dup = False
                for (cand_doc_id, cand_sig) in cands:
                    try:
                        cs = int(cand_sig)
                    except Exception:
                        cs = int(cand_sig)
                    if _hamming64(sig, cs) <= 3:
                        cur.execute(
                            """INSERT OR REPLACE INTO duplicates(path, dup_of_doc_id, reason)
                                       VALUES(?,?,?)""",
                            (path, cand_doc_id, "simhash<=3")
                        )
                        self.conn.commit()
                        is_dup = True
                        break
                if is_dup:
                    continue
                mtime = int(os.path.getmtime(path)) if os.path.exists(
                    path) else int(time.time())
                now = int(time.time())
                self._index_doc(doc_id, path, text, given_title=title,
                                last_modified=mtime, crawled_at=now)
                self._insert_signature(doc_id, sig)
                doc_id += 1

    def stats(self):
        cur = self.conn.cursor()
        total_docs = cur.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        total_terms = cur.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
        return {"total_docs": total_docs, "total_terms": total_terms}

    def doc_meta(self, doc_id: int):
        cur = self.conn.cursor()
        return cur.execute("SELECT domain, lang, mime FROM docs WHERE id=?", (doc_id,)).fetchone()

    def filter_doc(self, doc_id: int, site: str | None, lang: str | None, mtype: str | None) -> bool:
        row = self.doc_meta(doc_id)
        if not row:
            return False
        domain, dlang, mime = row
        if site and (domain or "").lower() != site.lower():
            return False
        if lang and (dlang or "").lower() != lang.lower():
            return False
        if mtype and (mime or "").lower() != mtype.lower():
            return False
        return True

    def facet_counts(self, doc_ids: list[int], limit: int = 6):
        cur = self.conn.cursor()
        if not doc_ids:
            return {"domain": [], "lang": [], "type": []}
        placeholders = ",".join(["?"] * len(doc_ids))
        q1 = f"SELECT domain, COUNT(*) c FROM docs WHERE id IN ({placeholders}) GROUP BY domain ORDER BY c DESC LIMIT ?"
        q2 = f"SELECT lang, COUNT(*) c   FROM docs WHERE id IN ({placeholders}) GROUP BY lang   ORDER BY c DESC LIMIT ?"
        q3 = f"SELECT mime, COUNT(*) c   FROM docs WHERE id IN ({placeholders}) GROUP BY mime   ORDER BY c DESC LIMIT ?"
        dom = [{"value": d or "unknown", "count": c}
               for d, c in cur.execute(q1, (*doc_ids, limit))]
        lng = [{"value": l or "unknown", "count": c}
               for l, c in cur.execute(q2, (*doc_ids, limit))]
        typ = [{"value": t or "unknown", "count": c}
               for t, c in cur.execute(q3, (*doc_ids, limit))]
        return {"domain": dom, "lang": lng, "type": typ}

    def _idf(self, df: int, N: int): return math.log(1.0 + (N / (1.0 + df)))

    def search(self, query: str, k: int = 5):
        q_terms = tokenize(query)
        if not q_terms:
            return [], 0
        cur = self.conn.cursor()
        N = cur.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        if N == 0:
            return [], 0
        scores: Dict[int, float] = {}
        for t in q_terms:
            row = cur.execute(
                "SELECT df FROM terms WHERE term = ?", (t,)).fetchone()
            if not row:
                continue
            idf = self._idf(row[0], N)
            for doc_id, tf in cur.execute("SELECT doc_id, tf FROM postings WHERE term = ?", (t,)):
                length = cur.execute(
                    "SELECT length FROM docs WHERE id = ?", (doc_id,)).fetchone()[0]
                score = (tf * idf) / max(1.0, length ** 0.5)
                scores[doc_id] = scores.get(doc_id, 0.0) + score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for doc_id, score in ranked:
            path, title = cur.execute(
                "SELECT path, title FROM docs WHERE id= ?", (doc_id,)).fetchone()
            try:
                with open(path, "r", errors="ignore") as f:
                    text = f.read()
            except Exception:
                text = ""
            snippet = text.strip().replace("\n", " ")[:200]
            results.append((doc_id, path, title, score, snippet))
        return results, N

    def get_doc_meta(self, doc_id: int):
        cur = self.conn.cursor()
        return cur.execute("SELECT path, title FROM docs WHERE id= ?", (doc_id,)).fetchone()

    def reset(self):
        cur = self.conn.cursor()
        cur.execute("DROP TABLE IF EXISTS postings")
        cur.execute("DROP TABLE IF EXISTS terms")
        cur.execute("DROP TABLE IF EXISTS docs")
        cur.execute("DROP TABLE IF EXISTS vectors")
        cur.execute("DROP TABLE IF EXISTS term_blocks")
        cur.execute("DROP TABLE IF EXISTS signatures")
        cur.execute("DROP TABLE IF EXISTS duplicates")
        self.conn.commit()
        self.conn.executescript(
            """
        CREATE TABLE IF NOT EXISTS docs(
          id INTEGER PRIMARY KEY,
          path TEXT UNIQUE,
          title TEXT,
          length INTEGER,
          last_modified INTEGER,
          crawled_at INTEGER,
          domain TEXT,
          lang TEXT,
          mime TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_docs_domain ON docs(domain);
        CREATE INDEX IF NOT EXISTS idx_docs_lang ON docs(lang);
        CREATE INDEX IF NOT EXISTS idx_docs_mime ON docs(mime);
        CREATE TABLE IF NOT EXISTS terms(term TEXT PRIMARY KEY, df INTEGER);
        CREATE TABLE IF NOT EXISTS postings(term TEXT, doc_id INTEGER, tf INTEGER, positions TEXT, PRIMARY KEY(term, doc_id));
        CREATE TABLE IF NOT EXISTS vectors(doc_id INTEGER PRIMARY KEY, vec BLOB);
        CREATE TABLE IF NOT EXISTS term_blocks(
          term TEXT,
          start_doc_id INTEGER,
          end_doc_id INTEGER,
          max_tf INTEGER,
          PRIMARY KEY(term, start_doc_id)
        );
        CREATE TABLE IF NOT EXISTS signatures(
          doc_id INTEGER PRIMARY KEY,
          simhash INTEGER,
          band0 INTEGER,
          band1 INTEGER,
          band2 INTEGER,
          band3 INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_sig_band0 ON signatures(band0);
        CREATE INDEX IF NOT EXISTS idx_sig_band1 ON signatures(band1);
        CREATE INDEX IF NOT EXISTS idx_sig_band2 ON signatures(band2);
        CREATE INDEX IF NOT EXISTS idx_sig_band3 ON signatures(band3);
        CREATE TABLE IF NOT EXISTS duplicates(
          path TEXT PRIMARY KEY,
          dup_of_doc_id INTEGER,
          reason TEXT
        );
        """
        )
        self.conn.commit()

    def count_hits_for_terms(self, terms: list[str]) -> int:
        if not terms:
            return 0
        cur = self.conn.cursor()
        placeholders = ",".join(["?"] * len(terms))
        sql = f"""
        SELECT COUNT(DISTINCT doc_id) FROM postings WHERE term IN ({placeholders})
        """
        row = cur.execute(sql, terms).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def upsert_vector(self, doc_id: int, vec):
        import numpy as np
        cur = self.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO vectors(doc_id, vec) VALUES(?, ?)",
                    (doc_id, np.asarray(vec, dtype="float32").tobytes()))
        self.conn.commit()

    def get_vector(self, doc_id: int):
        import numpy as np
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT vec FROM vectors WHERE doc_id= ?", (doc_id,)).fetchone()
        if not row:
            return None
        buf = row[0]
        arr = np.frombuffer(buf, dtype="float32")
        return arr

    def iter_docs(self):
        cur = self.conn.cursor()
        for doc_id, path, title in cur.execute("SELECT id, path, title FROM docs"):
            yield doc_id, path, title

    def read_doc_text(self, doc_id: int):
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT path, title FROM docs WHERE id= ?", (doc_id,)).fetchone()
        if not row:
            return "", ""
        path, title = row
        try:
            with open(path, "r", errors="ignore") as f:
                raw = f.read()
        except Exception:
            return title, ""
        if path.lower().endswith((".html", ".htm")):
            from .utils import html_to_text
            t, text = html_to_text(raw)
            return (t or title), text
        return title, raw

    def wipe_file(self, db_path: str):
        self.close()
        try:
            os.remove(db_path)
        except FileNotFoundError:
            pass

    def suggest_terms(self, prefix: str, limit: int = 5):
        cur = self.conn.cursor()
        like = prefix.lower() + "%"
        rows = cur.execute(
            "SELECT term, df FROM terms WHERE term LIKE ? ORDER BY df DESC LIMIT ?", (like, limit)).fetchall()
        return [{"term": t, "df": df} for (t, df) in rows]

    def all_terms(self):
        cur = self.conn.cursor()
        return [t for (t,) in cur.execute("SELECT term FROM terms")]

    def rebuild_blocks(self, block_size: int = 128):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM term_blocks")
        for (term,) in cur.execute("SELECT term FROM terms"):
            rows = list(cur.execute(
                "SELECT doc_id, tf FROM postings WHERE term=? ORDER BY doc_id ASC", (
                    term,)
            ))
            if not rows:
                continue
            for i in range(0, len(rows), block_size):
                chunk = rows[i:i+block_size]
                start = chunk[0][0]
                end = chunk[-1][0]
                max_tf = max(tf for _, tf in chunk)
                cur.execute(
                    "INSERT OR REPLACE INTO term_blocks(term, start_doc_id, end_doc_id, max_tf) VALUES(?,?,?,?)",
                    (term, start, end, int(max_tf))
                )
        self.conn.commit()

    def upsert_virtual(self, slug: str, title: str, body: str) -> int:
        import time
        cur = self.conn.cursor()
        cur.execute("INSERT OR IGNORE INTO virtual_docs(slug,title,body,created_at) VALUES(?,?,?,?)",
                    (slug, title, body, int(time.time())))
        if cur.rowcount == 0:
            cur.execute(
                "UPDATE virtual_docs SET title=?, body=? WHERE slug=?", (title, body, slug))
        self.conn.commit()
        row = cur.execute(
            "SELECT id FROM virtual_docs WHERE slug=?", (slug,)).fetchone()
        return int(row[0]) if row else 0

    def get_virtual(self, slug: str):
        cur = self.conn.cursor()
        return cur.execute("SELECT id, title, body FROM virtual_docs WHERE slug=?", (slug,)).fetchone()
