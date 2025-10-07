from .cache import LRUCacheTTL, PostingsCache
from fastapi import Query as FQuery
from fastapi import FastAPI, Query, HTTPException, Request, Path
from fastapi import Body
import os
import time
import logging
import numpy as np
from difflib import get_close_matches
from fastapi.middleware.cors import CORSMiddleware
from .indexer import InvertedIndex
from .models import SearchResponse, SearchResult
from .ranker import BM25Ranker
from .utils import highlight_snippet, tokenize, parse_query, positions_match_phrase, positions_match_near, best_passage_span, build_snippet_from_span, highlight_html_snippet, make_snippet
from .vectors import VectorReRanker
from .ann import ANNIndex
from .wand import BMWScorer
from .cache import LRUCacheTTL, PostingsCache, normalize_query
import math
from .crawler import fetch_and_parse, robots_ok, norm_url
import httpx
import asyncio
from .reranker import CrossReranker
from typing import List
import hashlib
import textwrap
from .ollama import generate as ollama_generate
from slugify import slugify
from .discover import discover_urls, ddg_urls, fetch_metadata
from pydantic import BaseModel
from typing import List, Optional
from .websearch import discover_urls_robust, gather_meta
from .ollama_client import ollama_generate

BASE = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE, "..", "data", "index", "quanta.sqlite")
RAW_PATH = os.path.join(BASE, "..", "data", "raw")

app = FastAPI(title="Quanta", version="0.8.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
_index = None
_ranker = None
_vec = None
_ann = None
_RATE = {"limit_per_min": 60, "bucket": {}}
_QUERY_COUNT = 0
# Caches
_QUERY_CACHE = LRUCacheTTL(capacity=256, ttl_sec=300)
_POSTINGS_CACHE = PostingsCache(capacity=512, ttl_sec=600)

# Feature flags / knobs
VEC_ENABLED = os.environ.get("QUANTA_VEC_ENABLED", "1") == "1"
VEC_WEIGHT = float(os.environ.get("QUANTA_VEC_WEIGHT", "0.35"))
HYBRID_ON = os.environ.get("QUANTA_HYBRID", "1") == "1"
ANN_K = int(os.environ.get("QUANTA_ANN_K", "80"))
BM25_POOL = int(os.environ.get("QUANTA_BM25_POOL", "80"))
FINAL_POOL = int(os.environ.get("QUANTA_FINAL_POOL", "80"))
BMW_ON = os.environ.get("QUANTA_BMW", "1") == "1"
RECENCY_HALFLIFE = float(os.environ.get("QUANTA_RECENCY_HALFLIFE_DAYS", "180"))
RECENCY_WEIGHT = float(os.environ.get("QUANTA_RECENCY_WEIGHT", "0.20"))


def _recency_boost(last_modified: int | None, crawled_at: int | None,
                   half_life_days: float, weight: float) -> float:
    ts = last_modified or crawled_at or int(time.time())
    age_days = max(0.0, (time.time() - float(ts)) / 86400.0)
    base = 0.5 + 0.5 * math.exp(-age_days / max(1e-6, half_life_days))
    return (1.0 - weight) + weight * base


# ANN persistence paths
ANN_INDEX_PATH = os.path.join(BASE, "..", "data", "index", "quanta.ann.bin")
ANN_IDS_PATH = os.path.join(BASE, "..", "data", "index", "quanta.ann.ids.npy")


def get_vec():
    global _vec
    if _vec is None:
        _vec = VectorReRanker(weight=VEC_WEIGHT)
    return _vec


def get_ann():
    global _ann
    if _ann is None:
        _ann = ANNIndex(dim=384, space="cosine")
    # auto-load if files exist and not yet ready
    if not _ann.is_ready() and os.path.exists(ANN_INDEX_PATH) and os.path.exists(ANN_IDS_PATH):
        try:
            _ann.load(ANN_INDEX_PATH, ANN_IDS_PATH)
        except Exception:
            pass
    return _ann


def allow_request(ip: str) -> bool:
    now = int(time.time() // 60)
    b = _RATE["bucket"].get(ip)
    if not b or b["ts"] != now:
        _RATE["bucket"][ip] = {"ts": now, "count": 1}
        return True
    if b["count"] >= _RATE["limit_per_min"]:
        return False
    b["count"] += 1
    return True


def get_index():
    # return a fresh connection per request to avoid cross-thread sqlite issues
    return InvertedIndex(DB_PATH)


# remove cached ranker and always build fresh per request to avoid cross-thread sqlite issues
# _ranker = None


def get_ranker():
    idx = get_index()
    return BM25Ranker(idx.conn)


@app.on_event("shutdown")
def _shutdown():
    global _index
    if _index:
        _index.close()


@app.get("/health")
def health(): return {"ok": True}


@app.post("/index")
def build_index(folder: str | None = None):
    idx = get_index()
    idx.build_from_folder(folder or RAW_PATH)
    return {"message": "indexed", **idx.stats()}


# Wipe-aware reindex


@app.post("/reindex")
def reindex(folder: str | None = None, wipe: bool = FQuery(False)):
    global _index, _ranker, _vec, _ann
    idx = get_index()
    if wipe:
        db = DB_PATH
        idx.wipe_file(db)
        _index = None
        _ranker = None
        _vec = None
        _ann = None
        idx = get_index()

    idx.reset()
    idx.build_from_folder(folder or RAW_PATH)
    idx.rebuild_blocks(block_size=128)
    # no cached ranker; built fresh on each call
    # _ranker = BM25Ranker(idx.conn)
    return {"message": "reindexed", **idx.stats(), "wiped": wipe}


@app.get("/stats")
def stats():
    idx = get_index()
    return {"index": idx.stats(), "query_count": _QUERY_COUNT, "rate_limit_per_min": _RATE["limit_per_min"], "cache": {"q_size": len(_QUERY_CACHE.data), "p_size": len(_POSTINGS_CACHE.data)}}


@app.post("/embed")
def embed_all():
    idx = get_index()
    vec = get_vec()
    count = 0
    for doc_id, path, title in idx.iter_docs():
        if idx.get_vector(doc_id) is not None:
            continue
        t, text = idx.read_doc_text(doc_id)
        emb = vec.embed([(t + " " + (text[:500] if text else "")).strip()])[0]
        idx.upsert_vector(doc_id, emb)
        count += 1
    return {"message": "embedded", "new_vectors": count}


# ---- ANN endpoints ----
@app.post("/ann/build")
def ann_build():
    idx = get_index()
    ann = get_ann()
    vecs = []
    ids = []
    for doc_id, path, title in idx.iter_docs():
        v = idx.get_vector(doc_id)
        if v is None:
            continue
        vecs.append(v)
        ids.append(doc_id)
    if not vecs:
        raise HTTPException(
            status_code=400, detail="No vectors found. Run /embed first.")
    X = np.vstack(vecs).astype("float32")
    ids_arr = np.array(ids, dtype=np.int32)
    ann.create(num_elements=X.shape[0], M=32, efC=200)
    ann.add(X, labels=np.arange(X.shape[0]))
    os.makedirs(os.path.dirname(ANN_INDEX_PATH), exist_ok=True)
    ann.save(ANN_INDEX_PATH, ANN_IDS_PATH, ids_arr)
    return {"message": "ann built", "count": int(X.shape[0]), "index_path": ANN_INDEX_PATH}


@app.post("/ann/load")
def ann_load():
    ann = get_ann()
    if not (os.path.exists(ANN_INDEX_PATH) and os.path.exists(ANN_IDS_PATH)):
        raise HTTPException(
            status_code=404, detail="ANN files not found; build with /ann/build")
    ann.load(ANN_INDEX_PATH, ANN_IDS_PATH)
    return {"message": "ann loaded", "index_path": ANN_INDEX_PATH}


@app.get("/ann/stats")
def ann_stats():
    ann = get_ann()
    ready = ann.is_ready()
    return {"ready": ready, "path": ANN_INDEX_PATH if ready else None, "k_default": ANN_K}


@app.get("/doc/{doc_id}")
def get_doc(doc_id: int = Path(..., ge=1)):
    idx = get_index()
    row = idx.get_doc_meta(doc_id)
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    path, title = row
    t, text = idx.read_doc_text(doc_id)
    return {
        "doc_id": doc_id,
        "path": path,
        "title": t or title,
        "score": 0.0,
        "snippet": "",
        "full_text": text,
    }


@app.get("/suggest")
def suggest(q: str = Query(..., min_length=1, max_length=40), limit: int = Query(5, ge=1, le=20)):
    idx = get_index()
    return {"items": idx.suggest_terms(q, limit=limit)}


@app.get("/spell")
def spell(q: str = Query(..., min_length=1, max_length=40)):
    idx = get_index()
    terms = idx.all_terms()
    matches = get_close_matches(q.lower(), terms, n=5, cutoff=0.8)
    return {"input": q, "candidates": matches}


@app.get("/search", response_model=SearchResponse)
async def search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=200),
    page: int = Query(1, ge=1, le=100000),
    per_page: int = Query(10, ge=1, le=50),
    site: str | None = Query(
        None, description="domain filter, e.g., example.com"),
    lang: str | None = Query(None, description="language filter, e.g., en"),
    type: str | None = Query(
        None, description="mime filter, e.g., text/html or text/plain"),
    discover: bool = Query(True, description="discover web results when thin")
):
    ip = request.client.host if request.client else "unknown"
    if not allow_request(ip):
        raise HTTPException(status_code=429, detail="Too Many Requests")
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query required")

    t0 = time.time()
    try:
        idx = get_index()
        ranker = get_ranker()

        qobj = parse_query(q)
        q_terms = tokenize(q)

        # ---- Hybrid recall pool with BMW + cache ----
        from .cache import normalize_query
        q_norm = normalize_query(q)
        ranked_pool = None
        if page == 1 and per_page == 10:
            cached = _QUERY_CACHE.get(q_norm)
        else:
            cached = None
        if cached is not None:
            ranked_pool = cached
        else:
            use_bmw = BMW_ON and not ('"' in q or 'NEAR/' in q.upper())
            if use_bmw:
                bmw = BMWScorer(get_index().conn)
                ranked_pool = bmw.search(q_terms, k=max(BM25_POOL, FINAL_POOL))
            else:
                ranked_pool, _ = ranker.search_page(
                    q, page=1, per_page=max(BM25_POOL, FINAL_POOL))
            if page == 1 and per_page == 10:
                _QUERY_CACHE.set(q_norm, ranked_pool)

        bm25_map = {d: s for d, s in ranked_pool}
        pool_ids = set(bm25_map.keys())

        # ANN union (hybrid)
        if HYBRID_ON and VEC_ENABLED and get_ann().is_ready():
            vec = get_vec()
            q_emb = vec.embed([q])[0]
            ann_hits = get_ann().query(q_emb, k=ANN_K)
            for d, sim in ann_hits:
                if d not in pool_ids:
                    bm25_map.setdefault(d, 0.0)
                    pool_ids.add(d)

        pool_ids_list = list(pool_ids)[:FINAL_POOL]
        pool = [(d, bm25_map.get(d, 0.0)) for d in pool_ids_list]

        # Fallback: if pool is empty, try ANN-only or direct vector sim over all docs
        if VEC_ENABLED and not pool:
            vec = get_vec()
            q_emb = vec.embed([q])[0]
            if get_ann().is_ready():
                ann_hits = get_ann().query(q_emb, k=max(ANN_K, FINAL_POOL))
                pool = [(d, 0.0) for d, _ in ann_hits]
            else:
                import numpy as _np
                ids = []
                embs = []
                for doc_id, _p, _t in get_index().iter_docs():
                    v = get_index().get_vector(doc_id)
                    if v is not None:
                        ids.append(doc_id)
                        embs.append(v)
                if embs:
                    X = _np.vstack(embs).astype("float32")
                    sims = X @ q_emb
                    top_idx = sims.argsort()[::-1][:FINAL_POOL]
                    pool = [(int(ids[i]), 0.0) for i in top_idx]

        # Optional vector blend re-rank
        if VEC_ENABLED and pool:
            vec = get_vec()
            pool = vec.rerank(
                q, pool, fetch_text_fn=lambda d: idx.read_doc_text(d))

        # positions-based filtering for phrases and NEAR/k
        needed_terms = set(qobj["terms"]) | set(t for ph in qobj["phrases"] for t in tokenize(
            ph)) | set(x for abk in qobj["nears"] for x in (abk[0], abk[1]))

        def doc_passes(doc_id: int):
            cur = idx.conn.cursor()
            posmap = {}
            for t in needed_terms:
                row = cur.execute(
                    "SELECT positions FROM postings WHERE term=? AND doc_id=?", (t, doc_id)).fetchone()
                if row and row[0]:
                    try:
                        posmap[t] = list(map(int, row[0].split(",")))
                    except Exception:
                        posmap[t] = []
            for ph in qobj["phrases"]:
                words = tokenize(ph)
                if not words or any(w not in posmap for w in words):
                    return False, None
                plists = [sorted(posmap[w]) for w in words]
                s0 = set(plists[0])
                ok = True
                for shift, pl in enumerate(plists[1:], start=1):
                    sset = set(pl)
                    s0 = {p for p in s0 if (p + shift) in sset}
                    if not s0:
                        ok = False
                        break
                if not ok:
                    return False, None
                first_start = min(s0) if s0 else None
                return True, first_start
            for (a, b, k) in qobj["nears"]:
                if a not in posmap or b not in posmap:
                    return False, None
                if not positions_match_near(sorted(posmap[a]), sorted(posmap[b]), k):
                    return False, None
            first_start = None
            for t in needed_terms:
                if t in posmap and posmap[t]:
                    first_start = min(posmap[t])
                    break
            return True, first_start

        filtered = []
        pass_starts = {}
        for (d, s) in pool:
            ok, start_pos = doc_passes(d)
            if ok:
                filtered.append((d, s))
                pass_starts[d] = start_pos

        # metadata filters
        if site or lang or type:
            filtered = [(d, s) for (d, s)
                        in filtered if idx.filter_doc(d, site, lang, type)]

        # facets on filtered pool
        pool_ids_filtered = [d for (d, _) in filtered]
        facets = idx.facet_counts(pool_ids_filtered, limit=6)

        # paginate
        start_idx = max(0, (page - 1) * per_page)
        ranked_page = filtered[start_idx:start_idx + per_page]

        # build snippets around best passage window
        results = []
        for doc_id, score in ranked_page:
            path, title = idx.get_doc_meta(doc_id)
            try:
                with open(path, "r", errors="ignore") as f:
                    raw = f.read()
            except Exception:
                raw = ""
            q_terms = tokenize(q)
            title_clean, text_clean = idx.read_doc_text(doc_id)
            base_text = text_clean or raw
            # recency boost
            meta = idx.conn.execute(
                "SELECT last_modified, crawled_at FROM docs WHERE id= ?", (
                    doc_id,)
            ).fetchone()
            lm, ca = (meta or (None, None))
            boost = _recency_boost(lm, ca, RECENCY_HALFLIFE, RECENCY_WEIGHT)
            adj_score = float(score) * float(boost)
            if base_text:
                cs, ce = best_passage_span(base_text, q_terms, win_tokens=60)
                snip = build_snippet_from_span(base_text, cs, ce, pad=80)
                snip = highlight_html_snippet(snip, q_terms)
            else:
                snip = highlight_html_snippet(make_snippet(raw), q_terms)
            results.append(SearchResult(
                doc_id=doc_id, path=path, title=title, score=round(float(adj_score), 4), snippet=snip
            ))

        total_docs = idx.stats()["total_docs"]

        # ---- Web discovery if few results ----
        web_links = []
        if discover and len(results) < 5:
            try:
                urls = ddg_urls(q, limit=8)
                web_links = await fetch_metadata(urls)
                # Optionally ingest a few serially to grow local corpus later
                for u in urls[:2]:
                    try:
                        await ingest_url(u)
                    except Exception:
                        pass
            except Exception:
                web_links = []

        # ---- AI Answer from local passages + web snippets ----
        ai_text = None
        try:
            passages = []
            q_terms2 = tokenize(q)
            for r in results[:3]:
                t, text = idx.read_doc_text(r.doc_id)
                if text:
                    cs, ce = best_passage_span(text, q_terms2, win_tokens=80)
                    span = text[max(0, cs-140):min(len(text), ce+140)]
                    passages.append((len(passages)+1, t or r.title, span))
            for w in web_links[:3]:
                passages.append((len(passages)+1, w["title"], w["snippet"]))
            if passages:
                ctx = "\n\n".join(f"[{i}] {t}\n{s}" for (i, t, s) in passages)
                prompt = f"""Answer the question using ONLY the context. If insufficient, say so.\n\nQuestion: {q}\n\nContext:\n{ctx}\n\nRules:\n- 2–4 concise sentences.\n- Neutral tone.\n- Add citation markers like [1], [2].\n- No info outside the context.\n\nAnswer:"""
                ai_text = ollama_generate(prompt)
        except Exception:
            ai_text = None

        dt = (time.time() - t0) * 1000
        global _QUERY_COUNT
        _QUERY_COUNT += 1
        logging.info(
            f'q="{q}" page={page} per_page={per_page} hits={len(filtered)}/{total_docs} latency_ms={dt:.1f} ip={ip} vec={VEC_ENABLED} hybrid={HYBRID_ON}')
        return SearchResponse(
            query=q,
            total_docs=total_docs,
            total_hits=len(filtered),
            page=page,
            per_page=per_page,
            results=results,
            facets=facets,
            filters={"site": site, "lang": lang, "type": type},
            web_results=web_links[:8],
            ai=ai_text
        )
    except Exception as e:
        logging.exception("/search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dedup/stats")
def dedup_stats():
    c = get_index().conn.cursor()
    sigs = c.execute("SELECT COUNT(*) FROM signatures").fetchone()[0]
    dups = c.execute("SELECT COUNT(*) FROM duplicates").fetchone()[0]
    return {"signatures": sigs, "duplicates": dups}


@app.get("/dedup/examples")
def dedup_examples(limit: int = 5):
    c = get_index().conn.cursor()
    rows = c.execute(
        "SELECT path, dup_of_doc_id, reason FROM duplicates LIMIT ?", (limit,)).fetchall()
    return {"items": [{"path": p, "dup_of_doc_id": d, "reason": r} for (p, d, r) in rows]}


_rerank = None


def get_rerank():
    global _rerank
    if _rerank is None:
        _rerank = CrossReranker()
    return _rerank


@app.post("/ingest/url")
async def ingest_url(url: str):
    url = norm_url(url)
    async with httpx.AsyncClient() as client:
        if not await robots_ok(client, url):
            raise HTTPException(403, detail="Blocked by robots.txt")
        parsed = await fetch_and_parse(client, url)
    if not parsed or not parsed["text"]:
        raise HTTPException(400, detail="No extractable text")
    import hashlib
    import os
    h = hashlib.md5(url.encode()).hexdigest()[:16]
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    fpath = os.path.join(raw_dir, f"{h}.html.txt")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(parsed["title"]+"\n\n"+parsed["text"])
    idx = get_index()
    idx.build_from_folder(raw_dir)
    return {"message": "ingested", "url": url, "path": fpath}


@app.post("/crawl/seed")
async def crawl_seed(seeds: list[str], max_pages: int = 50, same_host_only: bool = True):
    from urllib.parse import urlparse
    seen: set[str] = set()
    q: List[str] = [norm_url(s) for s in seeds]
    async with httpx.AsyncClient() as client:
        while q and len(seen) < max_pages:
            u = q.pop(0)
            if u in seen:
                continue
            if same_host_only and urlparse(u).netloc != urlparse(seeds[0]).netloc:
                continue
            if not await robots_ok(client, u):
                seen.add(u)
                continue
            parsed = await fetch_and_parse(client, u)
            seen.add(u)
            if parsed and parsed["text"]:
                await ingest_url(u)
                for v in parsed["links"]:
                    if v not in seen:
                        q.append(v)
    return {"crawled": len(seen)}


def _top_passages_for_q(idx, q: str, top_k: int = 6):
    ranker = get_ranker()
    pool, _ = ranker.search_page(q, page=1, per_page=50)
    doc_ids = [d for d, _ in pool][:top_k]
    passages, sources = [], []
    from .utils import tokenize as _tok, best_passage_span as _bps
    for i, d in enumerate(doc_ids, start=1):
        path, title = idx.get_doc_meta(d)
        t, text = idx.read_doc_text(d)
        if not text:
            continue
        q_terms = _tok(q)
        cs, ce = _bps(text, q_terms, win_tokens=80)
        span = text[max(0, cs-120):min(len(text), ce+120)].strip()
        passages.append((i, d, (t or title or path), span))
        sources.append({"num": i, "doc_id": d, "title": (
            t or title or path), "path": path})
    return passages, sources


def _answer_prompt(question: str, passages: list[tuple[int, int, str, str]]) -> str:
    context = "\n\n".join(
        f"[{num}] {title}\n{text}" for (num, _, title, text) in passages
    )
    return f"""You are a careful assistant. Answer the question ONLY using the context below.
If the context is insufficient, say you don't have enough information.

Question: {question}

Context:
{context}

Instructions:
- Write 2 to 4 concise sentences.
- Use neutral, factual tone.
- Add citation markers like [1], [2], etc. at the end of the sentences where relevant.
- Do NOT invent facts beyond the context.

Answer:"""


@app.post("/answer")
def answer(q: str = Body(..., embed=True), k: int = 6):
    idx = get_index()
    passages, sources = _top_passages_for_q(idx, q, top_k=k)
    if not passages:
        return {"answer": "I don’t have enough evidence to answer that from the indexed content.", "sources": []}
    prompt = _answer_prompt(q, passages)
    try:
        resp = ollama_generate(prompt)
    except Exception as e:
        return {"answer": f"Model error: {e}", "sources": []}
    if len(resp) < 20:
        first = passages[0][3]
        resp = (first[:350] + "…") if len(first) > 350 else first
    return {"answer": resp, "sources": sources}


@app.get("/ai/{slug}")
def ai_virtual(slug: str):
    idx = get_index()
    row = idx.get_virtual(slug)
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    vid, title, body = row
    return {"title": title, "body": body, "slug": slug}


@app.post("/ensure_results")
async def ensure_results(q: str, min_hits: int = 5, discover_limit: int = 12):
    idx = get_index()
    ranker = get_ranker()
    pool, _ = ranker.search_page(q, page=1, per_page=50)
    hits = len(pool)
    added_urls: list[str] = []
    discovered: list[str] = []
    if hits < min_hits:
        discovered = await discover_urls(q, limit_total=discover_limit)
        # NOTE: Skip direct ingestion here to avoid cross-thread SQLite errors under async.
        # Users can POST /ingest/url for any of the returned URLs from the client.
    pool2, _ = ranker.search_page(q, page=1, per_page=50)
    hits2 = len(pool2)
    virtual_slug = None
    if hits2 == 0:
        from .ollama import generate as ollama_generate
        prompt = f"You are writing a brief neutral overview for query: {q}. Write 3-4 sentences and 3 bullets."
        try:
            body = ollama_generate(prompt)
        except Exception:
            body = f"Overview for {q} is unavailable right now."
        title = f"{q} — Overview"
        slug = slugify(q)[:60] or "ai"
        idx.upsert_virtual(slug, title, body)
        virtual_slug = slug
    return {"before": hits, "after": max(hits2, hits), "discovered": discovered, "virtual": virtual_slug}


class WebLink(BaseModel):
    url: str
    title: str
    snippet: str


class WebSearchResponse(BaseModel):
    query: str
    ai: Optional[str] = None
    web_results: List[WebLink]


@app.get("/search_web", response_model=WebSearchResponse)
async def search_web(q: str = Query(..., min_length=1, max_length=200), k: int = 8):
    try:
        urls = await discover_urls_robust(q, limit=k)
    except Exception:
        urls = []
    try:
        links = await gather_meta(urls)
    except Exception:
        links = []

    # Rank snippets to favor programming sense when ambiguous
    def prog_score(w: dict) -> int:
        title = (w.get("title") or "").lower()
        snip = (w.get("snippet") or "").lower()
        hits = 0
        for kw in ["programming", "software", "platform", "language", "jvm", "java community process"]:
            if kw in title or kw in snip:
                hits += 1
        # down-rank obvious non-programming signals
        for neg in ["island", "indonesia", "coffee"]:
            if neg in title or neg in snip:
                hits -= 1
        return hits

    links_sorted = sorted(links, key=prog_score, reverse=True)

    # ------- AI answer from scraped snippets (grounded) -------
    ai = None
    if links_sorted:
        ctx_parts = []
        for i, w in enumerate(links_sorted[:4], start=1):  # tighter context
            title = (w.get("title") or w["url"])[:140]
            snippet = (w.get("snippet") or "")[:450]
            ctx_parts.append(f"[{i}] {title}\n{snippet}")
        ctx = "\n\n".join(ctx_parts)
        steer = "Assume the user means the computing/software sense unless they clearly ask about the island or coffee."
        prompt = f"""Answer the question using ONLY the context. If insufficient, say so.

Intent: {steer}

Question: {q}

Context:
{ctx}

Rules:
- 2–4 concise sentences.
- Neutral tone.
- Add citation markers like [1], [2] corresponding to the context items above.
- Do not invent facts outside the context.

Answer:"""
        try:
            text = ollama_generate(prompt).strip()
            ai = text if len(
                text) >= 20 else "I don’t have enough evidence from these pages to answer confidently."
        except Exception as e:
            ai = f"(AI unavailable: {e})"

    return WebSearchResponse(query=q, ai=ai, web_results=[WebLink(**w) for w in links_sorted])
