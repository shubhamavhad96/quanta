import re
import asyncio
import httpx
from typing import List, Dict
from duckduckgo_search import DDGS
from trafilatura import extract

UA = {"User-Agent": "QuantaBot/0.2"}
WIKI_API = "https://en.wikipedia.org/w/api.php"


def ddg_urls(q: str, limit: int = 8) -> List[str]:
    outs: List[str] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(q, max_results=limit):
                u = (r.get("href") or r.get("url") or "").strip()
                if u.startswith("http"):
                    outs.append(u)
    except Exception:
        outs = []
    # de-dupe
    seen, uniq = set(), []
    for u in outs:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq[:limit]


async def wiki_urls(q: str, limit: int = 8) -> List[str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": q,
        "srlimit": str(limit),
        "format": "json",
        "utf8": 1,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=UA) as c:
            r = await c.get(WIKI_API, params=params)
            r.raise_for_status()
            data = r.json()
        outs: List[str] = []
        for hit in data.get("query", {}).get("search", [])[:limit]:
            title = (hit.get("title") or "").replace(" ", "_")
            if title:
                outs.append(f"https://en.wikipedia.org/wiki/{title}")
        return outs
    except Exception:
        return []


async def discover_urls_robust(q: str, limit: int = 8) -> List[str]:
    # Try DDG first, then fill with Wikipedia
    ddg = ddg_urls(q, limit=limit)
    if len(ddg) < limit:
        wiki = await wiki_urls(q, limit=limit - len(ddg))
    else:
        wiki = []
    seen, outs = set(), []
    for u in ddg + wiki:
        if u not in seen:
            seen.add(u)
            outs.append(u)
    return outs[:limit]


async def fetch_meta(url: str) -> Dict | None:
    try:
        async with httpx.AsyncClient(timeout=12.0, headers=UA, follow_redirects=True) as c:
            r = await c.get(url)
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
            return None
        html = r.text
        m = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
        title = (m.group(1).strip() if m else url)[:140]
        text = extract(html, include_comments=False,
                       include_tables=False, favor_recall=True) or ""
        snippet = " ".join(text.split())[:500]
        return {"url": url, "title": title, "snippet": snippet}
    except Exception:
        return None


async def gather_meta(urls: List[str]) -> List[Dict]:
    tasks = [fetch_meta(u) for u in urls]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r]
