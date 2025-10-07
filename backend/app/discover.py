import re
import httpx
from typing import List, Dict, Tuple
from duckduckgo_search import DDGS
from trafilatura import extract
import asyncio

WIKI_API = "https://en.wikipedia.org/w/api.php"


async def wiki_urls(query: str, limit: int = 5) -> List[str]:
    params = {
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": str(limit), "format": "json", "utf8": 1
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(WIKI_API, params=params)
        r.raise_for_status()
        data = r.json()
    outs: List[str] = []
    for hit in data.get("query", {}).get("search", [])[:limit]:
        title = hit.get("title", "{}").replace(" ", "_")
        outs.append(f"https://en.wikipedia.org/wiki/{title}")
    return outs


def ddg_urls(query: str, limit: int = 8) -> List[str]:
    outs: List[str] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=limit):
            u = (r.get("href") or r.get("url") or "").strip()
            if u.startswith("http"):
                outs.append(u)
    seen, uniq = set(), []
    for u in outs:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq[:limit]


async def _fetch(client: httpx.AsyncClient, url: str) -> Tuple[str, str]:
    r = await client.get(url, timeout=12, follow_redirects=True, headers={"User-Agent": "QuantaBot/0.1"})
    if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
        return ("", "")
    html = r.text
    m = re.search(r"<title>(.*?)</title>", html, flags=re.I | re.S)
    title = (m.group(1).strip() if m else url)[:140]
    text = extract(html, include_comments=False,
                   include_tables=False, favor_recall=True) or ""
    snippet = " ".join(text.split())[:400]
    return (title, snippet)


async def fetch_metadata(urls: List[str], max_concurrency: int = 5) -> List[Dict]:
    results: List[Dict] = []
    sem = httpx.AsyncClient.limits_class(
        max_keepalive_connections=max_concurrency, max_connections=max_concurrency)
    async with httpx.AsyncClient(limits=sem) as client:
        async def task(u: str):
            try:
                title, snippet = await _fetch(client, u)
                if title:
                    results.append(
                        {"url": u, "title": title, "snippet": snippet})
            except Exception:
                pass
        await asyncio.gather(*(task(u) for u in urls))
    return results


async def discover_urls(query: str, limit_total: int = 12) -> List[str]:
    wiki = await wiki_urls(query, limit=5)
    ddg = ddg_urls(query, limit=max(0, limit_total - len(wiki)))
    seen, outs = set(), []
    for u in wiki + ddg:
        if u not in seen:
            seen.add(u)
            outs.append(u)
    return outs[:limit_total]
