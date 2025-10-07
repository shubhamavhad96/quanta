import re
import asyncio
import httpx
from typing import List, Dict
from duckduckgo_search import DDGS
from trafilatura import extract
import json
import datetime
import tldextract
from email.utils import parsedate_to_datetime

UA = {"User-Agent": "QuantaBot/0.2"}
WIKI_API = "https://en.wikipedia.org/w/api.php"


def domain_of(url: str) -> str:
    ex = tldextract.extract(url)
    return ".".join(p for p in [ex.domain, ex.suffix] if p)


def display_name(url: str) -> str:
    ex = tldextract.extract(url)
    brand = ex.domain.capitalize() if ex.domain else url
    return brand


def favicon_for(url: str) -> str:
    dom = domain_of(url)
    return f"https://icons.duckduckgo.com/ip3/{dom}.ico"


def _parse_meta_time(html: str) -> str | None:
    for prop in [
        "article:published_time",
        "article:modified_time",
        "og:updated_time",
        "og:published_time",
    ]:
        m = re.search(
            rf'<meta[^>]+property=["\']{prop}["\'][^>]*content=["\']([^"\']+)["\']',
            html,
            flags=re.I,
        )
        if m:
            return m.group(1)
    for name in ["date", "pubdate"]:
        m = re.search(
            rf'<meta[^>]+name=["\']{name}["\'][^>]*content=["\']([^"\']+)["\']',
            html,
            flags=re.I,
        )
        if m:
            return m.group(1)
    m = re.search(r'<time[^>]+datetime=["\']([^"\']+)["\']', html, flags=re.I)
    if m:
        return m.group(1)
    for m in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        flags=re.I | re.S,
    ):
        try:
            data = json.loads(m.group(1))
        except Exception:
            continue

        def _find_dates(obj):
            if isinstance(obj, dict):
                for k in ["datePublished", "dateModified", "uploadDate"]:
                    if k in obj and isinstance(obj[k], str):
                        return obj[k]
                for v in obj.values():
                    x = _find_dates(v)
                    if x:
                        return x
            elif isinstance(obj, list):
                for v in obj:
                    x = _find_dates(v)
                    if x:
                        return x
            return None

        dt = _find_dates(data)
        if dt:
            return dt
    return None


def _to_iso(dt_str: str | None, header_last_modified: str | None) -> str | None:
    if header_last_modified:
        try:
            return (
                parsedate_to_datetime(header_last_modified)
                .astimezone(datetime.timezone.utc)
                .isoformat()
            )
        except Exception:
            pass
    if dt_str:
        try:
            return (
                datetime.datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                .astimezone(datetime.timezone.utc)
                .isoformat()
            )
        except Exception:
            try:
                return (
                    parsedate_to_datetime(dt_str)
                    .astimezone(datetime.timezone.utc)
                    .isoformat()
                )
            except Exception:
                return None
    return None


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
        title = (m.group(1).strip() if m else url)[:180]
        text = extract(html, include_comments=False,
                       include_tables=False, favor_recall=True) or ""
        snippet = " ".join(text.split())[:550]
        site = display_name(url)
        domain = domain_of(url)
        favicon = favicon_for(url)
        meta_dt = _parse_meta_time(html)
        last_mod = r.headers.get("Last-Modified")
        iso = _to_iso(meta_dt, last_mod)
        return {
            "url": url,
            "title": title,
            "snippet": snippet,
            "site": site,
            "domain": domain,
            "favicon": favicon,
            "published_at": iso,
        }
    except Exception:
        return None


async def gather_meta(urls: List[str]) -> List[Dict]:
    tasks = [fetch_meta(u) for u in urls]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r]
