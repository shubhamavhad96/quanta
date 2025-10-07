import asyncio
import time
import re
import urllib.parse
import httpx
from dataclasses import dataclass, field
from typing import Optional, Set, Dict, List
from trafilatura import extract
from urllib.parse import urljoin, urlparse

ROBOT_CACHE: Dict[str, dict] = {}
LAST_HIT: Dict[str, float] = {}
CRAWL_DELAY_SEC = 2.0
USER_AGENT = "QuantaBot/0.1 (+https://example.local)"
MAX_PAGES = 200


def norm_url(u: str) -> str:
    pu = urllib.parse.urlsplit(u)
    host = pu.hostname.lower() if pu.hostname else ""
    scheme = pu.scheme.lower()
    path = pu.path or "/"
    query = "&".join(sorted(urllib.parse.parse_qsl(pu.query)))
    return urllib.parse.urlunsplit((scheme, host, path, query, ""))


async def robots_ok(client: httpx.AsyncClient, url: str) -> bool:
    host = urlparse(url).netloc
    if host not in ROBOT_CACHE:
        try:
            r = await client.get(f"{urlparse(url).scheme}://{host}/robots.txt", timeout=10)
            ROBOT_CACHE[host] = {"text": r.text,
                                 "allow_all": r.status_code >= 400}
        except Exception:
            ROBOT_CACHE[host] = {"text": "", "allow_all": True}
    rob = ROBOT_CACHE[host]
    if rob["allow_all"]:
        return True
    disallow = []
    for line in rob["text"].splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("user-agent:") and "*" in line:
            pass
        if line.lower().startswith("disallow:"):
            disallow.append(line.split(":")[1].strip())
    path = urlparse(url).path or "/"
    return not any(path.startswith(d) for d in disallow if d)


async def polite_wait(url: str):
    host = urlparse(url).netloc
    now = time.time()
    last = LAST_HIT.get(host, 0.0)
    delta = now - last
    if delta < CRAWL_DELAY_SEC:
        await asyncio.sleep(CRAWL_DELAY_SEC - delta)
    LAST_HIT[host] = time.time()


def extract_links(base_url: str, html: str) -> List[str]:
    hrefs = re.findall(r'href=["\'](.*?)["\']', html, flags=re.I)
    outs = []
    for h in hrefs:
        if h.startswith("javascript:") or h.startswith("mailto:"):
            continue
        try:
            out = norm_url(urljoin(base_url, h))
            if out.startswith("http"):
                outs.append(out)
        except Exception:
            continue
    return outs


async def fetch_and_parse(client: httpx.AsyncClient, url: str):
    await polite_wait(url)
    r = await client.get(url, headers={"User-Agent": USER_AGENT}, timeout=20, follow_redirects=True)
    if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
        return None
    html = r.text
    text = extract(html, include_comments=False,
                   include_tables=False, favor_recall=True) or ""
    title = re.search(r"<title>(.*?)</title>", html, flags=re.I | re.S)
    title = (title.group(1).strip() if title else url)[:200]
    links = extract_links(url, html)
    return {"url": url, "title": title, "text": text, "links": links}
