from bs4 import BeautifulSoup
import re
from typing import List
import html
import shlex
from typing import Tuple

STOP = set("""
a an the and or of to in is are for on with as by at from this that these those be been being it its into
""".split())
_token_re = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    words = [w.lower() for w in _token_re.findall(text)]
    return [w for w in words if w not in STOP and len(w) > 1]


def html_to_text(html: str) -> tuple[str, str]:
    """
    Returns (title, clean_text) from raw HTML using BeautifulSoup.
    Keeps readable text, drops scripts/styles/nav.
    """
    soup = BeautifulSoup(html, "lxml")

    # title
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # remove noise
    for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "form"]):
        tag.decompose()

    # get main text
    text = soup.get_text(separator=" ", strip=True)
    return title, text


def make_snippet(text: str, max_len: int = 220) -> str:
    s = " ".join(text.split())
    return s[:max_len]


def highlight_snippet(text: str, terms: list[str], max_len: int = 220) -> str:
    """
    Returns HTML-safe snippet with <mark>...</mark> around query terms.
    """
    safe = html.escape(" ".join(text.split()))
    if not terms:
        return safe[:max_len]
    pattern = r"(" + "|".join(re.escape(t) for t in set(terms) if t) + r")"
    def _wrap(m): return f"<mark>{m.group(0)}</mark>"
    highlighted = re.sub(pattern, _wrap, safe, flags=re.IGNORECASE)
    return highlighted[:max_len]


def parse_query(q: str):
    parts = shlex.split(q)
    phrases, terms, nears = [], [], []
    i = 0
    while i < len(parts):
        p = parts[i]
        if p.upper().startswith("NEAR/"):
            k = int(p.split("/", 1)[1])
            a = parts[i-1] if i-1 >= 0 else ""
            b = parts[i+1] if i+1 < len(parts) else ""
            if a and b:
                nears.append((a, b, k))
                i += 2
        elif (p.startswith('"') and p.endswith('"')):
            phrases.append(p.strip('"'))
        elif (p.startswith("'") and p.endswith("'")):
            phrases.append(p.strip("'"))
        else:
            terms.append(p)
        i += 1
    return {"phrases": phrases, "terms": terms, "nears": nears}


def positions_match_phrase(pos_lists, phrase_len):
    if not pos_lists:
        return False
    s0 = set(pos_lists[0])
    for shift, pl in enumerate(pos_lists[1:], start=1):
        sset = set(pl)
        s0 = {p for p in s0 if (p + shift) in sset}
        if not s0:
            return False
    return True


def positions_match_near(pA, pB, k):
    i = 0
    j = 0
    while i < len(pA) and j < len(pB):
        if abs(pA[i] - pB[j]) <= k:
            return True
        if pA[i] < pB[j]:
            i += 1
        else:
            j += 1
    return False


def tokenize_with_spans(text: str):
    """
    Returns (tokens, spans) where tokens[i] is the normalized token and
    spans[i] = (start_char, end_char) in the original text.
    """
    toks, spans = [], []
    for m in _token_re.finditer(text):
        raw = m.group(0)
        t = raw.lower()
        if t in STOP or len(t) <= 1:
            continue
        toks.append(t)
        spans.append((m.start(), m.end()))
    return toks, spans


def best_passage_span(text: str, query_terms: list[str], win_tokens: int = 60) -> Tuple[int, int]:
    """
    Slide a token window (size win_tokens) and choose the span that covers
    the most query terms with best density. Returns (char_start, char_end).
    Fallback to first win_tokens if no hit.
    """
    if not text or not query_terms:
        return 0, min(len(text), 240)

    qset = set(query_terms)
    toks, spans = tokenize_with_spans(text)
    if not toks:
        return 0, min(len(text), 240)

    is_q = [t in qset for t in toks]

    best_score, best_i = -1.0, 0
    n = len(toks)
    W = max(10, win_tokens)

    left = 0
    counts = {}
    unique_hits = 0

    for right in range(n):
        t = toks[right]
        if t in qset:
            counts[t] = counts.get(t, 0) + 1
            if counts[t] == 1:
                unique_hits += 1
        while right - left + 1 > W:
            lt = toks[left]
            if lt in qset:
                counts[lt] -= 1
                if counts[lt] == 0:
                    unique_hits -= 1
            left += 1
        width = max(1, spans[right][1] - spans[left][0])
        density = (sum(1 for i in range(left, right+1)
                   if is_q[i])) / (right - left + 1)
        score = unique_hits + 0.2 * density
        if score > best_score:
            best_score, best_i = score, left
            best_j = right

    char_start, char_end = spans[best_i][0], spans[best_j][1]
    return char_start, char_end


def build_snippet_from_span(text: str, start: int, end: int, pad: int = 80) -> str:
    s = max(0, start - pad)
    e = min(len(text), end + pad)
    snippet = text[s:e].strip().replace("\n", " ")
    return html.escape(snippet)


def highlight_html_snippet(snippet_html: str, terms: list[str]) -> str:
    if not terms:
        return snippet_html
    pattern = r"(" + "|".join(re.escape(t) for t in set(terms) if t) + r")"
    return re.sub(pattern, lambda m: f"<mark>{m.group(0)}</mark>", snippet_html, flags=re.IGNORECASE)
