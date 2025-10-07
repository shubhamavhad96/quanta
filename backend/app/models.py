from pydantic import BaseModel
from typing import List, Optional


class SearchResult(BaseModel):
    doc_id: int
    path: str
    title: str
    score: float
    snippet: str  # may contain <mark> tags (HTML)


class FacetItem(BaseModel):
    value: str
    count: int


class Facets(BaseModel):
    domain: List[FacetItem]
    lang: List[FacetItem]
    type: List[FacetItem]


class WebLink(BaseModel):
    url: str
    title: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    total_docs: int
    total_hits: int
    page: int
    per_page: int
    results: List[SearchResult]
    facets: Facets
    filters: Optional[dict] = None
    web_results: List[WebLink] = []
    ai: Optional[str] = None
