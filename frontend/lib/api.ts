export type SearchResult = {
  doc_id: number;
  path: string;
  title: string;
  score: number;
  snippet: string; // may include <mark>
};
export type FacetItem = { value: string; count: number };
export type Facets = { domain: FacetItem[]; lang: FacetItem[]; type: FacetItem[] };
export type SearchResponse = {
  query: string;
  total_docs: number;
  total_hits: number;
  page: number;
  per_page: number;
  results: SearchResult[];
  facets: Facets;
  filters?: { site?: string | null; lang?: string | null; type?: string | null };
};

export type WebLink = { url: string; title: string; snippet: string };
export type WebSearchResponse = { query: string; ai?: string | null; web_results: WebLink[] };

const BASE = process.env.NEXT_PUBLIC_API_URL!;

export async function searchApi(q: string, page = 1, perPage = 10, filters?: {site?: string, lang?: string, type?: string}): Promise<SearchResponse> {
  const url = new URL("/search", BASE);
  url.searchParams.set("q", q);
  url.searchParams.set("page", String(page));
  url.searchParams.set("per_page", String(perPage));
  if (filters?.site) url.searchParams.set("site", filters.site);
  if (filters?.lang) url.searchParams.set("lang", filters.lang);
  if (filters?.type) url.searchParams.set("type", filters.type);
  const r = await fetch(url.toString(), { cache: "no-store" });
  if (!r.ok) throw new Error(`Search failed: ${r.status}`);
  return r.json();
}

export async function getDoc(docId: number) {
  const r = await fetch(`${BASE}/doc/${docId}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`Doc fetch failed: ${r.status}`);
  return r.json();
}

export async function getStats() {
  const r = await fetch(`${BASE}/stats`, { cache: "no-store" });
  if (!r.ok) throw new Error(`Stats failed: ${r.status}`);
  return r.json();
}

export async function getAnswer(q: string) {
  const r = await fetch(`${BASE}/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ q }),
    cache: "no-store"
  });
  if (!r.ok) throw new Error(`Answer failed: ${r.status}`);
  return r.json() as Promise<{ answer: string; sources: { num: number; doc_id: number; title: string; path: string }[] }>;
}

export async function webSearch(q: string): Promise<WebSearchResponse> {
  const r = await fetch(`${BASE}/search_web?q=${encodeURIComponent(q)}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`web search failed: ${r.status}`);
  return r.json();
}
