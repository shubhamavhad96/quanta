"use client";
import { useState } from "react";
import { webSearch, type WebSearchResponse } from "@/lib/api";
import WebResults from "@/components/WebResults";
import AiOverview from "@/components/AiOverview";

export default function Home() {
  const [q, setQ] = useState("");
  const [data, setData] = useState<WebSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    if (!q.trim()) return;
    setLoading(true); setErr(null);
    try {
      const res = await webSearch(q.trim());
      setData(res);
    } catch (e:any) {
      setErr(e.message || "search failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="max-w-3xl mx-auto p-6 space-y-6">
      <h1 className="text-2xl font-bold">Quanta</h1>
      <form className="flex gap-2" onSubmit={(e)=>{e.preventDefault(); run();}}>
        <input className="w-full border rounded-xl px-4 py-2" placeholder="Search the web…" value={q} onChange={e=>setQ(e.target.value)} />
        <button className="px-4 py-2 rounded-xl border" type="submit">Search</button>
      </form>

      {loading && <div className="text-sm text-gray-500">Searching…</div>}
      {err && <div className="text-sm text-red-600">{err}</div>}

      {data && (
        <>
          <AiOverview text={data.ai || undefined} sources={(data.web_results || []).map(w=>({url:w.url, title:w.title}))} />
          <div className="mt-4">
            <WebResults items={data.web_results} />
          </div>
        </>
      )}
    </main>
  );
}
