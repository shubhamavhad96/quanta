export default function AiOverview({ text, sources }:{ text?: string|null; sources: { url:string; title:string }[] }) {
  if (!text || text.trim().length === 0) return null;
  const cites = (sources || []).slice(0, 5);
  return (
    <section className="rounded-2xl border p-5 bg-white shadow-sm space-y-3">
      <div className="text-xs font-semibold uppercase tracking-wide text-green-700">AI Overview</div>
      <div className="text-sm leading-relaxed whitespace-pre-wrap">{text}</div>
      {cites.length > 0 && (
        <div className="flex flex-wrap gap-2 pt-1">
          {cites.map((s, i) => (
            <a key={s.url} href={s.url} target="_blank" rel="noreferrer" className="px-2 py-1 text-xs rounded-full border hover:bg-gray-50">
              [{i+1}] {s.title || s.url}
            </a>
          ))}
        </div>
      )}
    </section>
  );
}
