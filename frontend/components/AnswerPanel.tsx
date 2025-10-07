export default function AnswerPanel({ answer, sources }:{ answer:string; sources:{num:number; title:string; path:string}[] }) {
  return (
    <div className="rounded-2xl border p-4 space-y-3 bg-white">
      <div className="text-sm leading-relaxed whitespace-pre-wrap">{answer}</div>
      {sources?.length>0 && (
        <div className="text-xs text-gray-600 flex flex-wrap gap-2">
          {sources.map(s => (
            <a key={s.num} href={`/doc/${s.path.split("/").pop() ?? s.num}`} className="px-2 py-1 rounded-full border hover:bg-gray-50" title={s.path}>
              [{s.num}] {s.title || s.path}
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
