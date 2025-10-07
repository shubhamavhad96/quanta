export default function WebResults({ items }:{ items:{url:string; title:string; snippet:string}[] }) {
  if (!items?.length) return null;
  return (
    <div className="space-y-3">
      {items.map(w => (
        <a key={w.url} href={w.url} target="_blank" rel="noreferrer" className="block rounded-2xl border p-4 hover:shadow-sm">
          <div className="text-blue-700 font-medium">{w.title}</div>
          <div className="text-xs text-gray-600 mt-1 truncate">{w.url}</div>
          <div className="text-sm text-gray-800 mt-2 line-clamp-3">{w.snippet}</div>
        </a>
      ))}
    </div>
  );
}
