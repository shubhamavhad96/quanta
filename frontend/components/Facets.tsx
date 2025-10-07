"use client";

type Item = { value: string; count: number };

export default function Facets({
  title, items, onPick, active
}: { title: string; items: Item[]; onPick: (v: string)=>void; active?: string|null }) {
  return (
    <div className="space-y-2">
      <div className="text-xs font-semibold text-gray-600">{title}</div>
      <div className="flex flex-wrap gap-2">
        {items.map(it => (
          <button
            key={title+it.value}
            onClick={()=>onPick(it.value)}
            className={`px-2 py-1 rounded-full border text-xs ${active===it.value ? "bg-black text-white" : "hover:bg-gray-50"}`}
            title={`${it.count} results`}
          >
            {it.value} <span className="opacity-60">({it.count})</span>
          </button>
        ))}
      </div>
    </div>
  );
}
