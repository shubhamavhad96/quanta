import Link from "next/link";
import type { SearchResult } from "@/lib/api";

export default function ResultCard({ r }: { r: SearchResult }) {
  return (
    <div className="rounded-2xl border border-gray-200 p-4 hover:shadow-sm transition bg-white">
      <div className="flex items-baseline justify-between">
        <Link href={`/doc/${r.doc_id}`} className="text-lg font-semibold hover:underline text-blue-700">
          {r.title || r.path.split("/").pop()}
        </Link>
        <span className="text-xs text-gray-500">score: {r.score.toFixed(3)}</span>
      </div>
      <div className="mt-2 text-sm text-gray-800 marked" dangerouslySetInnerHTML={{ __html: r.snippet }} />
      <div className="mt-2 text-xs text-gray-500 truncate">{r.path}</div>
    </div>
  );
}
