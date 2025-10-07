"use client";
export default function Pagination({
  page, perPage, total, onPage
}: { page: number; perPage: number; total: number; onPage: (p:number)=>void }) {
  const pages = Math.max(1, Math.ceil(total / perPage));
  return (
    <div className="flex items-center justify-center gap-2 mt-6">
      <button disabled={page<=1} onClick={()=>onPage(page-1)} className="px-3 py-1 border rounded disabled:opacity-50">Prev</button>
      <span className="text-sm">Page {page} / {pages}</span>
      <button disabled={page>=pages} onClick={()=>onPage(page+1)} className="px-3 py-1 border rounded disabled:opacity-50">Next</button>
    </div>
  );
}
