"use client";
import { useState } from "react";

export default function SearchBar({ initial, onSearch }: { initial?: string; onSearch: (q: string)=>void }) {
  const [q, setQ] = useState(initial ?? "");
  return (
    <form
      className="flex gap-2"
      onSubmit={(e) => { e.preventDefault(); onSearch(q.trim()); }}
    >
      <input
        className="w-full border border-gray-200 rounded-full px-5 py-3 outline-none focus:ring focus:ring-blue-100 shadow-sm"
        placeholder="Search documents…"
        value={q}
        onChange={(e)=>setQ(e.target.value)}
      />
      <button className="px-5 py-3 rounded-full border border-gray-200 hover:bg-gray-50 shadow-sm" type="submit">Search</button>
    </form>
  );
}
