import { getDoc } from "@/lib/api";

type Params = { params: { id: string } };

export default async function DocPage({ params }: Params) {
  const id = Number(params.id);
  const data = await getDoc(id);
  return (
    <main className="max-w-3xl mx-auto p-6 space-y-4">
      <a href="/" className="text-sm text-blue-600 hover:underline">← Back</a>
      <h1 className="text-2xl font-semibold">{data.title || data.path}</h1>
      <div className="text-xs text-gray-500">{data.path}</div>
      <article className="whitespace-pre-wrap leading-relaxed">{data.full_text}</article>
    </main>
  );
}
