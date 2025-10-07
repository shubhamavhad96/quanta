import os
import tempfile
import shutil
from app.indexer import InvertedIndex
from app.ranker import BM25Ranker


def write(p, text):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


def test_index_and_bm25_search():
    tmpd = tempfile.mkdtemp()
    try:
        db = os.path.join(tmpd, "q.sqlite")
        data = os.path.join(tmpd, "raw")
        os.makedirs(data, exist_ok=True)

        write(os.path.join(data, "a.txt"), "alpha beta beta gamma")
        write(os.path.join(data, "b.txt"), "alpha alpha beta")
        write(os.path.join(data, "c.txt"),
              "<html><head><title>Gamma</title></head><body>gamma gamma beta</body></html>")

        idx = InvertedIndex(db)
        idx.build_from_folder(data)

        ranker = BM25Ranker(idx.conn, k1=1.4, b=0.75)
        res = ranker.search("alpha beta", k=3)

        assert len(res) >= 1
        top_doc_id, score = res[0]
        assert score > 0
    finally:
        shutil.rmtree(tmpd)
