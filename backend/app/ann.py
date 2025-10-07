import os
import numpy as np
import hnswlib
from typing import List, Tuple, Optional


class ANNIndex:
    """
    HNSW ANN wrapper for Quanta.
    Stores normalized float32 vectors (dim = 384 for MiniLM).
    Persists to disk with .bin + sidecar mapping file.
    """

    def __init__(self, dim: int = 384, space: str = "cosine"):
        self.dim = dim
        self.space = space
        self.index: Optional[hnswlib.Index] = None
        self._ids: Optional[np.ndarray] = None
        self._path_index = None
        self._path_ids = None

    def create(self, num_elements: int, M: int = 32, efC: int = 200):
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=num_elements,
                              ef_construction=efC, M=M)
        self.index.set_ef(64)

    def add(self, vectors: np.ndarray, labels: np.ndarray):
        assert self.index is not None
        self.index.add_items(vectors, labels)

    def save(self, path_index: str, path_ids: str, ids: np.ndarray):
        assert self.index is not None
        self.index.save_index(path_index)
        np.save(path_ids, ids)
        self._path_index = path_index
        self._path_ids = path_ids
        self._ids = ids

    def load(self, path_index: str, path_ids: str):
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.load_index(path_index)
        self.index.set_ef(64)
        self._ids = np.load(path_ids)
        self._path_index = path_index
        self._path_ids = path_ids

    def is_ready(self) -> bool:
        return self.index is not None and self._ids is not None

    def query(self, vector: np.ndarray, k: int = 50) -> List[Tuple[int, float]]:
        """
        Returns list of (doc_id, sim) with cosine similarity (higher is better).
        """
        assert self.is_ready()
        if vector.ndim == 1:
            vector = vector[None, :]
        # clamp k to number of indexed items
        try:
            # type: ignore[attr-defined]
            current = self.index.get_current_count()
        except Exception:
            current = len(self._ids) if self._ids is not None else k
        k = max(1, min(k, current))
        labels, dists = self.index.knn_query(vector, k=k)
        sims = 1.0 - dists[0]
        out = []
        for lbl, sim in zip(labels[0], sims):
            if lbl < 0:
                continue
            doc_id = int(self._ids[lbl])
            out.append((doc_id, float(sim)))
        return out
