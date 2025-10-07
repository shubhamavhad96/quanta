import time
from collections import OrderedDict
from typing import Any, Optional


class LRUCacheTTL:
    def __init__(self, capacity: int = 256, ttl_sec: int = 300):
        self.capacity = capacity
        self.ttl = ttl_sec
        self.data: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        if key in self.data:
            ts, val = self.data[key]
            if now - ts <= self.ttl:
                self.data.move_to_end(key)
                return val
            del self.data[key]
        return None

    def set(self, key: str, value: Any):
        now = time.time()
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = (now, value)
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)


def normalize_query(q: str) -> str:
    return " ".join(q.lower().split())


class PostingsCache(LRUCacheTTL):
    pass
