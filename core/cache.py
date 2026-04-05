from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import diskcache

from config import cfg


class QueryCache:
    def __init__(self, cache_dir: Path = cfg.cache_dir, ttl: int = cfg.cache_ttl):
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(cache_dir))
        self._ttl = ttl

    def _make_key(self, query: str, context: str = "") -> str:
        raw = json.dumps({"q": query.strip().lower(), "ctx": context}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query: str, context: str = "") -> Optional[str]:
        return self._cache.get(self._make_key(query, context))

    def set(self, query: str, answer: str, context: str = "") -> None:
        self._cache.set(self._make_key(query, context), answer, expire=self._ttl)

    def clear(self) -> None:
        self._cache.clear()

    def stats(self) -> dict:
        return {"size": len(self._cache), "volume_bytes": self._cache.volume()}
