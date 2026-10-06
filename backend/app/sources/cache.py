import hashlib
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class JSONResponseCache:
    """Simple TTL file cache for external API JSON payloads."""

    def __init__(self, cache_dir: Path, ttl_seconds: int = 86_400) -> None:
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str) -> dict | None:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)
            return None

        cached_at = float(payload.get("_cached_at", 0))
        if time.time() - cached_at > self.ttl_seconds:
            path.unlink(missing_ok=True)
            return None
        data = payload.get("data")
        return data if isinstance(data, dict) else None

    def set(self, key: str, data: dict) -> None:
        path = self._path_for_key(key)
        envelope = {"_cached_at": time.time(), "data": data}
        path.write_text(json.dumps(envelope), encoding="utf-8")
        logger.debug("Cached external API response key=%s", key[:80])
