from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
from typing import List, Optional, Protocol

from .types import NewsItem


class NewsProvider(Protocol):

    def fetch_latest(self, symbol: str, limit: int = 5) -> List[NewsItem]:
        raise NotImplementedError


@dataclass
class StaticNewsProvider:
    """Deterministic provider for demos/tests (no network, no API keys)."""

    items: List[NewsItem]

    def fetch_latest(self, symbol: str, limit: int = 5) -> List[NewsItem]:
        _ = symbol
        return list(self.items)[:limit]


@dataclass
class CryptoPanicProvider:

    api_token: Optional[str] = None
    base_url: Optional[str] = None
    session: Optional[object] = None

    def __post_init__(self) -> None:
        try:
            import requests as _requests
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Optional dependency missing: 'requests'. Install requirements.txt or use StaticNewsProvider."
            ) from e

        if self.api_token is None:
            self.api_token = os.getenv("CRYPTOPANIC_API_TOKEN")
        if self.base_url is None:
            # Per CryptoPanic docs (and WEEX participant setups), the base endpoint is typically:
            # https://cryptopanic.com/api/developer/v2
            self.base_url = os.getenv("CRYPTOPANIC_BASE_URL", "https://cryptopanic.com/api/developer/v2")
        if self.session is None:
            self.session = _requests.Session()

    def fetch_latest(self, symbol: str, limit: int = 5) -> List[NewsItem]:
        if not self.api_token:
            raise RuntimeError("CRYPTOPANIC_API_TOKEN is missing. Use StaticNewsProvider for local demo.")

        # Optional daily quota guard (off by default).
        # Example: CRYPTOPANIC_MAX_REQ_PER_DAY=7
        max_per_day_s = os.getenv("CRYPTOPANIC_MAX_REQ_PER_DAY")
        if max_per_day_s:
            try:
                max_per_day = int(max_per_day_s)
            except Exception:
                max_per_day = 0

            if max_per_day > 0:
                usage_path = Path(os.getenv("CRYPTOPANIC_USAGE_PATH", "ai_logs/cryptopanic_usage.json"))
                usage_path.parent.mkdir(parents=True, exist_ok=True)

                today = datetime.utcnow().strftime("%Y-%m-%d")
                try:
                    data = json.loads(usage_path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}

                used = int(data.get(today, 0) or 0)
                if used >= max_per_day:
                    # Do not call the API; return empty news so sentiment can fall back.
                    return []

                data[today] = used + 1
                usage_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        base = (self.base_url or "https://cryptopanic.com/api/developer/v2").rstrip("/")
        url = f"{base}/posts/"
        params = {
            "auth_token": self.api_token,
            "currencies": symbol.split("/")[0],
            "public": "true",
        }

        assert self.session is not None
        resp = self.session.get(url, params=params, timeout=15)
        if resp.status_code == 404:
            alt_base = None
            if "/api/v1" in base:
                alt_base = "https://cryptopanic.com/api/developer/v2"
            elif "/api/developer/v2" in base:
                alt_base = "https://cryptopanic.com/api/v1"

            if alt_base:
                alt_url = f"{alt_base.rstrip('/')}/posts/"
                resp = self.session.get(alt_url, params=params, timeout=15)

        if resp.status_code != 200:
            # Avoid leaking auth token in exceptions.
            raise RuntimeError(f"CryptoPanic HTTP {resp.status_code} while fetching posts")

        try:
            payload = resp.json()
        except Exception as e:
            raise RuntimeError("CryptoPanic returned non-JSON response") from e

        results = payload.get("results") or []
        items: List[NewsItem] = []
        for r in results[:limit]:
            title = r.get("title") or ""
            published = r.get("published_at") or r.get("created_at")
            source = (r.get("source") or {}).get("title") if isinstance(r.get("source"), dict) else (r.get("source") or "cryptopanic")
            link = r.get("url") or r.get("link")

            if not title or not published:
                continue
            try:
                published_at = datetime.fromisoformat(published.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                continue

            items.append(NewsItem(title=title, published_at=published_at, source=str(source), url=link))

        return items
