from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
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
        if self.session is None:
            self.session = _requests.Session()

    def fetch_latest(self, symbol: str, limit: int = 5) -> List[NewsItem]:
        if not self.api_token:
            raise RuntimeError("CRYPTOPANIC_API_TOKEN is missing. Use StaticNewsProvider for local demo.")

        # Import lazily so StaticNewsProvider does not require requests.
        import requests

        # CryptoPanic API docs vary; this is a pragmatic baseline.
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": self.api_token,
            "currencies": symbol.split("/")[0],
            "public": "true",
        }

        assert self.session is not None
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()

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
