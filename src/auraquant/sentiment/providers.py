from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Protocol

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


@dataclass
class CoinGeckoTrendingProvider:
    """Fallback news provider using CoinGecko Trending API.
    
    Fetches trending coins from CoinGecko when CryptoPanic quota is exhausted.
    Generates synthetic news items from trending data for sentiment analysis.
    """

    def fetch_latest(self, symbol: str, limit: int = 5) -> List[NewsItem]:
        try:
            from auraquant.data.coingecko_client import CoinGeckoClient, WEEX_BASE_TO_COINGECKO_ID
        except Exception as e:
            raise RuntimeError("CoinGecko client unavailable") from e

        internal = symbol if "/" in symbol else f"{symbol}/USDT"
        base = internal.split("/")[0].upper().strip()
        cid = WEEX_BASE_TO_COINGECKO_ID.get(base)
        if not cid:
            return []

        ttl_seconds = float(os.getenv("COINGECKO_TRENDING_TTL_SECONDS", "600"))

        client = CoinGeckoClient()
        
        try:
            trending_data = client.get_trending(ttl_seconds=ttl_seconds)
        except Exception:
            # If trending fails, return empty and let momentum provider handle it
            return []

        items: List[NewsItem] = []
        now = datetime.utcnow()

        # Check if our symbol is in trending coins
        trending_coins = trending_data.get("coins", [])
        symbol_trending = False
        symbol_rank = None
        symbol_price_change = None
        
        for idx, coin_wrapper in enumerate(trending_coins):
            coin = coin_wrapper.get("item", {})
            coin_symbol = str(coin.get("symbol", "")).upper()
            if coin_symbol == base:
                symbol_trending = True
                symbol_rank = idx + 1
                # Extract 24h price change if available
                price_data = coin.get("data", {})
                pct_24h = price_data.get("price_change_percentage_24h", {})
                if isinstance(pct_24h, dict):
                    symbol_price_change = pct_24h.get("usd")
                break

        if symbol_trending:
            bias = "bullish"  # Trending = positive attention
            title = f"{internal} coingecko TRENDING rank #{symbol_rank} (high interest)"
            if symbol_price_change is not None:
                title += f" chg={symbol_price_change:+.2f}%"
            items.append(NewsItem(
                title=title,
                published_at=now,
                source="coingecko-trending",
                url=None
            ))

        # Check trending categories for relevant signals
        trending_categories = trending_data.get("categories", [])
        for cat in trending_categories[:3]:  # Top 3 categories
            cat_name = cat.get("name", "")
            cat_change = cat.get("data", {}).get("market_cap_change_percentage_24h", {})
            usd_change = cat_change.get("usd") if isinstance(cat_change, dict) else None
            
            # Check if category is relevant to our symbol
            cat_lower = cat_name.lower()
            base_lower = base.lower()
            
            relevance = False
            if base_lower in cat_lower:
                relevance = True
            elif base == "SOL" and "solana" in cat_lower:
                relevance = True
            elif base == "ETH" and "ethereum" in cat_lower:
                relevance = True
            elif base == "BTC" and "bitcoin" in cat_lower:
                relevance = True
            elif "meme" in cat_lower and base in ["DOGE", "SHIB"]:
                relevance = True

            if relevance:
                bias = "bullish" if (usd_change and usd_change > 0) else "bearish"
                title = f"{internal} category '{cat_name}' trending ({bias})"
                if usd_change:
                    title += f" mktcap_chg={usd_change:+.2f}%"
                items.append(NewsItem(
                    title=title,
                    published_at=now,
                    source="coingecko-trending",
                    url=None
                ))
        
        # FALLBACK: Use coin-specific momentum data from CoinGecko markets API
        if not items:
            try:
                from auraquant.data.coingecko_client import CoinGeckoClient, WEEX_BASE_TO_COINGECKO_ID
                client = CoinGeckoClient()
                markets = client.get_markets(vs_currency="usd", ids=[cid], ttl_seconds=ttl_seconds)
                if markets:
                    m = markets[0]
                    chg_pct = float(m.price_change_percentage_24h or 0.0)
                    market_bias = "bullish" if chg_pct > 0.3 else ("bearish" if chg_pct < -0.3 else "neutral")
                    title = f"{internal} coingecko 24h change {market_bias} (chg={chg_pct:+.2f}%)"
                    items.append(NewsItem(
                        title=title,
                        published_at=now,
                        source="coingecko-market",
                        url=None
                    ))
            except Exception:
                pass  

        return items[:limit]


@dataclass
class CoinGeckoMomentumProvider:
    

    def fetch_latest(self, symbol: str, limit: int = 5) -> List[NewsItem]:
        try:
            from auraquant.data.coingecko_client import CoinGeckoClient, WEEX_BASE_TO_COINGECKO_ID
        except Exception as e:
            raise RuntimeError("CoinGecko client unavailable") from e

        internal = symbol if "/" in symbol else f"{symbol}/USDT"
        base = internal.split("/")[0].upper().strip()
        cid = WEEX_BASE_TO_COINGECKO_ID.get(base)
        if not cid:
            return []

        ttl_seconds = float(os.getenv("COINGECKO_SENTIMENT_TTL_SECONDS", "300"))
        threshold_pct = float(os.getenv("COINGECKO_SENTIMENT_THRESHOLD_PCT", "0.3"))

        client = CoinGeckoClient()
        markets = client.get_markets(vs_currency="usd", ids=[cid], ttl_seconds=ttl_seconds)
        if not markets:
            return []

        m = markets[0]
        chg_pct = float(m.price_change_percentage_24h or 0.0)

        bias = "flat"
        if chg_pct > threshold_pct:
            bias = "bullish"
        elif chg_pct < -threshold_pct:
            bias = "bearish"

        now = datetime.utcnow()
        titles = [
            f"{internal} coingecko 24h momentum {bias} (chg={chg_pct:+.2f}%)",
            f"{internal} coingecko market snapshot (id={m.id})",
        ]
        items = [
            NewsItem(title=t, published_at=now, source="coingecko", url=None)
            for t in titles
        ]
        return items[:limit]


@dataclass
class ChainNewsProvider:
    """Chains multiple news providers with automatic fallback.
    
    Tries providers in order until one returns non-empty results.
    Use case: CryptoPanic (limited quota) -> CoinGecko Trending -> Momentum fallback
    """

    providers: List[NewsProvider]

    def fetch_latest(self, symbol: str, limit: int = 5) -> List[NewsItem]:
        for provider in self.providers:
            try:
                items = provider.fetch_latest(symbol=symbol, limit=limit)
                if items:  # Got results, stop chain
                    return items
            except Exception:
                # Provider failed, continue to next
                continue
        return []  # All providers exhausted


def create_default_news_provider() -> NewsProvider:
    """Factory function to create the default news provider chain.
    
    Chain order:
    1. CryptoPanic (real news, but 7 calls/day limit)
    2. CoinGecko Trending (trending coins/categories sentiment)
    3. CoinGecko Momentum (24h price momentum as synthetic news)
    """
    providers: List[NewsProvider] = []
    
    # Primary: CryptoPanic (if API token available)
    try:
        cp = CryptoPanicProvider()
        if cp.api_token:
            providers.append(cp)
    except Exception:
        pass
    
    # Secondary: CoinGecko Trending (requires API key for reliable access)
    cg_api_key = os.getenv("COINGECKO_API_KEY")
    if cg_api_key:
        providers.append(CoinGeckoTrendingProvider())
    
    # Tertiary: CoinGecko Momentum (always available as fallback)
    providers.append(CoinGeckoMomentumProvider())
    
    if len(providers) == 1:
        return providers[0]
    
    return ChainNewsProvider(providers=providers)
