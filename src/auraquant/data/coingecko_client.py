from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


PUBLIC_API_BASE_URL = "https://api.coingecko.com/api/v3"
PRO_API_BASE_URL = "https://pro-api.coingecko.com/api/v3"


@dataclass(frozen=True)
class CoinGeckoMarket:
    """Small, auditable subset of CoinGecko /coins/markets fields."""

    id: str
    symbol: str
    name: str
    current_price: float
    market_cap: Optional[float]
    total_volume: Optional[float]
    price_change_percentage_24h: Optional[float]
    price_change_percentage_1h: Optional[float]
    high_24h: Optional[float]
    low_24h: Optional[float]
    ath: Optional[float]
    ath_change_percentage: Optional[float]
    last_updated: Optional[str]

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "CoinGeckoMarket":
        return cls(
            id=str(obj.get("id") or ""),
            symbol=str(obj.get("symbol") or ""),
            name=str(obj.get("name") or ""),
            current_price=float(obj.get("current_price") or 0.0),
            market_cap=float(obj["market_cap"]) if obj.get("market_cap") is not None else None,
            total_volume=float(obj["total_volume"]) if obj.get("total_volume") is not None else None,
            price_change_percentage_24h=float(obj["price_change_percentage_24h"]) if obj.get("price_change_percentage_24h") is not None else None,
            price_change_percentage_1h=float(obj["price_change_percentage_1h_in_currency"]) if obj.get("price_change_percentage_1h_in_currency") is not None else None,
            high_24h=float(obj["high_24h"]) if obj.get("high_24h") is not None else None,
            low_24h=float(obj["low_24h"]) if obj.get("low_24h") is not None else None,
            ath=float(obj["ath"]) if obj.get("ath") is not None else None,
            ath_change_percentage=float(obj["ath_change_percentage"]) if obj.get("ath_change_percentage") is not None else None,
            last_updated=str(obj.get("last_updated")) if obj.get("last_updated") is not None else None,
        )


class _DiskCache:
    def __init__(self, cache_dir: str | Path = "runtime_cache/coingecko"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in key)
        return self.cache_dir / f"{safe}.json"

    def get_json(self, key: str, ttl_seconds: float) -> Optional[Any]:
        p = self._path_for_key(key)
        if not p.exists():
            return None
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                return None
            saved_at = float(obj.get("saved_at", 0.0))
            if (time.time() - saved_at) > float(ttl_seconds):
                return None
            return obj.get("data")
        except Exception:
            return None

    def set_json(self, key: str, data: Any) -> None:
        p = self._path_for_key(key)
        payload = {"saved_at": time.time(), "data": data}
        try:
            p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception:
            return


@dataclass
class CoinGeckoClient:
    """CoinGecko REST client (minimal) with safe caching.

    This is intended for:
    - market discovery / universe selection
    - research & analytics snapshots

    NOT intended to drive execution tick-by-tick (use WEEX prices for execution).
    """

    base_url: str = PUBLIC_API_BASE_URL
    api_key: Optional[str] = None
    timeout_seconds: float = 15.0
    cache: Optional[_DiskCache] = None
    session: Optional[requests.Session] = None

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("COINGECKO_API_KEY")

       
        env_base_url = os.getenv("COINGECKO_BASE_URL")
        if env_base_url:
            self.base_url = env_base_url
        else:
            # Demo keys start with "CG-" and must use public API base
            is_demo_key = bool(self.api_key and self.api_key.startswith("CG-"))
            if is_demo_key:
                self.base_url = PUBLIC_API_BASE_URL
            elif self.api_key:
                self.base_url = PRO_API_BASE_URL
            else:
                self.base_url = PUBLIC_API_BASE_URL
        self.base_url = self.base_url.rstrip("/")

        if self.cache is None:
            self.cache = _DiskCache(cache_dir=os.getenv("COINGECKO_CACHE_DIR", "runtime_cache/coingecko"))
        if self.session is None:
            self.session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        # CoinGecko keys:

        headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "AuraQuant/1.0 (CoinGecko Track)",
        }
        if self.api_key:
            if self.api_key.startswith("CG-"):
                headers["x-cg-demo-api-key"] = self.api_key
            else:
                headers["x-cg-pro-api-key"] = self.api_key
        return headers

    def get_markets(
        self,
        *,
        vs_currency: str = "usd",
        ids: Iterable[str],
        per_page: int = 250,
        ttl_seconds: float = 300.0,
    ) -> List[CoinGeckoMarket]:
        """Fetch /coins/markets for a fixed set of IDs with caching."""

        ids_list = [str(i).strip() for i in ids if str(i).strip()]
        ids_key = ",".join(sorted(ids_list))
        cache_key = f"markets_{vs_currency}_{ids_key}"

        assert self.cache is not None
        cached = self.cache.get_json(cache_key, ttl_seconds=float(ttl_seconds))
        if isinstance(cached, list):
            return [CoinGeckoMarket.from_json(x) for x in cached if isinstance(x, dict)]

        url = f"{self.base_url}/coins/markets"
        params = {
            "vs_currency": vs_currency,
            "ids": ids_key,
            "order": "market_cap_desc",
            "per_page": int(per_page),
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "1h,24h",  # Request both 1h and 24h price changes
        }

        assert self.session is not None
        try:
            resp = self.session.get(url, params=params, headers=self._headers(), timeout=float(self.timeout_seconds))
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:

            if e.response.status_code == 400 and "price_change_percentage" in str(e):
                params.pop("price_change_percentage", None)
                resp = self.session.get(url, params=params, headers=self._headers(), timeout=float(self.timeout_seconds))
                resp.raise_for_status()
            else:
                raise
        
        data = resp.json()
        if not isinstance(data, list):
            raise ValueError("Unexpected CoinGecko response")

        self.cache.set_json(cache_key, data)
        return [CoinGeckoMarket.from_json(x) for x in data if isinstance(x, dict)]

    def get_trending(self, ttl_seconds: float = 600.0) -> Dict[str, Any]:
        """Fetch /search/trending for trending coins, NFTs, categories.
        
        Returns raw JSON dict with 'coins', 'nfts', 'categories' keys.
        Cached for 10 minutes by default.
        """
        cache_key = "search_trending"
        
        assert self.cache is not None
        cached = self.cache.get_json(cache_key, ttl_seconds=float(ttl_seconds))
        if isinstance(cached, dict):
            return cached

        url = f"{self.base_url}/search/trending"
        
        assert self.session is not None
        resp = self.session.get(url, headers=self._headers(), timeout=float(self.timeout_seconds))
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError("Unexpected CoinGecko trending response")

        self.cache.set_json(cache_key, data)
        return data

    def get_global(self, ttl_seconds: float = 300.0) -> Dict[str, Any]:
        """Fetch /global for total market cap, BTC dominance, etc.
        """
        cache_key = "global_data"
        
        assert self.cache is not None
        cached = self.cache.get_json(cache_key, ttl_seconds=float(ttl_seconds))
        if isinstance(cached, dict):
            return cached

        url = f"{self.base_url}/global"
        
        assert self.session is not None
        resp = self.session.get(url, headers=self._headers(), timeout=float(self.timeout_seconds))
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError("Unexpected CoinGecko global response")

        self.cache.set_json(cache_key, data)
        return data

    def get_market_trend_filter(
        self,
        symbol: str,
        ttl_seconds: float = 180.0,
    ) -> Dict[str, Any]:
        """Analyze 24h trend to determine if entry is safe.
        
        Returns:
            {
                "allow_long": bool,
                "allow_short": bool,
                "trend": "uptrend" | "downtrend" | "sideways",
                "change_24h": float,
                "change_1h": float,
                "volatility_24h": float,  # (high-low)/price as percentage
                "btc_dominance": float,
                "market_regime": "risk_on" | "risk_off" | "neutral"
            }
        """
        # Parse symbol to get base asset
        base = symbol.replace("/USDT", "").replace("USDT", "").upper().strip()
        cid = WEEX_BASE_TO_COINGECKO_ID.get(base)
        
        result = {
            "allow_long": True,
            "allow_short": True,
            "trend": "sideways",
            "change_24h": 0.0,
            "change_1h": 0.0,
            "volatility_24h": 0.0,
            "btc_dominance": 0.0,
            "market_regime": "neutral",
        }
        
        if not cid:
            return result
        
        try:
            markets = self.get_markets(vs_currency="usd", ids=[cid], ttl_seconds=ttl_seconds)
            if not markets:
                return result
            
            m = markets[0]
            change_24h = float(m.price_change_percentage_24h or 0.0)
            change_1h = float(m.price_change_percentage_1h or 0.0)
            
            # Calculate 24h volatility
            high = float(m.high_24h or m.current_price)
            low = float(m.low_24h or m.current_price)
            price = float(m.current_price or 1.0)
            volatility = ((high - low) / price) * 100.0 if price > 0 else 0.0
            
            result["change_24h"] = change_24h
            result["change_1h"] = change_1h
            result["volatility_24h"] = volatility
            
            if change_24h > 2.0:
                result["trend"] = "uptrend"
            elif change_24h < -2.0:
                result["trend"] = "downtrend"
            else:
                result["trend"] = "sideways"
            
            block_threshold = float(os.getenv("TREND_BLOCK_THRESHOLD_PCT", "5.0")) 
            
            if change_24h < -block_threshold:
                result["allow_long"] = False
            if change_24h > block_threshold:
                result["allow_short"] = False
            
            # Allow reversal trades based on 1h momentum
            if change_24h < -block_threshold and change_1h > 1.5:
                result["allow_long"] = True  # Potential reversal
            
            if change_1h < -1.0:  # 1h drop > 1%
                result["allow_short"] = True  # Short-term momentum reversal
            
            # Get global market data for regime
            try:
                global_data = self.get_global(ttl_seconds=ttl_seconds)
                data = global_data.get("data", {})
                
                btc_dom = float(data.get("market_cap_percentage", {}).get("btc", 0.0))
                total_mcap_change = float(data.get("market_cap_change_percentage_24h_usd", 0.0))
                
                result["btc_dominance"] = btc_dom
                
                # Market regime detection
                # Risk-on: total market up, BTC dominance stable/down (altcoins gaining)
                # Risk-off: total market down, BTC dominance up (flight to BTC)
                if total_mcap_change > 2.0 and btc_dom < 55.0:
                    result["market_regime"] = "risk_on"
                elif total_mcap_change < -2.0 or btc_dom > 60.0:
                    result["market_regime"] = "risk_off"
                else:
                    result["market_regime"] = "neutral"
                
                if result["market_regime"] == "risk_off" and base not in ["BTC", "ETH"]:
                    if change_24h < 0:
                        result["allow_long"] = False
            except Exception:
                pass  
                
        except Exception:
            pass 
        
        return result


WEEX_BASE_TO_COINGECKO_ID: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "DOGE": "dogecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "BNB": "binancecoin",
    "LTC": "litecoin",
}


def _score_market(m: CoinGeckoMarket) -> float:
    """Heuristic score for universe selection.

    Goal: pick liquid + moving assets.
    - volume dominates (log-scaled)
    - add a small bonus for absolute 24h move
    """

    vol = float(m.total_volume or 0.0)
    move = abs(float(m.price_change_percentage_24h or 0.0))

    # log1p for stability
    return float(math.log1p(max(vol, 0.0)) + 0.05 * move)


def pick_weex_contract_symbol_by_liquidity(
    *,
    client: CoinGeckoClient,
    allowed_contract_symbols: Iterable[str],
    ttl_seconds: float = 300.0,
) -> str:
    """Return one WEEX contract symbol, chosen using CoinGecko markets snapshot.

    This is *selection only*; execution still uses WEEX endpoints.
    """

    allowed = [s.lower() for s in allowed_contract_symbols]

    # Map WEEX contract symbols like cmt_btcusdt -> BTC
    base_assets: List[str] = []
    for sym in allowed:
        # expected format: cmt_<base>usdt
        if not sym.startswith("cmt_") or not sym.endswith("usdt"):
            continue
        base = sym[len("cmt_") : -len("usdt")].upper()
        if base:
            base_assets.append(base)

    ids = [WEEX_BASE_TO_COINGECKO_ID[b] for b in base_assets if b in WEEX_BASE_TO_COINGECKO_ID]
    if not ids:
        # Fallback to BTC
        return "cmt_btcusdt"

    markets = client.get_markets(vs_currency="usd", ids=ids, ttl_seconds=float(ttl_seconds))
    if not markets:
        return "cmt_btcusdt"

    # Pick best-scoring market
    best = max(markets, key=_score_market)

    # Map back to WEEX symbol
    best_base = None
    for base, cid in WEEX_BASE_TO_COINGECKO_ID.items():
        if cid == best.id:
            best_base = base
            break
    if not best_base:
        return "cmt_btcusdt"

    picked = f"cmt_{best_base.lower()}usdt"
    if picked.lower() not in allowed:
        return "cmt_btcusdt"
    return picked
