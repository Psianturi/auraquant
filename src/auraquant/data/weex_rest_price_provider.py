from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Tuple

import os
import requests

from .multi_price_provider import MultiPriceProvider


def _to_weex_contract_symbol(symbol: str) -> str:
    """Map internal symbol like 'BTC/USDT' -> WEEX contract symbol like 'cmt_btcusdt'."""

    s = symbol.replace("/", "").lower()
    return f"cmt_{s}"


@dataclass
class WeexRestMultiPriceProvider(MultiPriceProvider):
    """Public WEEX REST market data provider.

    Uses the Participant Guide public endpoint:
    - GET /capi/v2/market/ticker?symbol=cmt_btcusdt

    Notes:
    - No API keys required for public market data.
    - This provider maintains an in-memory history so CorrelationTrigger can work.
    - ATR is approximated from recent absolute price changes (MVP). You can later
      replace it with true ATR from kline/candles if desired.
    """

    base_url: str = "https://api-contract.weex.com"
    history_window: int = 300
    atr_lookback: int = 14

    def __post_init__(self) -> None:
        # Allow override via env for testing.
        self.base_url = os.getenv("WEEX_BASE_URL", self.base_url).rstrip("/")
        self._history: Dict[str, Deque[float]] = {}
        self._session = requests.Session()

    def get_tick(self, symbols: List[str], now: datetime) -> Dict[str, Tuple[float, float]]:
        _ = now
        out: Dict[str, Tuple[float, float]] = {}

        for sym in symbols:
            weex_sym = _to_weex_contract_symbol(sym)
            url = f"{self.base_url}/capi/v2/market/ticker"
            resp = self._session.get(url, params={"symbol": weex_sym}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            #
            last = float(data.get("last") or 0.0)

            h = self._history.setdefault(sym, deque(maxlen=int(self.history_window)))
            h.append(last)

            atr = self._atr_proxy(sym)
            out[sym] = (float(last), float(atr))

        return out

    def get_recent_prices(self, symbol: str, window: int) -> List[float]:
        h = self._history.get(symbol)
        if not h:
            return []
        w = max(int(window), 0)
        if w <= 0:
            return []
        return list(h)[-w:]

    def _atr_proxy(self, symbol: str) -> float:
        """Very small MVP proxy: average absolute diff of last N prices."""

        h = self._history.get(symbol)
        if not h or len(h) < 2:
            return 0.0
        n = min(int(self.atr_lookback), len(h) - 1)
        if n <= 0:
            return 0.0
        prices = list(h)[-(n + 1) :]
        diffs = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        return float(sum(diffs) / len(diffs))
