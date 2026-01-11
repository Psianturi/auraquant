from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Protocol, Tuple


class MultiPriceProvider(Protocol):
    """Provides synchronized ticks for multiple symbols + recent history."""

    def get_tick(self, symbols: List[str], now: datetime) -> Dict[str, Tuple[float, float]]:
        """Returns {symbol: (last_price, atr)} and updates internal history."""

    def get_recent_prices(self, symbol: str, window: int) -> List[float]:
        """Returns recent observed last prices (newest last)."""


@dataclass
class StaticMultiPriceProvider:
    """Deterministic, synchronized provider for demos.

    `series_by_symbol`: {"SOL/USDT": [(price, atr), ...], "BTC/USDT": [(price, atr), ...]}

    A single global index advances once per tick, keeping symbols time-aligned.
    """

    series_by_symbol: Dict[str, List[Tuple[float, float]]]
    history_window: int = 200

    _idx: int = 0
    _history: Dict[str, Deque[float]] = None

    def __post_init__(self) -> None:
        self._history = {sym: deque(maxlen=int(self.history_window)) for sym in self.series_by_symbol.keys()}

    def get_tick(self, symbols: List[str], now: datetime) -> Dict[str, Tuple[float, float]]:
        _ = now
        out: Dict[str, Tuple[float, float]] = {}
        for sym in symbols:
            series = self.series_by_symbol.get(sym)
            if not series:
                out[sym] = (0.0, 0.0)
                continue
            price, atr = series[self._idx % len(series)]
            out[sym] = (float(price), float(atr))
            self._history.setdefault(sym, deque(maxlen=int(self.history_window))).append(float(price))

        self._idx += 1
        return out

    def get_recent_prices(self, symbol: str, window: int) -> List[float]:
        h = self._history.get(symbol)
        if not h:
            return []
        w = max(int(window), 0)
        if w == 0:
            return []
        return list(h)[-w:]
