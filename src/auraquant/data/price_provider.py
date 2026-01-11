from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Protocol, Tuple


class PriceProvider(Protocol):
    def get_last_price(self, symbol: str, now: datetime) -> float:
        raise NotImplementedError

    def get_atr(self, symbol: str, now: datetime) -> float:
        raise NotImplementedError


@dataclass
class StaticPriceProvider:
    """Deterministic price feed for demos.

    Provide a list of (price, atr) values. Each call to get_last_price advances the index.
    """

    series: List[Tuple[float, float]]
    idx: int = 0

    def get_last_price(self, symbol: str, now: datetime) -> float:
        _ = symbol
        _ = now
        if not self.series:
            return 0.0
        price, _atr = self.series[self.idx % len(self.series)]
        self.idx += 1
        return float(price)

    def get_atr(self, symbol: str, now: datetime) -> float:
        _ = symbol
        _ = now
        if not self.series:
            return 0.0

        pos = (self.idx - 1) % len(self.series)
        _price, atr = self.series[pos]
        return float(atr)
