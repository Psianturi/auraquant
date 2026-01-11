from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from ..data.multi_price_provider import MultiPriceProvider
from ..sentiment.types import MarketBias
from ..util.jsonlog import log_json, utc_iso
from .types import CorrelationSignal


def _pct_returns(prices: List[float]) -> List[float]:
    if len(prices) < 2:
        return []
    rets: List[float] = []
    for i in range(1, len(prices)):
        p0 = float(prices[i - 1])
        p1 = float(prices[i])
        if p0 <= 0:
            continue
        rets.append((p1 - p0) / p0)
    return rets


def _corr(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n < 5:
        return 0.0
    a = a[-n:]
    b = b[-n:]

    ma = sum(a) / n
    mb = sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return 0.0
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return float(cov / math.sqrt(va * vb))


def _best_lag_corr(lead: List[float], target: List[float], max_lag: int) -> Tuple[float, int]:
    """Find best correlation by shifting lead vs target.

    lag > 0 means: lead leads target by `lag` steps (lead earlier).
    """

    best = 0.0
    best_lag = 0

    for lag in range(0, max_lag + 1):
        if lag == 0:
            c = _corr(lead, target)
        else:
            # lead earlier: compare lead[:-lag] with target[lag:]
            if len(lead) <= lag or len(target) <= lag:
                continue
            c = _corr(lead[:-lag], target[lag:])

        if abs(c) > abs(best):
            best = c
            best_lag = lag

    return float(best), int(best_lag)


@dataclass
class CorrelationTrigger:
    """Layer B: correlation-based timing confirmation.

    MVP behavior:
    - If bias is LONG/SHORT, confirm with correlation between lead_symbol and target symbol.
    - Requires |corr| >= threshold.
    - Emits CorrelationSignal with confidence ~ |corr|.

    This stays simple and auditable; you can later swap to more advanced lead-lag models.
    """

    logger: logging.Logger

    lead_symbol: str = "BTC/USDT"
    window: int = 30
    max_lag: int = 3
    corr_threshold: float = 0.25

    def generate(self, bias: MarketBias, symbol: str, prices: MultiPriceProvider, now: Optional[datetime] = None) -> Optional[CorrelationSignal]:
        now = now or datetime.utcnow()

        if bias == "NEUTRAL":
            return None

        lead_prices = prices.get_recent_prices(self.lead_symbol, self.window)
        tgt_prices = prices.get_recent_prices(symbol, self.window)

        lead_rets = _pct_returns(lead_prices)
        tgt_rets = _pct_returns(tgt_prices)

        corr, lag = _best_lag_corr(lead_rets, tgt_rets, self.max_lag)
        strength = abs(float(corr))

        allowed = strength >= float(self.corr_threshold)
        side = "LONG" if bias == "LONG" else "SHORT"

        payload = {
            "module": "CorrelationTrigger",
            "timestamp": utc_iso(now),
            "symbol": symbol,
            "lead_symbol": self.lead_symbol,
            "bias": bias,
            "corr": round(float(corr), 4),
            "lag": int(lag),
            "window": int(self.window),
            "threshold": float(self.corr_threshold),
            "decision": "APPROVED" if allowed else "DENIED",
            "why": "Correlation above threshold" if allowed else "Correlation below threshold",
        }
        log_json(self.logger, payload, level=logging.INFO)

        if not allowed:
            return None

        return CorrelationSignal(
            symbol=symbol,
            lead_symbol=self.lead_symbol,
            side=side,
            confidence=float(min(max(strength, 0.0), 1.0)),
            corr=float(corr),
            lag=int(lag),
            window=int(self.window),
            why="Lead/target correlation confirmed",
            evidence_json=payload,
        )
