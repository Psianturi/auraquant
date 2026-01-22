from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

Side = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class FeatureVector:
    """Compact, auditable feature vector for learning win/loss probability.

    Keep features simple, bounded, and explainable.
    """

    sentiment_score: float  # [-1, 1]
    corr: float  # [-1, 1]
    atr_pct: float  # [0, 0.2] typical
    lag_norm: float  # [0, 1]
    base_confidence: float  # [0, 1]
    side_long: float  # 1.0 if LONG else 0.0
    cg_volume_24h_rank: float  # [0, 1]
    cg_price_change_24h_rank: float  # [0, 1]
    cg_btc_dominance: float  # [0, 100]
    cg_total_mcap_change_24h_pct: float  # [-10, 10] typical

    def to_dict(self) -> Dict[str, float]:
        return {
            "sentiment_score": float(self.sentiment_score),
            "corr": float(self.corr),
            "atr_pct": float(self.atr_pct),
            "lag_norm": float(self.lag_norm),
            "base_confidence": float(self.base_confidence),
            "side_long": float(self.side_long),
            "cg_volume_24h_rank": float(self.cg_volume_24h_rank),
            "cg_price_change_24h_rank": float(self.cg_price_change_24h_rank),
            "cg_btc_dominance": float(self.cg_btc_dominance),
            "cg_total_mcap_change_24h_pct": float(self.cg_total_mcap_change_24h_pct),
        }


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def extract_features(
    *,
    side: Side,
    sentiment_score: float,
    corr: float,
    lag: int,
    max_lag: int,
    atr: float,
    price: float,
    base_confidence: float,
    cg_volume_24h_rank: float,
    cg_price_change_24h_rank: float,
    cg_btc_dominance: float,
    cg_total_mcap_change_24h_pct: float,
) -> FeatureVector:
    price = max(float(price), 1e-12)
    atr_pct = _clip(float(atr) / price, 0.0, 0.2)
    lag_norm = 0.0
    if int(max_lag) > 0:
        lag_norm = _clip(float(lag) / float(max_lag), 0.0, 1.0)

    return FeatureVector(
        sentiment_score=_clip(float(sentiment_score), -1.0, 1.0),
        corr=_clip(float(corr), -1.0, 1.0),
        atr_pct=atr_pct,
        lag_norm=lag_norm,
        base_confidence=_clip(float(base_confidence), 0.0, 1.0),
        side_long=1.0 if side == "LONG" else 0.0,
        cg_volume_24h_rank=_clip(cg_volume_24h_rank, 0.0, 1.0),
        cg_price_change_24h_rank=_clip(cg_price_change_24h_rank, 0.0, 1.0),
        cg_btc_dominance=_clip(cg_btc_dominance, 0.0, 100.0),
        cg_total_mcap_change_24h_pct=_clip(cg_total_mcap_change_24h_pct, -50.0, 50.0),
    )
