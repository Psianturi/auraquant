from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

MarketBias = Literal["LONG", "SHORT", "NEUTRAL"]


@dataclass(frozen=True)
class NewsItem:
    title: str
    published_at: datetime
    source: str = "unknown"
    url: Optional[str] = None


@dataclass(frozen=True)
class SentimentReport:
    symbol: str
    bias: MarketBias
    score: float

    news_count: int
    avg_age_mins: float
    top_headline: Optional[str]

    decay_applied: str

    evidence_json: dict = field(default_factory=dict)
