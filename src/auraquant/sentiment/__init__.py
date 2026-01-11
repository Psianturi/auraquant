"""Sentiment layer (Layer A) for AuraQuant."""

from .sentiment_processor import SentimentProcessor
from .providers import NewsProvider, CryptoPanicProvider, StaticNewsProvider
from .types import NewsItem, SentimentReport, MarketBias

__all__ = [
    "SentimentProcessor",
    "NewsProvider",
    "CryptoPanicProvider",
    "StaticNewsProvider",
    "NewsItem",
    "SentimentReport",
    "MarketBias",
]
