"""Sentiment layer (Layer A) for AuraQuant."""

from .sentiment_processor import SentimentProcessor
from .providers import (
    NewsProvider,
    CryptoPanicProvider,
    StaticNewsProvider,
    CoinGeckoTrendingProvider,
    CoinGeckoMomentumProvider,
    ChainNewsProvider,
    create_default_news_provider,
)
from .types import NewsItem, SentimentReport, MarketBias

__all__ = [
    "SentimentProcessor",
    "NewsProvider",
    "CryptoPanicProvider",
    "StaticNewsProvider",
    "CoinGeckoTrendingProvider",
    "CoinGeckoMomentumProvider",
    "ChainNewsProvider",
    "create_default_news_provider",
    "NewsItem",
    "SentimentReport",
    "MarketBias",
]
