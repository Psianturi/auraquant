from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..util.jsonlog import log_json, utc_iso
from .providers import NewsProvider
from .types import MarketBias, NewsItem, SentimentReport
from .gemini_scorer import get_gemini_scorer, GeminiSentimentResult


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def _dedup_key(item: NewsItem) -> str:
    base = (item.url or "") + "|" + item.source + "|" + _normalize_text(item.title)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


@dataclass
class SentimentProcessor:
    """Turns noisy headlines into a discrete market bias.

    MVP:
    - Fetch latest N items from a NewsProvider
    - Deduplicate by title/url hash
    - Score each item using Gemini AI (with heuristic fallback)
    - Apply half-life decay and aggregate into a single score
    - Map score into LONG/SHORT/NEUTRAL
    - Emit JSON evidence logs

    Live trading features:
    - News caching with TTL (default 10 min) to avoid API spam
    - Gemini 2.5 Pro for intelligent sentiment analysis
    - Fallback to heuristic when Gemini unavailable

    You can later swap the scorer to an LLM and keep everything else intact.
    """

    logger: logging.Logger
    provider: NewsProvider

    half_life_minutes: float = 30.0

    long_threshold: float = field(default_factory=lambda: float(os.getenv("SENTIMENT_LONG_THRESHOLD", "0.28")))
    short_threshold: float = field(default_factory=lambda: float(os.getenv("SENTIMENT_SHORT_THRESHOLD", "-0.18")))

    # News caching for live trading (TTL-based refresh)
    news_cache_ttl_minutes: float = 10.0  # Recommended: 10 min for CryptoPanic free tier
    _news_cache: Dict[str, List[NewsItem]] = field(default_factory=dict, init=False, repr=False)
    _cache_updated_at: Dict[str, datetime] = field(default_factory=dict, init=False, repr=False)
    
    _last_gemini_result: Optional[GeminiSentimentResult] = field(default=None, init=False, repr=False)
    _last_gemini_call_at: Dict[str, datetime] = field(default_factory=dict, init=False, repr=False)

    # Fallback behavior when provider fails
    fallback_on_error: MarketBias = "NEUTRAL"

    def analyze(self, symbol: str, limit: int = 5, now: Optional[datetime] = None) -> SentimentReport:
        now = now or datetime.utcnow()

        raw_items = self._get_news_with_cache(symbol=symbol, limit=limit, now=now)
        items = self._deduplicate(raw_items)

        disable_gemini = os.getenv("DISABLE_GEMINI_API", "0") == "1"
        use_gemini = (os.getenv("USE_GEMINI_SENTIMENT", "1") == "1") and (not disable_gemini)
        gemini_result: Optional[GeminiSentimentResult] = None

        try:
            gemini_min_interval_seconds = float(os.getenv("GEMINI_MIN_INTERVAL_SECONDS", "120"))
        except Exception:
            gemini_min_interval_seconds = 120.0
        if use_gemini and gemini_min_interval_seconds > 0:
            key = symbol.upper().strip()
            last_call = self._last_gemini_call_at.get(key)
            if last_call is not None:
                since = (now - last_call).total_seconds()
                if since < gemini_min_interval_seconds:
                    use_gemini = False
        
        if use_gemini and items:
            try:
                scorer = get_gemini_scorer()
                if scorer.is_available():
                    self._last_gemini_call_at[symbol.upper().strip()] = now
                    headlines = [item.title for item in items]
                    gemini_result = scorer.analyze_headlines(headlines, symbol, now)
                    self._last_gemini_result = gemini_result
                    
                    if gemini_result.model != "heuristic-fallback":
                    
                        agg_score = gemini_result.score
                        bias = self._score_to_bias(agg_score)
                        
                        avg_age = 0.0
                        if items:
                            avg_age = sum(max((now - i.published_at).total_seconds() / 60.0, 0.0) for i in items) / len(items)
                        
                        top_headline = items[0].title if items else None
                        
                        payload = {
                            "module": "SentimentProcessor",
                            "timestamp": utc_iso(now),
                            "symbol": symbol,
                            "bias": bias,
                            "score": round(float(agg_score), 4),
                            "model": gemini_result.model,
                            "reasoning": gemini_result.reasoning,
                            "confidence": round(gemini_result.confidence, 4),
                            "evidence": {
                                "news_count": len(items),
                                "avg_age_mins": round(float(avg_age), 2),
                                "top_headline": top_headline,
                                "scorer": "gemini",
                            },
                        }
                        log_json(self.logger, payload, level=logging.INFO)
                        
                        return SentimentReport(
                            symbol=symbol,
                            bias=bias,
                            score=float(agg_score),
                            news_count=len(items),
                            avg_age_mins=float(avg_age),
                            top_headline=top_headline,
                            decay_applied=f"Gemini-{gemini_result.model}",
                            evidence_json=payload,
                        )
            except Exception as e:
                self.logger.warning(f"[SentimentProcessor] Gemini analysis failed: {e}")

        # Fallback to heuristic scoring
        scored: List[Tuple[NewsItem, float, float]] = []
        # tuple: (item, sentiment_score[-1..1], weight)
        for item in items:
            age_mins = max((now - item.published_at).total_seconds() / 60.0, 0.0)
            weight = self._half_life_weight(age_mins)
            s = self._heuristic_score(item.title)
            scored.append((item, s, weight))

        agg_score = self._aggregate(scored)
        bias = self._score_to_bias(agg_score)

        avg_age = 0.0
        if items:
            avg_age = sum(max((now - i.published_at).total_seconds() / 60.0, 0.0) for i in items) / len(items)

        top_headline = items[0].title if items else None

        payload = {
            "module": "SentimentProcessor",
            "timestamp": utc_iso(now),
            "symbol": symbol,
            "bias": bias,
            "score": round(float(agg_score), 4),
            "evidence": {
                "news_count": len(items),
                "avg_age_mins": round(float(avg_age), 2),
                "top_headline": top_headline,
                "decay_applied": f"Half-life ({self.half_life_minutes:.0f}m)",
            },
        }
        log_json(self.logger, payload, level=logging.INFO)

        return SentimentReport(
            symbol=symbol,
            bias=bias,
            score=float(agg_score),
            news_count=len(items),
            avg_age_mins=float(avg_age),
            top_headline=top_headline,
            decay_applied=f"Half-life ({self.half_life_minutes:.0f}m)",
            evidence_json=payload,
        )

    def _deduplicate(self, items: Iterable[NewsItem]) -> List[NewsItem]:
        seen = set()
        out: List[NewsItem] = []
        for item in items:
            key = _dedup_key(item)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        # Keep newest first
        out.sort(key=lambda x: x.published_at, reverse=True)
        return out

    def _half_life_weight(self, age_minutes: float) -> float:
        # w(t) = 0.5^(t / h)
        h = max(float(self.half_life_minutes), 1e-6)
        return float(0.5 ** (float(age_minutes) / h))

    def _aggregate(self, scored: List[Tuple[NewsItem, float, float]]) -> float:
        if not scored:
            return 0.0
        num = sum(s * w for (_i, s, w) in scored)
        den = sum(w for (_i, _s, w) in scored)
        if den <= 0:
            return 0.0
        return float(num / den)

    def _score_to_bias(self, score: float) -> MarketBias:
        if score >= self.long_threshold:
            return "LONG"
        if score <= self.short_threshold:
            return "SHORT"
        return "NEUTRAL"

    def _get_news_with_cache(self, symbol: str, limit: int, now: datetime) -> List[NewsItem]:
        """Fetch news with TTL-based caching for live trading.

        - If cache is fresh (within TTL), return cached items
        - If cache is stale or missing, fetch fresh and update cache
        - On provider error, return cached items (stale) or empty list with fallback
        """
        cache_key = symbol.upper()
        last_update = self._cache_updated_at.get(cache_key)
        ttl = timedelta(minutes=float(self.news_cache_ttl_minutes))

        # Check if cache is still valid
        if last_update is not None and (now - last_update) < ttl:
            cached = self._news_cache.get(cache_key, [])
            if cached:
                return cached[:limit]

        # Cache is stale or missing, fetch fresh
        try:
            fresh_items = self.provider.fetch_latest(symbol=symbol, limit=limit * 2)  # Fetch extra for dedup
            self._news_cache[cache_key] = list(fresh_items)
            self._cache_updated_at[cache_key] = now
            return fresh_items[:limit]
        except Exception as e:
            # Fallback: use stale cache if available, otherwise empty
            self.logger.warning(f"News provider failed for {symbol}: {e}. Using cached/fallback.")
            cached = self._news_cache.get(cache_key, [])
            return cached[:limit] if cached else []

    def _heuristic_score(self, text: str) -> float:
        """Transparent keyword scorer for MVP.

        Returns:
        -1..+1 (clipped)
        """

        t = _normalize_text(text)

        positive = [
            "surge",
            "record",
            "ath",
            "inflows",
            "approval",
            "partnership",
            "upgrade",
            "growth",
            "bullish",
            "tvl",
        ]
        negative = [
            "hack",
            "exploit",
            "lawsuit",
            "ban",
            "outflows",
            "liquidation",
            "collapse",
            "bearish",
            "dump",
            "downtime",
        ]

        pos_hits = sum(1 for w in positive if w in t)
        neg_hits = sum(1 for w in negative if w in t)

        raw = pos_hits - neg_hits
        if raw == 0:
            return 0.0

        # squashing: keeps scores bounded and interpretable
        score = math.tanh(raw / 2.0)
        return float(max(min(score, 1.0), -1.0))
