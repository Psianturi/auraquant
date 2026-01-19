"""Gemini-based sentiment scorer for AuraQuant.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GeminiSentimentResult:
    """Result from Gemini sentiment analysis."""
    score: float # -1.0 (very bearish) to 1.0 (very bullish)
    confidence: float  
    reasoning: str
    model: str
    raw_response: Optional[str] = None


@dataclass
class GeminiScorer:
    """Gemini-powered sentiment scorer.
    
    Uses Gemini 2.5 Pro for intelligent sentiment analysis of crypto headlines.
    Falls back to heuristic scoring if Gemini API fails.
    """
    
    api_key: Optional[str] = None
    model_name: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-pro-preview-05-06"))
    _client: Optional[Any] = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model_name)
                self._initialized = True
                logger.info(f"[GeminiScorer] Initialized with model: {self.model_name}")
            except ImportError:
                logger.warning("[GeminiScorer] google-generativeai not installed. Using heuristic fallback.")
            except Exception as e:
                logger.warning(f"[GeminiScorer] Failed to initialize: {e}. Using heuristic fallback.")
    
    def is_available(self) -> bool:
        """Check if Gemini API is available."""
        return self._initialized and self._client is not None
    
    def analyze_headlines(
        self, 
        headlines: List[str], 
        symbol: str,
        now: Optional[datetime] = None
    ) -> GeminiSentimentResult:
        """Analyze headlines using Gemini and return sentiment with reasoning.
        
        Args:
            headlines: List of news headlines to analyze
            symbol: Trading symbol (e.g., "BTC/USDT")
            now: Current timestamp
            
        Returns:
            GeminiSentimentResult with score, confidence, and reasoning
        """
        if not self.is_available():
            return self._fallback_score(headlines)
        
        if not headlines:
            return GeminiSentimentResult(
                score=0.0,
                confidence=0.0,
                reasoning="No headlines to analyze",
                model=self.model_name
            )
        
        base_asset = symbol.split("/")[0] if "/" in symbol else symbol
        headlines_text = "\n".join(f"- {h}" for h in headlines[:10])  # Max 10 headlines
        
        prompt = f"""You are a crypto trading sentiment analyst. Analyze these headlines for {base_asset} and determine market sentiment.

Headlines:
{headlines_text}

Respond in JSON format ONLY (no markdown):
{{
    "sentiment_score": <float from -1.0 (very bearish) to 1.0 (very bullish)>,
    "confidence": <float from 0.0 to 1.0>,
    "reasoning": "<brief explanation of your analysis>",
    "key_signals": ["<signal1>", "<signal2>"]
}}

Rules:
- Score > 0.3: Bullish signals (surge, ATH, inflows, approval, partnership)
- Score < -0.3: Bearish signals (hack, lawsuit, ban, liquidation, dump)
- Score near 0: Neutral or mixed signals
- Be concise in reasoning (max 100 words)"""

        try:
            response = self._client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 500,
                }
            )
            
            raw_text = response.text.strip()
            
            # Parse JSON response
            # Handle markdown code blocks if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()
            
            result = json.loads(raw_text)
            
            score = float(result.get("sentiment_score", 0.0))
            score = max(-1.0, min(1.0, score))  # Clamp
            
            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            
            reasoning = str(result.get("reasoning", ""))
            key_signals = result.get("key_signals", [])
            
            if key_signals:
                reasoning += f" Key signals: {', '.join(key_signals[:3])}"
            
            return GeminiSentimentResult(
                score=score,
                confidence=confidence,
                reasoning=reasoning,
                model=self.model_name,
                raw_response=raw_text
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"[GeminiScorer] JSON parse error: {e}")
            return self._fallback_score(headlines)
        except Exception as e:
            logger.warning(f"[GeminiScorer] API error: {e}")
            return self._fallback_score(headlines)
    
    def generate_trade_reasoning(
        self,
        symbol: str,
        side: str,  # "LONG" or "SHORT"
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        sentiment_score: float,
        correlation: float,
        momentum_1h: Optional[float] = None,
    ) -> str:
        """Generate human-readable reasoning for a trade decision.
        
        This is shown in the Model Chat panel on WEEX Labs.
        """
        if not self.is_available():
            return self._fallback_reasoning(symbol, side, entry_price, confidence, sentiment_score)
        
        prompt = f"""You are an AI trading assistant explaining a trade decision. Generate a concise explanation.

Trade Details:
- Symbol: {symbol}
- Direction: {side}
- Entry: ${entry_price:,.4f}
- Stop Loss: ${stop_loss:,.4f}
- Take Profit: ${take_profit:,.4f}
- Confidence: {confidence:.1%}
- Sentiment Score: {sentiment_score:+.2f}
- BTC Correlation: {correlation:.2f}
{f"- 1h Momentum: {momentum_1h:+.2f}%" if momentum_1h else ""}

Write a 2-3 sentence explanation of why this trade was taken. Be specific about the signals.
Format: Direct statement, no preamble."""

        try:
            response = self._client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 200,
                }
            )
            return response.text.strip()
        except Exception as e:
            logger.warning(f"[GeminiScorer] Reasoning generation failed: {e}")
            return self._fallback_reasoning(symbol, side, entry_price, confidence, sentiment_score)
    
    def _fallback_score(self, headlines: List[str]) -> GeminiSentimentResult:
        """Fallback heuristic scoring when Gemini is unavailable."""
        import math
        
        positive = ["surge", "record", "ath", "inflows", "approval", "partnership", 
                   "upgrade", "growth", "bullish", "tvl", "rally", "breakout"]
        negative = ["hack", "exploit", "lawsuit", "ban", "outflows", "liquidation",
                   "collapse", "bearish", "dump", "downtime", "crash", "selloff"]
        
        pos_hits = 0
        neg_hits = 0
        
        for h in headlines:
            h_lower = h.lower()
            pos_hits += sum(1 for w in positive if w in h_lower)
            neg_hits += sum(1 for w in negative if w in h_lower)
        
        raw = pos_hits - neg_hits
        score = math.tanh(raw / 2.0) if raw != 0 else 0.0
        
        return GeminiSentimentResult(
            score=float(score),
            confidence=0.5,
            reasoning=f"Heuristic: {pos_hits} bullish, {neg_hits} bearish signals",
            model="heuristic-fallback"
        )
    
    def _fallback_reasoning(
        self, 
        symbol: str, 
        side: str, 
        entry_price: float,
        confidence: float,
        sentiment_score: float
    ) -> str:
        """Fallback reasoning when Gemini is unavailable."""
        direction = "bullish" if side == "LONG" else "bearish"
        sentiment_desc = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
        
        return (
            f"AI trading decision using AuraQuant model. "
            f"Opening {side} position on {symbol} at ${entry_price:,.4f} "
            f"based on {sentiment_desc} sentiment (score: {sentiment_score:+.2f}) "
            f"with {confidence:.1%} confidence. "
            f"Powered by AuraQuant AI."
        )


# Global singleton for efficiency
_gemini_scorer: Optional[GeminiScorer] = None


def get_gemini_scorer() -> GeminiScorer:
    """Get or create the global Gemini scorer instance."""
    global _gemini_scorer
    if _gemini_scorer is None:
        _gemini_scorer = GeminiScorer()
    return _gemini_scorer
