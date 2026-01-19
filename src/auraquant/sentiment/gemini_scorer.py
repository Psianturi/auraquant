"""Gemini-based sentiment scorer for AuraQuant.

Uses Google's Gemini 2.5 for intelligent sentiment analysis.
Migrated to new google.genai SDK (google-generativeai deprecated).
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
    score: float  # -1.0 (very bearish) to 1.0 (very bullish)
    confidence: float
    reasoning: str
    model: str
    raw_response: Optional[str] = None


@dataclass
class GeminiScorer:
    """Gemini-powered sentiment scorer.
    
    Uses Gemini 2.5 for intelligent sentiment analysis of crypto headlines.
    Falls back to heuristic scoring if Gemini API fails.
    """
    
    api_key: Optional[str] = None
    model_name: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    _client: Optional[Any] = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    _use_new_sdk: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            logger.warning("[GeminiScorer] No API key found. Using heuristic fallback.")
            return
        
        # Try new SDK first (google.genai), fallback to old (google.generativeai)
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=self.api_key)
            self._client = client
            self._use_new_sdk = True
            self._initialized = True
            logger.info(f"[GeminiScorer] Initialized with NEW SDK, model: {self.model_name}")
        except ImportError:
            # Fallback to old SDK
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model_name)
                self._use_new_sdk = False
                self._initialized = True
                logger.info(f"[GeminiScorer] Initialized with OLD SDK (deprecated), model: {self.model_name}")
            except ImportError:
                logger.warning("[GeminiScorer] No Gemini SDK installed. Using heuristic fallback.")
            except Exception as e:
                logger.warning(f"[GeminiScorer] Failed to initialize: {e}")
        except Exception as e:
            logger.warning(f"[GeminiScorer] Failed to initialize new SDK: {e}")
    
    def is_available(self) -> bool:
        """Check if Gemini API is available."""
        return self._initialized and self._client is not None
    
    def _call_gemini(self, prompt: str, max_tokens: int = 1020, temperature: float = 0.1) -> Optional[str]:
        """Call Gemini API with proper error handling for both SDKs."""
        if not self.is_available():
            return None
        
        try:
            if self._use_new_sdk:
                # New SDK: google.genai
                from google.genai import types
                
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                )
                
                text = None
                
                # Method 1: Direct .text attribute
                try:
                    if hasattr(response, 'text'):
                        text = response.text
                        if text:
                            logger.info(f"[GeminiScorer] Got text via .text")
                except Exception as e:
                    logger.info(f"[GeminiScorer] .text error: {e}")
                
                # Method 2: Access via candidates
                if not text and hasattr(response, 'candidates') and response.candidates:
                    try:
                        candidate = response.candidates[0]
                        finish_reason = getattr(candidate, 'finish_reason', None)
                        logger.info(f"[GeminiScorer] finish_reason={finish_reason}")
                        
                        if hasattr(candidate, 'content') and candidate.content:
                            parts = getattr(candidate.content, 'parts', [])
                            if parts and len(parts) > 0:
                                text = getattr(parts[0], 'text', None)
                                if text:
                                    logger.info(f"[GeminiScorer] Got text via candidates")
                    except Exception as e:
                        logger.info(f"[GeminiScorer] candidates error: {e}")
                
                if text:
                    logger.info(f"[GeminiScorer] Response: {text[:80]}...")
                    return text.strip()
                else:
                    has_cands = hasattr(response, 'candidates') and bool(response.candidates)
                    logger.info(f"[GeminiScorer] No text, has_candidates={has_cands}")
                    return None
            else:
                # Old SDK: google.generativeai
                response = self._client.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    }
                )
                
                if not response:
                    return None
                
                # Check candidates for safety blocks
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    if finish_reason and hasattr(finish_reason, 'value') and finish_reason.value == 2:
                        logger.debug("[GeminiScorer] Response blocked by safety filter")
                        return None
                
                try:
                    text = response.text
                    return text.strip() if text else None
                except (ValueError, AttributeError):
                    return None
                    
        except Exception as e:
            logger.debug(f"[GeminiScorer] API call failed: {e}")
            return None
    
    def analyze_headlines(
        self, 
        headlines: List[str], 
        symbol: str,
        now: Optional[datetime] = None
    ) -> GeminiSentimentResult:
        """Analyze headlines using Gemini and return sentiment with reasoning."""
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
        safe_headlines = [h.replace("dump", "drop").replace("crash", "decline") for h in headlines[:10]]
        headlines_text = "\n".join(f"- {h}" for h in safe_headlines)
        
        prompt = f"""You are a financial data analyst. Analyze market sentiment for {base_asset} cryptocurrency based on these data points.

Data points:
{headlines_text}

Respond with JSON only (no markdown):
{{"score": <number from -1.0 to 1.0>, "confidence": <number from 0.0 to 1.0>, "reasoning": "<one sentence explanation>"}}

Scoring: positive number = optimistic outlook, negative = cautious outlook, near zero = neutral"""

        raw_text = self._call_gemini(prompt, max_tokens=510, temperature=0.2)
        
        if not raw_text:
            logger.info(f"[GeminiScorer] No response, using heuristic fallback")
            return self._fallback_score(headlines)
        
        try:
            logger.debug(f"[GeminiScorer] Raw response: {raw_text[:200]}")
            
            if "```" in raw_text:
                parts = raw_text.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        raw_text = part
                        break
            
            result = json.loads(raw_text)
            
            score = float(result.get("score", result.get("sentiment_score", 0.0)))
            score = max(-1.0, min(1.0, score))
            
            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            
            reasoning = str(result.get("reasoning", "Gemini analysis"))
            
            logger.info(f"[GeminiScorer] Success: score={score:.2f}, reasoning={reasoning[:50]}")
            
            return GeminiSentimentResult(
                score=score,
                confidence=confidence,
                reasoning=reasoning,
                model=self.model_name,
                raw_response=raw_text
            )
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"[GeminiScorer] Parse error: {e}, raw: {raw_text[:100]}")
            return self._fallback_score(headlines)
    
    def generate_trade_reasoning(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        sentiment_score: float,
        correlation: float,
        momentum_1h: Optional[float] = None,
    ) -> str:
        """Generate human-readable reasoning for a trade decision."""
        if not self.is_available():
            return self._fallback_reasoning(symbol, side, entry_price, confidence, sentiment_score)
        
        momentum_str = f", 1h momentum: {momentum_1h:+.2f}%" if momentum_1h else ""
        
        prompt = f"""Explain this crypto trade in 2 sentences:
{side} {symbol} at ${entry_price:.2f}, SL ${stop_loss:.2f}, TP ${take_profit:.2f}
Sentiment: {sentiment_score:+.2f}, BTC corr: {correlation:.2f}{momentum_str}
Be direct, mention key signals."""

        text = self._call_gemini(prompt, max_tokens=256, temperature=0.3)
        
        if text:
            return text
        return self._fallback_reasoning(symbol, side, entry_price, confidence, sentiment_score)
    
    def _fallback_score(self, headlines: List[str]) -> GeminiSentimentResult:
        """Fallback heuristic scoring when Gemini is unavailable."""
        import math
        
        positive = ["surge", "record", "ath", "inflows", "approval", "partnership", 
                   "upgrade", "growth", "bullish", "tvl", "rally", "breakout", "pump"]
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
        sentiment_desc = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
        
        return (
            f"AI trading decision using AuraQuant model. "
            f"Opening {side} position on {symbol} at ${entry_price:,.2f} "
            f"based on {sentiment_desc} sentiment (score: {sentiment_score:+.2f}) "
            f"with {confidence:.0%} confidence. Powered by AuraQuant AI."
        )


# Global singleton
_gemini_scorer: Optional[GeminiScorer] = None


def get_gemini_scorer() -> GeminiScorer:
    """Get or create the global Gemini scorer instance."""
    global _gemini_scorer
    if _gemini_scorer is None:
        _gemini_scorer = GeminiScorer()
    return _gemini_scorer
