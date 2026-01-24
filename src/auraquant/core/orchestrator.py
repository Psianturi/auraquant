from __future__ import annotations

import logging
import os
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from ..data.multi_price_provider import MultiPriceProvider
from ..data.coingecko_client import CoinGeckoClient, WEEX_BASE_TO_COINGECKO_ID
from ..correlation import CorrelationTrigger
from ..risk import RiskEngine, TradeIntent
from ..sentiment import SentimentProcessor
from ..util.jsonlog import log_json, utc_iso
from ..execution.base_order_manager import BaseOrderManager
from ..util.ai_log.store import AiLogEvent, AiLogStore
from ..util.ai_log.realtime_uploader import RealTimeAiLogUploader
from ..weex import is_allowed_contract_symbol, to_weex_contract_symbol
from ..learning import TradePolicyLearner, extract_features

from .types import BotPhase, OrchestratorConfig, PositionSnapshot, ReconcileSnapshot, TickContext


@dataclass
class Orchestrator:
    """State-machine style orchestrator for AuraQuant.

    Key principle:
    - AI layers (Sentiment/Correlation) produce *intent*
    - RiskEngine is the final gatekeeper
    - Execution layer is deterministic (paper in MVP)

    MVP simplifications:
    - Single symbol
    - Correlation trigger is stubbed: bias LONG→LONG intent, bias SHORT→SHORT intent
    - One open position max
    """

    logger: logging.Logger
    config: OrchestratorConfig

    sentiment: SentimentProcessor
    correlation: CorrelationTrigger
    risk: RiskEngine
    prices: MultiPriceProvider
    execution: BaseOrderManager

    learner: Optional[TradePolicyLearner] = None

    ai_log_store: Optional[AiLogStore] = None
    ai_log_uploader: Optional[RealTimeAiLogUploader] = None

    phase: BotPhase = BotPhase.SCAN
    last_entry_at: Optional[datetime] = None

    _qualified_features_by_symbol: Dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _open_features_by_symbol: Dict[str, object] = field(default_factory=dict, init=False, repr=False)
    # Cache sentiment from SCAN phase to use in QUALIFY phase (same tick context)
    _last_scan_report: Optional[object] = field(default=None, init=False, repr=False)
    _last_scan_symbol: Optional[str] = field(default=None, init=False, repr=False)

    _offline_policy_cache: Optional[dict] = field(default=None, init=False, repr=False)
    _offline_policy_mtime: float = field(default=0.0, init=False, repr=False)

    def _offline_policy_multiplier(self, symbol: str, side: str) -> float:

        if os.getenv("ENABLE_OFFLINE_POLICY", "0") != "1":
            return 1.0

        path = os.getenv("OFFLINE_POLICY_PATH", "models/offline_policy.json")
        try:
            st = os.stat(path)
            mtime = float(st.st_mtime)
        except Exception:
            return 1.0

        try:
            if self._offline_policy_cache is None or mtime != float(self._offline_policy_mtime):
                with open(path, "r", encoding="utf-8") as f:
                    self._offline_policy_cache = json.load(f)
                self._offline_policy_mtime = mtime
        except Exception:
            return 1.0

        obj = self._offline_policy_cache
        if not isinstance(obj, dict):
            return 1.0
        symbols = obj.get("symbols")
        if not isinstance(symbols, dict):
            return 1.0
        sym_rec = symbols.get(symbol)
        if not isinstance(sym_rec, dict):
            return 1.0
        side_rec = sym_rec.get(str(side).upper())
        if not isinstance(side_rec, dict):
            return 1.0
        if side_rec.get("enabled") is not True:
            return 1.0

        try:
            mult = float(side_rec.get("confidence_multiplier", 1.0))
        except Exception:
            mult = 1.0

        try:
            max_delta = float(os.getenv("OFFLINE_POLICY_MAX_MULTIPLIER", "0.15"))
        except Exception:
            max_delta = 0.15
        lo = 1.0 - max_delta
        hi = 1.0 + max_delta
        return float(min(max(mult, lo), hi))

    def step(self, now: datetime) -> None:
        """Advance the bot by one tick."""

        symbol = self.config.symbol

        # Always update market data for this tick (target + lead) and update internal history
        symbols = [symbol, self.correlation.lead_symbol]
        tick_map = self.prices.get_tick(symbols=symbols, now=now)
        last_price, atr = tick_map.get(symbol, (0.0, 0.0))
        tick = TickContext(now=now, symbol=symbol, last_price=last_price, atr=atr)

        # Always let execution layer process SL/TP first
        trade_closed_source = "SL/TP"
        trade_closed = self.execution.on_price_tick(symbol=symbol, price=last_price, now=now)
        if trade_closed is None and hasattr(self.execution, "pop_pending_trade_result"):
            try:
                trade_closed = self.execution.pop_pending_trade_result()  
                if trade_closed is not None:
                    trade_closed_source = "SYNC"
            except Exception:
                trade_closed = None
        if trade_closed is not None:
       
            self.risk.update_account_state(equity_now=self.execution.equity(), trade_result=trade_closed, now=now)

            if self.learner is not None:
                fv = self._open_features_by_symbol.pop(trade_closed.symbol, None)
                if fv is not None:
                    p_before, seen = self.learner.update(fv, is_win=trade_closed.is_win)
                    self.learner.save() # --- PERSIST LEARNING ---
                    payload = {
                        "module": "Learner",
                        "timestamp": utc_iso(now),
                        "event": "MODEL_UPDATED_AND_SAVED",
                        "symbol": trade_closed.symbol,
                        "label_is_win": bool(trade_closed.is_win),
                        "p_win_before": round(float(p_before), 6),
                        "seen_trades": int(seen),
                        "features": fv.to_dict(), 
                    }
                    log_json(self.logger, payload, level=logging.INFO)

                    if self.ai_log_store is not None:
                        self.ai_log_store.append(
                            AiLogEvent(
                                stage="LEARN",
                                model="AuraQuant.OnlineLogistic",
                                input={
                                    "symbol": trade_closed.symbol,
                                    "features": fv.to_dict(), 
                                    "label_is_win": bool(trade_closed.is_win),
                                },
                                output={
                                    "p_win_before": round(float(p_before), 6),
                                    "seen_trades": int(seen),
                                    "weights": self.learner.weights(),
                                },
                                explanation="Online update after trade close (win/loss).",
                                timestamp=now,
                            )
                        )

            if self.ai_log_store is not None:
                order_id_int = None
                if trade_closed.order_id is not None:
                    s = str(trade_closed.order_id).strip()
                    if s.isdigit():
                        try:
                            order_id_int = int(s)
                        except Exception:
                            order_id_int = None
                self.ai_log_store.append(
                    AiLogEvent(
                        stage="TRADE_CLOSED",
                        model=f"AuraQuant.{type(self.execution).__name__}",
                        input={
                            "symbol": trade_closed.symbol,
                            "pnl_usdt": round(trade_closed.pnl_usdt, 6),
                        },
                        output={
                            "equity_now": round(self.execution.equity(), 6),
                            "trade_count": self.execution.trade_count(),
                        },
                        explanation=(
                            "Position closed (SL/TP). Realized PnL applied to equity."
                            if trade_closed_source == "SL/TP"
                            else "Position closed (exchange sync). Realized PnL applied to equity."
                        ),
                        order_id=order_id_int,
                        timestamp=now,
                    )
                )

        if self.phase == BotPhase.SCAN:
            self._scan(tick)
            self.phase = BotPhase.QUALIFY
            return

        if self.phase == BotPhase.QUALIFY:
            intent = self._qualify(tick)
            if intent is None:
                self.phase = BotPhase.SCAN
                return

            self.phase = BotPhase.ENTER
            self._enter(tick, intent)
            self.phase = BotPhase.RECONCILE
            return

        if self.phase == BotPhase.MANAGE:
            self._manage(tick)
            self.phase = BotPhase.RECONCILE
            return

        if self.phase == BotPhase.EXIT:
            self.phase = BotPhase.RECONCILE
            return

        if self.phase == BotPhase.RECONCILE:
            self._reconcile(tick)

            if self.execution.position() is not None:
                self.phase = BotPhase.MANAGE
            else:
                self.phase = BotPhase.SCAN
            return

        # Fallback
        self.phase = BotPhase.SCAN

    def _scan(self, tick: TickContext) -> None:
        report = self.sentiment.analyze(symbol=tick.symbol.split("/")[0], limit=5, now=tick.now)
        
        # Cache for QUALIFY phase
        self._last_scan_report = report
        self._last_scan_symbol = tick.symbol
        
        gemini_reasoning = None
        gemini_model = "AuraQuant.SentimentHeuristic"
        if hasattr(self.sentiment, '_last_gemini_result') and self.sentiment._last_gemini_result:
            gr = self.sentiment._last_gemini_result
            if gr.model != "heuristic-fallback":
                gemini_reasoning = gr.reasoning
                gemini_model = f"AuraQuant.Gemini.{gr.model}"
        
        payload = {
            "module": "Orchestrator",
            "timestamp": utc_iso(tick.now),
            "phase": BotPhase.SCAN.value,
            "symbol": tick.symbol,
            "price": tick.last_price,
            "atr": tick.atr,
            "sentiment": {"bias": report.bias, "score": round(report.score, 4)},
            "gemini_reasoning": gemini_reasoning,
        }
        log_json(self.logger, payload, level=logging.INFO)

        if self.ai_log_store is not None:
            explanation = gemini_reasoning if gemini_reasoning else "Heuristic sentiment scoring with dedup + half-life decay."
            self.ai_log_store.append(
                AiLogEvent(
                    stage=BotPhase.SCAN.value,
                    model=gemini_model,
                    input={"symbol": tick.symbol, "limit": 5, "price": tick.last_price},
                    output={"bias": report.bias, "score": round(report.score, 6)},
                    explanation=explanation[:1000], 
                    timestamp=tick.now,
                )
            )

    def _qualify(self, tick: TickContext) -> Optional[TradeIntent]:
        if self.config.enforce_weex_allowlist:
            contract_symbol = to_weex_contract_symbol(tick.symbol)
            if not is_allowed_contract_symbol(contract_symbol):
                payload = {
                    "module": "Orchestrator",
                    "timestamp": utc_iso(tick.now),
                    "phase": BotPhase.QUALIFY.value,
                    "symbol": tick.symbol,
                    "deny": "SYMBOL_NOT_ALLOWED_BY_WEEX_RULES",
                    "weex_contract_symbol": contract_symbol,
                }
                log_json(self.logger, payload, level=logging.WARNING)
                return None

        # Use cached sentiment from SCAN phase if available and symbol matches
        if self._last_scan_report is not None and self._last_scan_symbol == tick.symbol:
            report = self._last_scan_report
        else:
            report = self.sentiment.analyze(symbol=tick.symbol.split("/")[0], limit=5, now=tick.now)

        # If position already open, not generate new intents.
        if self.execution.position() is not None:
            return None

        # --- CoinGecko Data Fetching & Feature Prep ---
        cg_client = CoinGeckoClient()
        cg_markets = []
        global_data = {}
        try:
            # BUG FIX: Explicitly filter to only include IDs that are in the WEEX_BASE_TO_COINGECKO_ID map
            symbols_in_use = getattr(self.config, "symbols", None) or [self.config.symbol]
            cids_to_fetch = [cid for sym in symbols_in_use if (cid := WEEX_BASE_TO_COINGECKO_ID.get(sym.split('/')[0])) is not None]
            
            if cids_to_fetch:
                cg_markets = cg_client.get_markets(ids=cids_to_fetch, ttl_seconds=180.0)
            
            global_data = cg_client.get_global(ttl_seconds=180.0)
        except Exception as e:
            log_json(self.logger, {"module": "Orchestrator", "event": "COINGECKO_FETCH_FAILED", "error": str(e)}, level=logging.ERROR)
        
        current_cid = WEEX_BASE_TO_COINGECKO_ID.get(tick.symbol.split('/')[0].upper())
        current_market = next((m for m in cg_markets if m.id == current_cid), None)
        # --------------------------------

        effective_bias = report.bias
        
        # Calculate price position in 24h range (0=low, 1=high)
        price_position_24h = 0.5 
        if current_market and current_market.high_24h and current_market.low_24h:
            high = float(current_market.high_24h)
            low = float(current_market.low_24h)
            current = float(current_market.current_price)
            if high > low:
                price_position_24h = (current - low) / (high - low)
        
        #  Inverse bias based on 24h price position
        # High price (>0.85) → favor SHORT, Low price (<0.15) → favor LONG
        contrarian_mode = os.getenv("CONTRARIAN_MODE", "0") == "1"
        if contrarian_mode and current_market:
            overbought_threshold = float(os.getenv("OVERBOUGHT_THRESHOLD", "0.85"))
            oversold_threshold = float(os.getenv("OVERSOLD_THRESHOLD", "0.15"))
            
            if price_position_24h > overbought_threshold:

                if effective_bias == "LONG":
                    effective_bias = "NEUTRAL"  # Cancel LONG at high
                elif report.bias == "NEUTRAL":
                    effective_bias = "SHORT"  # Suggest SHORT at high
            elif price_position_24h < oversold_threshold:
                # Price near 24h low → favor LONG (buy low)
                if effective_bias == "SHORT":
                    effective_bias = "NEUTRAL" 
                elif report.bias == "NEUTRAL":
                    effective_bias = "LONG"  
        
        # MOMENTUM OVERRIDE: Force bias on strong 1h moves (with overbought/oversold check)
        use_momentum_override = os.getenv("USE_MOMENTUM_OVERRIDE", "0") == "1" 
        if use_momentum_override and report.bias == "NEUTRAL" and current_market:
            change_1h = float(current_market.price_change_percentage_1h or 0.0)
            momentum_threshold = float(os.getenv("MOMENTUM_OVERRIDE_MIN_CHANGE", "1.5"))  # Increase from 0.8 to 1.5
            
            overbought_threshold = float(os.getenv("OVERBOUGHT_THRESHOLD", "0.85"))
            oversold_threshold = float(os.getenv("OVERSOLD_THRESHOLD", "0.15"))
            
            if change_1h < -momentum_threshold and price_position_24h > oversold_threshold:
                effective_bias = "SHORT"
            elif change_1h > momentum_threshold and price_position_24h < overbought_threshold:
                effective_bias = "LONG"

        if effective_bias == "NEUTRAL":
            return None

        # Anti-spam entry guard
        if self.last_entry_at is not None:
            delta = (tick.now - self.last_entry_at).total_seconds()
            if delta < self.config.min_entry_interval_seconds:
                return None

        signal = self.correlation.generate(bias=effective_bias, symbol=tick.symbol, prices=self.prices, now=tick.now)
        if signal is None:
            return None

        # 24h Trend Filter - Block entries against strong trends
        use_trend_filter = os.getenv("USE_TREND_FILTER", "1") == "1"
        if use_trend_filter and current_market:
            change_24h = float(current_market.price_change_percentage_24h or 0.0)
            block_threshold = float(os.getenv("TREND_BLOCK_THRESHOLD_PCT", "5.0"))
            if (signal.side == "LONG" and change_24h < -block_threshold) or \
               (signal.side == "SHORT" and change_24h > block_threshold):
                return None # Deny trade

        if self.ai_log_store is not None:
            ai_input = {
                "symbol": tick.symbol,
                "lead_symbol": self.correlation.lead_symbol,
                "bias": report.bias,
                "effective_bias": effective_bias,
            }
            if current_market:
                ai_input["price_position_24h"] = round(price_position_24h, 3)
                ai_input["price_24h_high"] = current_market.high_24h
                ai_input["price_24h_low"] = current_market.low_24h
                ai_input["price_current"] = current_market.current_price
            
            self.ai_log_store.append(
                AiLogEvent(
                    stage=BotPhase.QUALIFY.value,
                    model="AuraQuant.CorrelationTrigger",
                    input=ai_input,
                    output={
                        "side": signal.side,
                        "confidence": round(signal.confidence, 6),
                    },
                    explanation="Rolling return correlation confirmation.",
                    timestamp=tick.now,
                )
            )

        # Combine confidence: sentiment strength * correlation strength
        sentiment_strength = min(max(abs(report.score), 0.0), 1.0)
        confidence = float(min(max(sentiment_strength * signal.confidence, 0.0), 1.0))

        # Note: ML never bypasses RiskEngine; it only adjusts *intent confidence*.
        p_win = None
        if self.learner is not None:
            vol_rank, price_change_rank = 0.5, 0.5
            btc_dominance, total_mcap_change_pct = 50.0, 0.0

            if cg_markets:
                valid_markets = [m for m in cg_markets if m.total_volume is not None and m.price_change_percentage_24h is not None]
                if current_market and len(valid_markets) > 1:
                    vol_sorted = sorted(valid_markets, key=lambda m: m.total_volume or 0.0)
                    change_sorted = sorted(valid_markets, key=lambda m: m.price_change_percentage_24h or 0.0)
                    vol_idx = next((i for i, m in enumerate(vol_sorted) if m.id == current_market.id), -1)
                    change_idx = next((i for i, m in enumerate(change_sorted) if m.id == current_market.id), -1)
                    if vol_idx != -1: vol_rank = vol_idx / (len(valid_markets) - 1)
                    if change_idx != -1: price_change_rank = change_idx / (len(valid_markets) - 1)

            if global_data and isinstance(global_data.get('data'), dict):
                g_data = global_data['data']
                btc_dominance = float(g_data.get('market_cap_percentage', {}).get('btc', 50.0))
                total_mcap_change_pct = float(g_data.get('market_cap_change_percentage_24h_usd', 0.0))

            fv = extract_features(
                side=signal.side,
                sentiment_score=float(report.score),
                corr=float(signal.corr),
                lag=int(signal.lag),
                max_lag=int(self.correlation.max_lag),
                atr=float(tick.atr),
                price=float(tick.last_price),
                base_confidence=float(confidence),
                cg_volume_24h_rank=vol_rank,
                cg_price_change_24h_rank=price_change_rank,
                cg_btc_dominance=btc_dominance,
                cg_total_mcap_change_24h_pct=total_mcap_change_pct,
            )
            p = float(self.learner.score(fv))
            p_win = p
            confidence = float(min(max(confidence * (0.5 + 0.5 * p), 0.0), 1.0))
            self._qualified_features_by_symbol[tick.symbol] = fv

        try:
            mult = self._offline_policy_multiplier(tick.symbol, signal.side)
            confidence = float(min(max(confidence * float(mult), 0.0), 1.0))
        except Exception:
            pass
        if confidence < self.config.min_confidence:
            payload = {
                "module": "Orchestrator",
                "timestamp": utc_iso(tick.now),
                "phase": BotPhase.QUALIFY.value,
                "symbol": tick.symbol,
                "deny": "CONFIDENCE_TOO_LOW",
                "confidence": round(confidence, 4),
                "min_confidence": self.config.min_confidence,
            }
            log_json(self.logger, payload, level=logging.WARNING)
            return None

        if tick.atr < self.config.min_atr:
            payload = {
                "module": "Orchestrator",
                "timestamp": utc_iso(tick.now),
                "phase": BotPhase.QUALIFY.value,
                "symbol": tick.symbol,
                "deny": "ATR_TOO_LOW",
                "atr": round(tick.atr, 6),
                "min_atr": self.config.min_atr,
            }
            log_json(self.logger, payload, level=logging.WARNING)
            return None

        intent = TradeIntent(
            symbol=tick.symbol,
            side=signal.side,  # type: ignore[arg-type]
            entry_price=float(tick.last_price),
            atr=float(tick.atr),
            confidence=float(confidence),
            requested_leverage=float(self.config.default_leverage),
            timestamp=tick.now,
        )

        payload = {
            "module": "Orchestrator",
            "timestamp": utc_iso(tick.now),
            "phase": BotPhase.QUALIFY.value,
            "symbol": tick.symbol,
            "intent": {
                "side": signal.side,
                "entry_price": tick.last_price,
                "atr": tick.atr,
                "confidence": round(confidence, 4),
            },
            "p_win": None if p_win is None else round(float(p_win), 6),
            "why": "Sentiment directional + correlation confirmed",
        }
        log_json(self.logger, payload, level=logging.INFO)
        return intent

    def _enter(self, tick: TickContext, intent: TradeIntent) -> None:
        equity_now = self.execution.equity()
        decision = self.risk.validate_intent(intent_data=intent, equity_now=equity_now, now=tick.now)

        payload = {
            "module": "Orchestrator",
            "timestamp": utc_iso(tick.now),
            "phase": BotPhase.ENTER.value,
            "symbol": tick.symbol,
            "risk_decision": {
                "decision": decision.decision,
                "reason": decision.reason,
                "notional_usdt": decision.position_notional_usdt,
                "stop_loss": decision.stop_loss,
                "take_profit": decision.take_profit,
            },
        }
        log_json(self.logger, payload, level=logging.INFO)

        if self.ai_log_store is not None:
            self.ai_log_store.append(
                AiLogEvent(
                    stage=BotPhase.ENTER.value,
                    model="AuraQuant.RiskEngine",
                    input={
                        "intent": {
                            "symbol": intent.symbol,
                            "side": intent.side,
                            "entry_price": intent.entry_price,
                            "atr": intent.atr,
                            "confidence": intent.confidence,
                            "requested_leverage": intent.requested_leverage,
                        },
                        "equity_now": round(equity_now, 6),
                    },
                    output={
                        "allowed": decision.allowed,
                        "decision": decision.decision,
                        "reason": decision.reason,
                        "notional_usdt": decision.position_notional_usdt,
                        "stop_loss": decision.stop_loss,
                        "take_profit": decision.take_profit,
                    },
                    explanation=decision.reason,
                    timestamp=tick.now,
                )
            )

        if not decision.allowed:
            return

        assert decision.position_notional_usdt is not None
        assert decision.stop_loss is not None
        assert decision.take_profit is not None

        # Pre-check margin before attempting order (avoid 40015 errors)
        leverage = float(decision.leverage or 0.0)
        if leverage <= 0:
            leverage = float(getattr(intent, "requested_leverage", 1.0) or 1.0)
        leverage = max(leverage, 1.0)

        required_margin = decision.position_notional_usdt / leverage
        if hasattr(self.execution, 'available_margin'):
            avail = self.execution.available_margin()
            if avail < required_margin * 1.1: 
                log_json(self.logger, {
                    "module": "Orchestrator",
                    "timestamp": utc_iso(tick.now),
                    "event": "ORDER_SKIPPED",
                    "symbol": tick.symbol,
                    "reason": "Insufficient margin (pre-check)",
                    "available": round(avail, 2),
                    "required": round(required_margin, 2),
                }, level=logging.WARNING)
                return

        try:
            self.execution.open_position(
                symbol=tick.symbol,
                side=decision.side,
                entry_price=decision.entry_price,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                notional_usdt=decision.position_notional_usdt,
                now=tick.now,
            )
        except RuntimeError as e:
            err_msg = str(e)
            if "margin" in err_msg.lower() or "insufficient" in err_msg.lower():
                log_json(self.logger, {
                    "module": "Orchestrator",
                    "timestamp": utc_iso(tick.now),
                    "event": "ORDER_SKIPPED",
                    "symbol": tick.symbol,
                    "reason": "Insufficient margin",
                    "error": err_msg[:100],
                }, level=logging.WARNING)
                return
            if "40015" in err_msg:
                log_json(self.logger, {
                    "module": "Orchestrator",
                    "timestamp": utc_iso(tick.now),
                    "event": "ORDER_SKIPPED",
                    "symbol": tick.symbol,
                    "reason": "WEEX rejected order (40015)",
                    "error": err_msg[:200],
                }, level=logging.WARNING)
                return
            raise  # Re-raise other errors
        self.last_entry_at = tick.now

        if self.learner is not None:
            fv = self._qualified_features_by_symbol.pop(tick.symbol, None)
            if fv is not None:
                self._open_features_by_symbol[tick.symbol] = fv

        payload2 = {
            "module": "Orchestrator",
            "timestamp": utc_iso(tick.now),
            "event": "POSITION_OPENED",
            "symbol": tick.symbol,
            "side": decision.side,
            "entry_price": decision.entry_price,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
            "notional_usdt": decision.position_notional_usdt,
        }
        log_json(self.logger, payload2, level=logging.INFO)

        if self.ai_log_store is not None:
            self.ai_log_store.append(
                AiLogEvent(
                    stage="EXECUTION",
                    model=f"AuraQuant.{type(self.execution).__name__}",
                    input={
                        "symbol": tick.symbol,
                        "side": decision.side,
                        "entry_price": decision.entry_price,
                        "stop_loss": decision.stop_loss,
                        "take_profit": decision.take_profit,
                        "notional_usdt": decision.position_notional_usdt,
                    },
                    output={"opened": True},
                    explanation="Execution layer opened a position.",
                    timestamp=tick.now,
                )
            )

    def _manage(self, tick: TickContext) -> None:
        pos = self.execution.position()
        payload = {
            "module": "Orchestrator",
            "timestamp": utc_iso(tick.now),
            "phase": BotPhase.MANAGE.value,
            "symbol": tick.symbol,
            "price": tick.last_price,
            "position_open": pos is not None,
        }
        log_json(self.logger, payload, level=logging.INFO)

    def _reconcile(self, tick: TickContext) -> ReconcileSnapshot:

        # Paper execution is authoritative, so reconcile() is a no-op there.
        try:
            self.execution.reconcile(now=tick.now)
        except TypeError:
            # Backward compatibility if a concrete manager still has reconcile() without kwargs.
            self.execution.reconcile()  # type: ignore[call-arg]

        pos = self.execution.position()
        snap_pos: Optional[PositionSnapshot] = None
        if pos is not None:
            snap_pos = PositionSnapshot(
                symbol=pos.symbol,
                side=pos.side,
                entry_price=pos.entry_price,
                stop_loss=pos.stop_loss,
                take_profit=pos.take_profit,
                notional_usdt=pos.notional_usdt,
                is_open=pos.is_open,
                opened_at=pos.opened_at,
            )

        equity_start = self.execution.starting_equity()
        equity_now = self.execution.equity()
        pnl_total = float(equity_now - equity_start)
        trades_closed = self.execution.trades_closed()
        positions_opened = self.execution.positions_opened()
        trade_count = trades_closed
        snapshot = ReconcileSnapshot(
            equity_start=equity_start,
            equity_now=equity_now,
            pnl_total=pnl_total,
            trade_count=trade_count,
            positions_opened=positions_opened,
            trades_closed=trades_closed,
            position=snap_pos,
        )

        payload = {
            "module": "Orchestrator",
            "timestamp": utc_iso(tick.now),
            "phase": BotPhase.RECONCILE.value,
            "symbol": tick.symbol,
            "equity_start": snapshot.equity_start,
            "equity_now": snapshot.equity_now,
            "pnl_total": round(snapshot.pnl_total, 6),
            "trade_count": snapshot.trade_count,
            "positions_opened": snapshot.positions_opened,
            "trades_closed": snapshot.trades_closed,
            "position": None
            if snapshot.position is None
            else {
                "side": snapshot.position.side,
                "entry_price": snapshot.position.entry_price,
                "stop_loss": snapshot.position.stop_loss,
                "take_profit": snapshot.position.take_profit,
                "notional_usdt": snapshot.position.notional_usdt,
            },
        }
        log_json(self.logger, payload, level=logging.INFO)

        if self.ai_log_store is not None:
            self.ai_log_store.append(
                AiLogEvent(
                    stage=BotPhase.RECONCILE.value,
                    model="AuraQuant.EquityBook",
                    input={"symbol": tick.symbol},
                    output={
                        "equity_start": snapshot.equity_start,
                        "equity_now": snapshot.equity_now,
                        "pnl_total": round(snapshot.pnl_total, 6),
                        "trade_count": snapshot.trade_count,
                        "positions_opened": snapshot.positions_opened,
                        "trades_closed": snapshot.trades_closed,
                    },
                    explanation="Bookkeeping snapshot for judge-auditable PnL.",
                    timestamp=tick.now,
                )
            )

    def _push_ai_log(
        self,
        stage: str,
        model: str,
        input_dict: Dict,
        output_dict: Dict,
        explanation: str,
        order_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Helper: append to local store AND push to real-time uploader (if configured).

        Ensures WEEX compliance: every AI decision is logged real-time (max 1 min delay).
        """
        event = AiLogEvent(
            stage=stage,
            model=model,
            input=input_dict,
            output=output_dict,
            explanation=explanation,
            order_id=order_id,
            timestamp=timestamp or datetime.now(timezone.utc),
        )

        # Always store locally (audit trail)
        if self.ai_log_store is not None:
            self.ai_log_store.append(event)

        # Push to WEEX real-time uploader (if configured)
        if self.ai_log_uploader is not None:
            payload = event.to_payload()
            self.ai_log_uploader.upload(payload)
