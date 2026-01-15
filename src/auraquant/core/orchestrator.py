from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from ..data.multi_price_provider import MultiPriceProvider
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

    def step(self, now: datetime) -> None:
        """Advance the bot by one tick."""

        symbol = self.config.symbol

        # Always update market data for this tick (target + lead) and update internal history
        symbols = [symbol, self.correlation.lead_symbol]
        tick_map = self.prices.get_tick(symbols=symbols, now=now)
        last_price, atr = tick_map.get(symbol, (0.0, 0.0))
        tick = TickContext(now=now, symbol=symbol, last_price=last_price, atr=atr)

        # Always let execution layer process SL/TP first
        trade_closed = self.execution.on_price_tick(symbol=symbol, price=last_price, now=now)
        if trade_closed is not None:
       
            self.risk.update_account_state(equity_now=self.execution.equity(), trade_result=trade_closed, now=now)

            if self.learner is not None:
                fv = self._open_features_by_symbol.pop(trade_closed.symbol, None)
                if fv is not None:
                    p_before, seen = self.learner.update(fv, is_win=trade_closed.is_win)  # type: ignore[arg-type]
                    payload = {
                        "module": "Learner",
                        "timestamp": utc_iso(now),
                        "event": "MODEL_UPDATED",
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
                self.ai_log_store.append(
                    AiLogEvent(
                        stage="TRADE_CLOSED",
                        model="AuraQuant.PaperOrderManager",
                        input={
                            "symbol": trade_closed.symbol,
                            "pnl_usdt": round(trade_closed.pnl_usdt, 6),
                        },
                        output={
                            "equity_now": round(self.execution.equity(), 6),
                            "trade_count": self.execution.trade_count(),
                        },
                        explanation="Paper position closed by SL/TP; realized PnL applied to equity.",
                        order_id=int(trade_closed.order_id.split("_")[1]) if trade_closed.order_id and trade_closed.order_id.startswith("paper_") else None,
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
        payload = {
            "module": "Orchestrator",
            "timestamp": utc_iso(tick.now),
            "phase": BotPhase.SCAN.value,
            "symbol": tick.symbol,
            "price": tick.last_price,
            "atr": tick.atr,
            "sentiment": {"bias": report.bias, "score": round(report.score, 4)},
        }
        log_json(self.logger, payload, level=logging.INFO)

        if self.ai_log_store is not None:
            self.ai_log_store.append(
                AiLogEvent(
                    stage=BotPhase.SCAN.value,
                    model="AuraQuant.SentimentHeuristic",
                    input={"symbol": tick.symbol, "limit": 5, "price": tick.last_price},
                    output={"bias": report.bias, "score": round(report.score, 6)},
                    explanation="Heuristic sentiment scoring with dedup + half-life decay.",
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

        report = self.sentiment.analyze(symbol=tick.symbol.split("/")[0], limit=5, now=tick.now)

        # If position already open, not generate new intents.
        if self.execution.position() is not None:
            return None

        if report.bias == "NEUTRAL":
            return None

        # Anti-spam entry guard
        if self.last_entry_at is not None:
            delta = (tick.now - self.last_entry_at).total_seconds()
            if delta < self.config.min_entry_interval_seconds:
                return None

        signal = self.correlation.generate(bias=report.bias, symbol=tick.symbol, prices=self.prices, now=tick.now)
        if signal is None:
            return None

        if self.ai_log_store is not None:
            self.ai_log_store.append(
                AiLogEvent(
                    stage=BotPhase.QUALIFY.value,
                    model="AuraQuant.CorrelationTrigger",
                    input={
                        "symbol": tick.symbol,
                        "lead_symbol": self.correlation.lead_symbol,
                        "window": self.correlation.window,
                        "max_lag": self.correlation.max_lag,
                        "threshold": self.correlation.corr_threshold,
                        "bias": report.bias,
                    },
                    output={
                        "side": signal.side,
                        "confidence": round(signal.confidence, 6),
                        "lag": signal.lag,
                        "corr": round(signal.corr, 6),
                    },
                    explanation="Rolling return correlation confirmation; requires corr >= threshold.",
                    timestamp=tick.now,
                )
            )

        # Combine confidence: sentiment strength * correlation strength
        sentiment_strength = min(max(abs(report.score), 0.0), 1.0)
        confidence = float(min(max(sentiment_strength * signal.confidence, 0.0), 1.0))

        # Note: ML never bypasses RiskEngine; it only adjusts *intent confidence*.
        p_win = None
        if self.learner is not None:
            fv = extract_features(
                side=signal.side,
                sentiment_score=float(report.score),
                corr=float(signal.corr),
                lag=int(signal.lag),
                max_lag=int(self.correlation.max_lag),
                atr=float(tick.atr),
                price=float(tick.last_price),
                base_confidence=float(confidence),
            )
            p = float(self.learner.score(fv))
            p_win = p
            confidence = float(min(max(confidence * (0.5 + 0.5 * p), 0.0), 1.0))
            self._qualified_features_by_symbol[tick.symbol] = fv
        if confidence < self.config.min_confidence:
            return None

        if tick.atr < self.config.min_atr:
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

        self.execution.open_position(
            symbol=tick.symbol,
            side=decision.side,
            entry_price=decision.entry_price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            notional_usdt=decision.position_notional_usdt,
            now=tick.now,
        )
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
                    model="AuraQuant.PaperOrderManager",
                    input={
                        "symbol": tick.symbol,
                        "side": decision.side,
                        "entry_price": decision.entry_price,
                        "stop_loss": decision.stop_loss,
                        "take_profit": decision.take_profit,
                        "notional_usdt": decision.position_notional_usdt,
                    },
                    output={"opened": True},
                    explanation="Deterministic paper execution (no exchange interaction).",
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
