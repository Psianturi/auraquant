from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json

import logging

from ..util.jsonlog import log_json, utc_iso
from .types import RiskDecision, Side, TradeIntent, TradeResult


class CircuitState(str, Enum):
    NORMAL = "NORMAL"
    COOLDOWN = "COOLDOWN"
    STOPPED = "STOPPED"


@dataclass
class CircuitBreaker:
    """Tracks whether the system is allowed to trade.

     rules:
    - Stop for the day if daily drawdown limit breached.
    - Cooldown if too many consecutive losses.

    This class is intentionally exchange-agnostic.
    """

    daily_drawdown_limit_pct: float = -2.0
    max_consecutive_losses: int = 3
    cooldown_minutes: int = 60

    state: CircuitState = CircuitState.NORMAL
    cooldown_until: Optional[datetime] = None

    equity_day_start: Optional[float] = None
    day: Optional[date] = None
    consecutive_losses: int = 0

    def to_json(self) -> Dict[str, Any]:
        return {
            "type": "circuit_breaker_v1",
            "state": self.state.value,
            "cooldown_until": utc_iso(self.cooldown_until) if self.cooldown_until else None,
            "equity_day_start": self.equity_day_start,
            "day": self.day.isoformat() if self.day else None,
            "consecutive_losses": int(self.consecutive_losses),
            "daily_drawdown_limit_pct": float(self.daily_drawdown_limit_pct),
            "max_consecutive_losses": int(self.max_consecutive_losses),
            "cooldown_minutes": int(self.cooldown_minutes),
        }

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "CircuitBreaker":
        if obj.get("type") != "circuit_breaker_v1":
            raise ValueError("Unsupported circuit breaker state")
        cb = cls(
            daily_drawdown_limit_pct=float(obj.get("daily_drawdown_limit_pct", -2.0)),
            max_consecutive_losses=int(obj.get("max_consecutive_losses", 3)),
            cooldown_minutes=int(obj.get("cooldown_minutes", 60)),
        )
        state = str(obj.get("state", CircuitState.NORMAL.value))
        cb.state = CircuitState(state) if state in {s.value for s in CircuitState} else CircuitState.NORMAL

        day_s = obj.get("day")
        if isinstance(day_s, str) and day_s:
            try:
                cb.day = date.fromisoformat(day_s)
            except Exception:
                cb.day = None

        cb.equity_day_start = float(obj["equity_day_start"]) if obj.get("equity_day_start") is not None else None
        cb.consecutive_losses = int(obj.get("consecutive_losses", 0))

        cu = obj.get("cooldown_until")
        if isinstance(cu, str) and cu:
            try:
                cb.cooldown_until = datetime.fromisoformat(cu.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                cb.cooldown_until = None
        return cb

    def _ensure_day(self, now: datetime) -> None:
        today = now.date()
        if self.day != today:
            self.day = today
            self.equity_day_start = None
            self.consecutive_losses = 0
            self.state = CircuitState.NORMAL
            self.cooldown_until = None

    def set_day_start_equity_if_missing(self, equity: float, now: Optional[datetime] = None) -> None:
        now = now or datetime.utcnow()
        self._ensure_day(now)
        if self.equity_day_start is None:
            self.equity_day_start = float(equity)

    def current_drawdown_pct(self, equity_now: float, now: Optional[datetime] = None) -> float:
        now = now or datetime.utcnow()
        self._ensure_day(now)
        if not self.equity_day_start or self.equity_day_start <= 0:
            return 0.0
        return (float(equity_now) - float(self.equity_day_start)) / float(self.equity_day_start) * 100.0

    def evaluate(self, equity_now: float, now: Optional[datetime] = None) -> Tuple[bool, str, Dict[str, Any]]:
        now = now or datetime.utcnow()
        self._ensure_day(now)

        if self.state == CircuitState.COOLDOWN:
            if self.cooldown_until and now < self.cooldown_until:
                return (
                    False,
                    f"Cooldown active until {utc_iso(self.cooldown_until)}",
                    {"state": self.state.value, "cooldown_until": utc_iso(self.cooldown_until)},
                )
            self.state = CircuitState.NORMAL
            self.cooldown_until = None

        dd_pct = self.current_drawdown_pct(equity_now, now)
        if dd_pct <= self.daily_drawdown_limit_pct:
            self.state = CircuitState.STOPPED
            return (
                False,
                f"Daily drawdown limit reached ({dd_pct:.2f}%)",
                {"state": self.state.value, "current_drawdown_pct": dd_pct, "limit_pct": self.daily_drawdown_limit_pct},
            )

        if self.state == CircuitState.STOPPED:
            return (
                False,
                "Circuit is STOPPED for today",
                {"state": self.state.value, "current_drawdown_pct": dd_pct, "limit_pct": self.daily_drawdown_limit_pct},
            )

        return True, "SAFE", {"state": self.state.value, "current_drawdown_pct": dd_pct, "limit_pct": self.daily_drawdown_limit_pct}

    def update_after_trade(self, trade_result: TradeResult, equity_now: float, now: Optional[datetime] = None) -> None:
        now = now or datetime.utcnow()
        self._ensure_day(now)
        self.set_day_start_equity_if_missing(equity_now, now)


        if self.state == CircuitState.STOPPED:
            if not trade_result.is_win:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            return

        if trade_result.is_win:
            self.consecutive_losses = 0
            return

        self.consecutive_losses += 1
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.state = CircuitState.COOLDOWN
            self.cooldown_until = now + timedelta(minutes=int(self.cooldown_minutes))


@dataclass
class RiskEngine:
    """Final safety filter that approves/denies trade intents.

    Responsibilities (MVP):
    - Circuit breaker gating (daily drawdown + loss streak)
    - Position sizing (simple risk-based)
    - SL/TP levels (ATR-based)
    - Evidence logging in JSON
    """

    logger: logging.Logger
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    state_path: Optional[str] = None
    _state_loaded: bool = field(default=False, init=False, repr=False)

    max_leverage_allowed: float = 20.0
    risk_per_trade_pct: float = 0.4  
    max_position_notional_pct: float = 4.0  

    sl_atr_mult: float = 2.0  
    tp_atr_mult: float = 4.0  
    min_atr_pct: float = 0.0035 
    
    # Sideways market filter - skip trading when volatility is too low
    min_atr_for_trade_pct: float = 0.002  

    def validate_intent(self, intent_data: TradeIntent, equity_now: float, now: Optional[datetime] = None) -> RiskDecision:
        now = now or datetime.utcnow()

        if self.state_path and not self._state_loaded:
            self._load_state_best_effort()
            self._state_loaded = True

        self.circuit_breaker.set_day_start_equity_if_missing(equity_now, now)

        atr_pct = float(intent_data.atr) / float(intent_data.entry_price) if float(intent_data.entry_price) > 0 else 0.0
        if atr_pct < self.min_atr_for_trade_pct:
            payload = {
                "module": "RiskEngine",
                "timestamp": utc_iso(now),
                "symbol": intent_data.symbol,
                "decision": "DENIED",
                "reason": "MARKET_TOO_SIDEWAYS",
                "metrics": {
                    "atr_pct": atr_pct,
                    "min_atr_for_trade_pct": self.min_atr_for_trade_pct,
                    "atr": float(intent_data.atr),
                    "entry_price": float(intent_data.entry_price),
                },
            }
            log_json(self.logger, payload, level=logging.WARNING)
            return RiskDecision(
                allowed=False,
                decision="DENIED",
                reason=f"Market too sideways (ATR {atr_pct*100:.3f}% < min {self.min_atr_for_trade_pct*100:.1f}%)",
                symbol=intent_data.symbol,
                side=intent_data.side,
                entry_price=float(intent_data.entry_price),
                evidence_json=payload,
            )

        allow, cb_reason, cb_metrics = self.circuit_breaker.evaluate(equity_now, now)
        if not allow:
            payload = {
                "module": "RiskEngine",
                "timestamp": utc_iso(now),
                "symbol": intent_data.symbol,
                "decision": "DENIED",
                "reason": cb_reason,
                "metrics": {
                    "equity_day_start": self.circuit_breaker.equity_day_start,
                    "equity_now": float(equity_now),
                    "current_drawdown_pct": cb_metrics.get("current_drawdown_pct"),
                    "limit_pct": cb_metrics.get("limit_pct"),
                    "consecutive_losses": self.circuit_breaker.consecutive_losses,
                    "state": cb_metrics.get("state"),
                },
            }
            log_json(self.logger, payload, level=logging.WARNING)
            return RiskDecision(
                allowed=False,
                decision="DENIED",
                reason=cb_reason,
                symbol=intent_data.symbol,
                side=intent_data.side,
                entry_price=float(intent_data.entry_price),
                evidence_json=payload,
            )

        leverage = min(float(intent_data.requested_leverage), float(self.max_leverage_allowed))
        if leverage <= 0:
            leverage = 1.0

        stop_loss, take_profit = self.calculate_sl_tp(
            entry_price=float(intent_data.entry_price),
            side=intent_data.side,
            volatility=float(intent_data.atr),
        )

        position_notional = self._calculate_position_notional(
            equity_now=float(equity_now),
            entry_price=float(intent_data.entry_price),
            stop_loss=float(stop_loss),
            side=intent_data.side,
            leverage=leverage,
        )

        payload = {
            "module": "RiskEngine",
            "timestamp": utc_iso(now),
            "symbol": intent_data.symbol,
            "decision": "APPROVED",
            "reason": "SAFE",
            "metrics": {
                "equity_day_start": self.circuit_breaker.equity_day_start,
                "equity_now": float(equity_now),
                "current_drawdown_pct": self.circuit_breaker.current_drawdown_pct(float(equity_now), now),
                "limit_pct": self.circuit_breaker.daily_drawdown_limit_pct,
                "consecutive_losses": self.circuit_breaker.consecutive_losses,
                "risk_per_trade_pct": self.risk_per_trade_pct,
                "max_position_notional_pct": self.max_position_notional_pct,
                "requested_leverage": float(intent_data.requested_leverage),
                "leverage_used": float(leverage),
                "atr": float(intent_data.atr),
                "sl_atr_mult": self.sl_atr_mult,
                "tp_atr_mult": self.tp_atr_mult,
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "position_notional_usdt": float(position_notional),
            },
        }
        log_json(self.logger, payload, level=logging.INFO)

        return RiskDecision(
            allowed=True,
            decision="APPROVED",
            reason="SAFE",
            symbol=intent_data.symbol,
            side=intent_data.side,
            entry_price=float(intent_data.entry_price),
            position_notional_usdt=float(position_notional),
            leverage=float(leverage),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            evidence_json=payload,
        )

    def calculate_sl_tp(self, entry_price: float, side: Side, volatility: float) -> Tuple[float, float]:
        atr = max(float(volatility), 0.0)
        
        # CRITICAL: Enforce minimum ATR as percentage of price
        # This prevents micro-scalping where fee > profit
        # 0.3% minimum ensures TP distance > 2x typical fee (0.06% x 2 = 0.12%)
        min_atr_abs = float(entry_price) * float(self.min_atr_pct)
        
        if atr < min_atr_abs:
            atr = min_atr_abs
        
        if atr == 0.0:
            # Fallback: 0.5% price-based bands if ATR missing
            atr = float(entry_price) * 0.005

        sl_dist = atr * float(self.sl_atr_mult)
        tp_dist = atr * float(self.tp_atr_mult)

        if side == "LONG":
            return max(entry_price - sl_dist, 0.0), max(entry_price + tp_dist, 0.0)
        return max(entry_price + sl_dist, 0.0), max(entry_price - tp_dist, 0.0)

    def update_account_state(self, equity_now: float, trade_result: TradeResult, now: Optional[datetime] = None) -> None:
        now = now or datetime.utcnow()
        self.circuit_breaker.update_after_trade(trade_result=trade_result, equity_now=float(equity_now), now=now)

        if self.state_path:
            self._save_state_best_effort()

        payload = {
            "module": "RiskEngine",
            "timestamp": utc_iso(now),
            "event": "TRADE_CLOSED",
            "symbol": trade_result.symbol,
            "pnl_usdt": float(trade_result.pnl_usdt),
            "metrics": {
                "equity_day_start": self.circuit_breaker.equity_day_start,
                "equity_now": float(equity_now),
                "current_drawdown_pct": self.circuit_breaker.current_drawdown_pct(float(equity_now), now),
                "consecutive_losses": self.circuit_breaker.consecutive_losses,
                "state": self.circuit_breaker.state.value,
                "cooldown_until": utc_iso(self.circuit_breaker.cooldown_until) if self.circuit_breaker.cooldown_until else None,
            },
        }
        log_json(self.logger, payload, level=logging.INFO)

    def _load_state_best_effort(self) -> None:
        try:
            p = Path(str(self.state_path))
            if not p.exists():
                return
            obj = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                return
            self.circuit_breaker = CircuitBreaker.from_json(obj)
        except Exception:
            # Do not block trading because state couldn't be loaded.
            return

    def _save_state_best_effort(self) -> None:
        try:
            p = Path(str(self.state_path))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self.circuit_breaker.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            return

    def _calculate_position_notional(
        self,
        equity_now: float,
        entry_price: float,
        stop_loss: float,
        side: Side,
        leverage: float,
    ) -> float:
        """Compute position notional in USDT using simple risk-based sizing.

        - Risk budget = equity * risk_per_trade_pct
        - Loss per 1 USDT notional ~= |entry-stop|/entry
        - notional = risk_budget / sl_pct

        This avoids contract-specific qty math; order layer can convert notional to qty.
        """

        equity_now = max(float(equity_now), 0.0)
        if equity_now <= 0:
            return 0.0

        entry = max(float(entry_price), 1e-12)
        sl = float(stop_loss)
        sl_pct = abs(entry - sl) / entry
        sl_pct = max(sl_pct, 1e-6)

        risk_budget = equity_now * (float(self.risk_per_trade_pct) / 100.0)
        notional = risk_budget / sl_pct

        # Clamp by max notional fraction of equity (keeps things sane under tiny ATR)
        max_notional = equity_now * (float(self.max_position_notional_pct) / 100.0)
        notional = min(notional, max_notional)

        # Leverage does not increase risk budget; it affects margin, which order layer can handle.
        # Still, keep leverage within bounds.
        _ = leverage
        _ = side

        return max(float(notional), 0.0)
