from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class BotPhase(str, Enum):
    SCAN = "SCAN"
    QUALIFY = "QUALIFY"
    ENTER = "ENTER"
    MANAGE = "MANAGE"
    EXIT = "EXIT"
    RECONCILE = "RECONCILE"


@dataclass
class OrchestratorConfig:
    symbol: str = "SOL/USDT"
    tick_seconds: int = 60

    lead_symbol: str = "BTC/USDT"
    correlation_window: int = 30
    correlation_threshold: float = 0.25
    correlation_max_lag: int = 3

    min_confidence: float = 0.6
    min_atr: float = 0.0

    # Anti-spam guard (minimum time between entries for the same symbol)
    min_entry_interval_seconds: int = 300

    default_leverage: float = 10.0

    enforce_weex_allowlist: bool = True


@dataclass
class TickContext:
    now: datetime
    symbol: str
    last_price: float
    atr: float


@dataclass
class PositionSnapshot:
    symbol: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    notional_usdt: float
    is_open: bool
    opened_at: datetime


@dataclass
class ReconcileSnapshot:
    equity_start: float
    equity_now: float
    pnl_total: float
    trade_count: int
    positions_opened: int
    trades_closed: int
    position: Optional[PositionSnapshot]
