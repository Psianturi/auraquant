from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

Side = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class TradeIntent:
    """A request to open a new position, produced by strategy layers.

    """

    symbol: str
    side: Side
    entry_price: float
    atr: float
    confidence: float = 1.0
    requested_leverage: float = 10.0
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())


@dataclass(frozen=True)
class RiskDecision:
    """Result from RiskEngine.validate_intent()."""

    allowed: bool
    decision: Literal["APPROVED", "DENIED"]
    reason: str

    symbol: str
    side: Side
    entry_price: float

    position_notional_usdt: Optional[float] = None
    leverage: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    evidence_json: Optional[dict] = None


@dataclass(frozen=True)
class TradeResult:
    """Outcome of a completed trade."""

    symbol: str
    pnl_usdt: float
    closed_at: datetime = field(default_factory=lambda: datetime.utcnow())
    order_id: Optional[str] = None  # order_id for AI log

    @property
    def is_win(self) -> bool:
        return self.pnl_usdt > 0
