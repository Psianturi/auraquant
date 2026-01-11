from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..risk.types import Side, TradeResult
from .base_order_manager import BaseOrderManager, PositionLike


@dataclass
class WeexPosition:
    symbol: str
    side: Side
    entry_price: float
    stop_loss: float
    take_profit: float
    notional_usdt: float
    opened_at: datetime
    is_open: bool = True
    order_id: Optional[str] = None


class WeexOrderManager(BaseOrderManager):

    def __init__(self, starting_equity: float = 1000.0):
        self._starting_equity = float(starting_equity)

    def starting_equity(self) -> float:
        return float(self._starting_equity)

    def equity(self) -> float:
        raise NotImplementedError("WEEX equity query not implemented yet")

    def trade_count(self) -> int:
        raise NotImplementedError("WEEX trade_count not implemented yet")

    def positions_opened(self) -> int:
        raise NotImplementedError("WEEX positions_opened not implemented yet")

    def trades_closed(self) -> int:
        raise NotImplementedError("WEEX trades_closed not implemented yet")

    def open_position(
        self,
        symbol: str,
        side: Side,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        notional_usdt: float,
        now: datetime,
    ) -> PositionLike:
        raise NotImplementedError("WEEX order placement not implemented yet")

    def position(self) -> Optional[PositionLike]:
        raise NotImplementedError("WEEX position query not implemented yet")

    def on_price_tick(self, symbol: str, price: float, now: datetime) -> Optional[TradeResult]:
        # Live execution should rely on exchange order updates, not local price ticks.
        return None

    def reconcile(self, now: Optional[datetime] = None) -> None:
        raise NotImplementedError("WEEX reconcile not implemented yet")
