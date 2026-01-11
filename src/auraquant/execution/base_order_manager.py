from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Protocol

from ..risk.types import Side, TradeResult


class PositionLike(Protocol):
    symbol: str
    side: Side
    entry_price: float
    stop_loss: float
    take_profit: float
    notional_usdt: float
    opened_at: datetime
    is_open: bool
    order_id: Optional[str]


class BaseOrderManager(ABC):
    """Execution interface used by the orchestrator.
    """

    @abstractmethod
    def starting_equity(self) -> float: ...

    @abstractmethod
    def equity(self) -> float: ...

    @abstractmethod
    def trade_count(self) -> int: ...

    @abstractmethod
    def positions_opened(self) -> int: ...

    @abstractmethod
    def trades_closed(self) -> int: ...

    @abstractmethod
    def open_position(
        self,
        symbol: str,
        side: Side,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        notional_usdt: float,
        now: datetime,
    ) -> PositionLike: ...

    @abstractmethod
    def position(self) -> Optional[PositionLike]: ...

    @abstractmethod
    def on_price_tick(self, symbol: str, price: float, now: datetime) -> Optional[TradeResult]: ...

    def reconcile(self, now: Optional[datetime] = None) -> None:
        """Optional: sync local state with exchange state.
        Paper execution is already authoritative; live execution should override.
        """

        return None
