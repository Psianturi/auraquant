from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, Protocol

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

    def estimate_required_margin(
        self,
        *,
        symbol: str,
        entry_price: float,
        notional_usdt: float,
        leverage: Optional[float] = None,
    ) -> float:

        _ = symbol
        _ = entry_price

        lev = float(leverage or 0.0)
        if lev <= 0:
            lev = 1.0
        return float(max(float(notional_usdt) / lev, 0.0))

    def debug_order_sizing(
        self,
        *,
        symbol: str,
        entry_price: float,
        notional_usdt: float,
        leverage: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Optional structured debug info for how an execution backend sizes orders.

        Live exchanges may override to provide min lot/step, computed qty, etc.
        Orchestrator uses this only for enriched logging.
        """

        _ = symbol
        _ = entry_price
        _ = notional_usdt
        _ = leverage
        return None
