from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..risk.types import Side, TradeResult


@dataclass
class PaperPosition:
    symbol: str
    side: Side
    entry_price: float
    stop_loss: float
    take_profit: float
    notional_usdt: float
    opened_at: datetime
    is_open: bool = True
    order_id: str | None = None  # order_id (paper uses simulated ID)


class PaperOrderManager:
    """Deterministic paper execution.

    Responsibilities:
    - Open a single position (MVP: one at a time)
    - Close when SL/TP is hit
    - Track equity using realized PnL only (simple, judge-friendly)

    This is intentionally minimal; real WEEX execution will replace it.
    """

    def __init__(self, starting_equity: float = 1000.0):
        self._starting_equity = float(starting_equity)
        self._equity = float(starting_equity)
        self._position: Optional[PaperPosition] = None
        self._positions_opened: int = 0
        self._trades_closed: int = 0

    def starting_equity(self) -> float:
        return float(self._starting_equity)

    def equity(self) -> float:
        return float(self._equity)

    def trade_count(self) -> int:
        # Backward-compatible alias: number of CLOSED trades (realized PnL events).
        return int(self._trades_closed)

    def positions_opened(self) -> int:
        return int(self._positions_opened)

    def trades_closed(self) -> int:
        return int(self._trades_closed)

    def open_position(
        self,
        symbol: str,
        side: Side,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        notional_usdt: float,
        now: datetime,
    ) -> PaperPosition:
        if self._position and self._position.is_open:
            raise RuntimeError("PaperOrderManager MVP supports only one open position at a time")

        # Generate simulated order_id for paper trades 
        simulated_order_id = f"paper_{int(now.timestamp() * 1000)}_{self._positions_opened + 1}"

        pos = PaperPosition(
            symbol=symbol,
            side=side,
            entry_price=float(entry_price),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            notional_usdt=float(notional_usdt),
            opened_at=now,
            order_id=simulated_order_id,
        )
        self._position = pos
        self._positions_opened += 1
        return pos

    def position(self) -> Optional[PaperPosition]:
        if self._position and self._position.is_open:
            return self._position
        return None

    def on_price_tick(self, symbol: str, price: float, now: datetime) -> Optional[TradeResult]:
        pos = self.position()
        if not pos or pos.symbol != symbol:
            return None

        price = float(price)

        hit_tp = False
        hit_sl = False
        if pos.side == "LONG":
            hit_tp = price >= pos.take_profit
            hit_sl = price <= pos.stop_loss
        else:
            hit_tp = price <= pos.take_profit
            hit_sl = price >= pos.stop_loss

        if not (hit_tp or hit_sl):
            return None

        exit_price = pos.take_profit if hit_tp else pos.stop_loss
        pnl = self._realized_pnl_usdt(pos, exit_price)

        pos.is_open = False
        self._equity += pnl
        self._trades_closed += 1

        return TradeResult(symbol=pos.symbol, pnl_usdt=float(pnl), closed_at=now, order_id=pos.order_id)

    def _realized_pnl_usdt(self, pos: PaperPosition, exit_price: float) -> float:
        # PnL (USDT) = notional * return
        entry = max(float(pos.entry_price), 1e-12)
        ret = (float(exit_price) - entry) / entry
        if pos.side == "SHORT":
            ret = -ret
        return float(pos.notional_usdt) * float(ret)
