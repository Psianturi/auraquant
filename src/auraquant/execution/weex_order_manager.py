from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import requests

from ..risk.types import Side, TradeResult
from ..weex.private_client import WeexPrivateRestClient
from ..weex.symbols import to_weex_contract_symbol
from .base_order_manager import BaseOrderManager


logger = logging.getLogger(__name__)


@dataclass
class WeexLivePosition:
    symbol: str
    side: Side
    entry_price: float
    stop_loss: float
    take_profit: float
    notional_usdt: float
    opened_at: datetime
    is_open: bool = True
    order_id: Optional[str] = None
    size: float = 0.0
    weex_symbol: Optional[str] = None


class WeexOrderManager(BaseOrderManager):
    """Live WEEX execution.

    This replaces PaperOrderManager when you want real account equity and real
    order placement via WEEX contract API.

    Endpoints used (matching the official API test flow):
    - GET  /capi/v2/account/assets
    - POST /capi/v2/account/leverage
    - POST /capi/v2/order/placeOrder

    Notes:
    - MVP supports ONE open position at a time.
    - Orders are MARKET by default.
    - SL/TP are passed as presetStopLossPrice / presetTakeProfitPrice.
    """

    def __init__(
        self,
        client: Optional[WeexPrivateRestClient] = None,
        execute_orders: Optional[bool] = None,
        default_leverage: Optional[int] = None,
        margin_mode: int = 1,
    ) -> None:
        self.client = client or WeexPrivateRestClient()
        self.client.require_env()

        if execute_orders is None:
            execute_orders = os.getenv("WEEX_EXECUTE_ORDER", "0") == "1"
        self.execute_orders = bool(execute_orders)

        if default_leverage is None:
            try:
                default_leverage = int(os.getenv("WEEX_LEVERAGE", "2"))
            except Exception:
                default_leverage = 2
        self.default_leverage = max(1, min(int(default_leverage), 20))
        self.margin_mode = int(margin_mode)

        # If True, send preset SL/TP to exchange on entry. Default False to avoid
        # double-closing (bot-managed close vs exchange-triggered SL/TP).
        self.use_preset_sltp = os.getenv("WEEX_USE_PRESET_SLTP", "0") == "1"

        self._starting_equity: Optional[float] = None
        self._equity: float = 0.0
        self._position: Optional[WeexLivePosition] = None
        self._positions_opened: int = 0
        self._trades_closed: int = 0
        # Cache: {weex_symbol: (min_order_size, step_size)}
        self._contract_rules_cache: Dict[str, Tuple[Optional[float], float]] = {}

        self.reconcile(now=datetime.utcnow())

    def starting_equity(self) -> float:
        return float(self._starting_equity if self._starting_equity is not None else self._equity)

    def equity(self) -> float:
        return float(self._equity)

    def trade_count(self) -> int:
        return int(self._trades_closed)

    def positions_opened(self) -> int:
        return int(self._positions_opened)

    def trades_closed(self) -> int:
        return int(self._trades_closed)

    def position(self) -> Optional[WeexLivePosition]:
        if self._position and self._position.is_open:
            return self._position
        return None

    def set_leverage(self, symbol: str, leverage: int) -> None:
        lev = max(1, min(int(leverage), 20))
        weex_symbol = self._resolve_weex_symbol(symbol)
        body = {
            "symbol": weex_symbol,
            "marginMode": int(self.margin_mode),
            "longLeverage": str(lev),
            "shortLeverage": str(lev),
        }
        resp = self.client.signed_post("/capi/v2/account/leverage", body)
        if resp.status_code != 200:
            raise RuntimeError(f"WEEX leverage set failed HTTP {resp.status_code}: {resp.text[:200]}")

    def open_position(
        self,
        symbol: str,
        side: Side,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        notional_usdt: float,
        now: datetime,
    ) -> WeexLivePosition:
        if self.position() is not None:
            raise RuntimeError("WeexOrderManager MVP supports only one open position at a time")

        if not self.execute_orders:
            raise RuntimeError("Live execution disabled. Set WEEX_EXECUTE_ORDER=1 to place real orders.")

        weex_symbol = self._resolve_weex_symbol(symbol)
        self.set_leverage(symbol=symbol, leverage=self.default_leverage)

        size = self._compute_order_size(weex_symbol=weex_symbol, notional_usdt=float(notional_usdt), price=float(entry_price))

        type_code = "1" if side == "LONG" else "2"  # 1 open long, 2 open short
        body = {
            "symbol": weex_symbol,
            "client_oid": f"auraquant_{int(time.time() * 1000)}",
            "size": self._format_size(size),
            "type": type_code,
            "order_type": "0",
            "match_price": "1",
            "price": str(float(entry_price)),
            "marginMode": int(self.margin_mode),
        }

        if self.use_preset_sltp:
            body["presetTakeProfitPrice"] = str(float(take_profit))
            body["presetStopLossPrice"] = str(float(stop_loss))

        resp = self.client.signed_post("/capi/v2/order/placeOrder", body)
        if resp.status_code != 200:
            raise RuntimeError(f"WEEX placeOrder failed HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            data = resp.json()
        except Exception:
            data = {}

        order_id = data.get("order_id") or data.get("orderId")
        pos = WeexLivePosition(
            symbol=symbol,
            side=side,
            entry_price=float(entry_price),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            notional_usdt=float(notional_usdt),
            opened_at=now,
            order_id=str(order_id) if order_id is not None else None,
            size=float(size),
            weex_symbol=weex_symbol,
        )
        self._position = pos
        self._positions_opened += 1
        return pos

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
        equity_before = float(self._equity)

        # Important safety rule: only mark the position closed if the exchange
        # close order succeeds.
        self._close_position_market(pos=pos)

        try:
            self.reconcile(now=now)
            pnl = float(self._equity - equity_before)
        except Exception:
            pnl = self._pnl_proxy(pos, float(exit_price))

        pos.is_open = False
        self._trades_closed += 1
        return TradeResult(symbol=pos.symbol, pnl_usdt=float(pnl), closed_at=now, order_id=pos.order_id)

    def close_open_position_best_effort(self) -> bool:
        """Attempt to close the currently tracked open position.

        Returns True if a close order was sent successfully.
        Does nothing (and returns False) if there is no open position.
        """

        pos = self.position()
        if pos is None:
            return False

        try:
            self._close_position_market(pos=pos)
            return True
        except RuntimeError as exc:
            msg = str(exc)
            # If exchange reports position side invalid, it usually means the position

            if "40015" in msg or "position side invalid" in msg.lower():
                logger.warning("[WEEX] Close rejected (position side invalid). Syncing with WEEX...")
                try:
                    self._sync_position_from_weex()
                except Exception as sync_err:
                    logger.error(f"[WEEX] Position sync failed: {sync_err}")
                # Only clear local state if sync confirms no open position
                if self._position is None or not self._position.is_open:
                    logger.info("[WEEX] Sync confirmed: no open position on exchange.")
                    return False
                else:
                    logger.warning("[WEEX] Sync shows position still open. Manual intervention may be needed.")
                    return False
            raise

    def _sync_position_from_weex(self) -> None:

        try:
            resp = self.client.signed_get("/capi/v2/position/allPosition")
            if resp.status_code != 200:
                logger.warning(f"[WEEX] Position query failed HTTP {resp.status_code}")
                return
            
            data = resp.json()
            positions = data.get("data", []) if isinstance(data, dict) else data
            
            if not positions or not isinstance(positions, list):
                if self._position is not None:
                    logger.info("[WEEX] No positions on exchange. Clearing local state.")
                    self._position.is_open = False
                    self._position = None
                return
            
            # Check if any position is actually open (has size > 0)
            has_open = False
            for pos in positions:
                if not isinstance(pos, dict):
                    continue
                size = pos.get("total") or pos.get("available") or pos.get("size") or 0
                try:
                    if float(size) > 0:
                        has_open = True
                        break
                except (ValueError, TypeError):
                    continue
            
            if not has_open:
                if self._position is not None:
                    logger.info("[WEEX] No active position found. Clearing local state.")
                    self._position.is_open = False
                    self._position = None
            else:
                logger.info("[WEEX] Active position still exists on exchange.")
                
        except Exception as e:
            logger.error(f"[WEEX] Position sync error: {e}")

    def reconcile(self, now: Optional[datetime] = None) -> None:
        _ = now
        resp = self.client.signed_get("/capi/v2/account/assets")
        if resp.status_code != 200:
            raise RuntimeError(f"WEEX assets failed HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError("WEEX assets returned non-JSON") from e

        equity = self._parse_usdt_equity(data)
        if equity is None:
            return

        self._equity = float(equity)
        if self._starting_equity is None:
            self._starting_equity = float(equity)

    def available_margin(self) -> float:
        """Get available margin for new positions.

        Returns available USDT that can be used to open new positions.
        This accounts for margin already locked in open positions.
        """
        resp = self.client.signed_get("/capi/v2/account/assets")
        if resp.status_code != 200:
            logger.warning(f"WEEX assets check failed HTTP {resp.status_code}")
            return 0.0

        try:
            data = resp.json()
        except Exception:
            return 0.0

        available = self._parse_available_margin(data)
        return float(available) if available is not None else 0.0

    def _parse_available_margin(self, data: object) -> Optional[float]:
        if isinstance(data, dict):
            payload = data.get("data") if "data" in data else data
        else:
            payload = data

        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                currency = (item.get("currency") or item.get("coin") or item.get("asset") or "").upper()
                if currency and currency != "USDT":
                    continue
                for k in ("available", "availableMargin", "availableBalance", "free"):
                    v = item.get(k)
                    if isinstance(v, (int, float)):
                        return float(v)
                    if isinstance(v, str):
                        try:
                            return float(v)
                        except Exception:
                            continue
        return None

    def _resolve_weex_symbol(self, symbol: str) -> str:
        pinned = os.getenv("WEEX_PRODUCT_SYMBOL")
        if pinned:
            return str(pinned)
        return to_weex_contract_symbol(symbol)

    def _close_position_market(self, pos: WeexLivePosition) -> None:
        if not self.execute_orders:
            raise RuntimeError("Live execution disabled; cannot close position")

        weex_symbol = pos.weex_symbol or self._resolve_weex_symbol(pos.symbol)
        type_code = "3" if pos.side == "LONG" else "4"  # 3 close long, 4 close short
        body = {
            "symbol": weex_symbol,
            "client_oid": f"auraquant_close_{int(time.time() * 1000)}",
            "size": self._format_size(float(pos.size)),
            "type": type_code,
            "order_type": "0",
            "match_price": "1",
            "price": str(float(pos.entry_price)),
            "marginMode": int(self.margin_mode),
        }

        resp = self.client.signed_post("/capi/v2/order/placeOrder", body)
        if resp.status_code != 200:
            raise RuntimeError(f"WEEX close order failed HTTP {resp.status_code}: {resp.text[:200]}")

    def _get_contract_rules(self, weex_symbol: str) -> Tuple[Optional[float], float]:
        cached = self._contract_rules_cache.get(weex_symbol)
        if cached is not None:
            return cached

        base_url = os.getenv("WEEX_BASE_URL", "https://api-contract.weex.com").rstrip("/")
        url = f"{base_url}/capi/v2/market/contracts"
        resp = requests.get(url, params={"symbol": weex_symbol}, timeout=15)
        if resp.status_code != 200:
            self._contract_rules_cache[weex_symbol] = (None, 1.0)
            return self._contract_rules_cache[weex_symbol]

        try:
            payload = resp.json()
        except Exception:
            self._contract_rules_cache[weex_symbol] = (None, 1.0)
            return self._contract_rules_cache[weex_symbol]

        items = None
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = [data]

        min_order_size: Optional[float] = None
        step_size: Optional[float] = None

        if items and isinstance(items[0], dict):
            first = items[0]
            mos = first.get("minOrderSize")
            if isinstance(mos, (int, float)):
                min_order_size = float(mos)
            elif isinstance(mos, str):
                try:
                    min_order_size = float(mos)
                except Exception:
                    min_order_size = None


            for key in ("stepSize", "step_size", "sizeStep", "size_step", "sizeIncrement", "size_increment"):
                inc = first.get(key)
                if inc is None:
                    continue
                try:
                    candidate = float(inc)
                except Exception:
                    continue

                # Some WEEX contract payloads expose `size_increment=0` while still
                # enforcing a non-1 lot size via minOrderSize (e.g. DOGE=100, ADA=10).
                # Treat non-positive increments as unusable and fall back later.
                if candidate > 0:
                    step_size = candidate
                    break

        if step_size is None or step_size <= 0:
            if isinstance(min_order_size, (int, float)) and float(min_order_size) > 0:
                step_size = float(min_order_size)
            else:
                step_size = 1.0

        self._contract_rules_cache[weex_symbol] = (min_order_size, float(step_size))
        return self._contract_rules_cache[weex_symbol]

    def _compute_order_size(self, weex_symbol: str, notional_usdt: float, price: float) -> float:
        min_order_size, step_size = self._get_contract_rules(weex_symbol)
        price = max(float(price), 1e-12)
        raw = float(notional_usdt) / price

        if isinstance(min_order_size, (int, float)) and raw < float(min_order_size):
            raw = float(min_order_size)

        step = float(step_size) if float(step_size) > 0 else 1.0
        # Quantize down to the allowed increment (e.g. stepSize=10 => 40, 50, ...)
        floored = math.floor((raw / step) + 1e-12) * step
        # If we floored to zero but a min size exists, bump to the smallest valid step >= min.
        if floored <= 0 and isinstance(min_order_size, (int, float)) and float(min_order_size) > 0:
            floored = math.ceil(float(min_order_size) / step) * step

        return float(max(floored, 0.0))

    def _format_size(self, size: float) -> str:
        s = f"{float(size):.8f}".rstrip("0").rstrip(".")
        return s if s else "0"

    def _pnl_proxy(self, pos: WeexLivePosition, exit_price: float) -> float:
        entry = max(float(pos.entry_price), 1e-12)
        ret = (float(exit_price) - entry) / entry
        if pos.side == "SHORT":
            ret = -ret
        return float(pos.notional_usdt) * float(ret)

    def _parse_usdt_equity(self, data: object) -> Optional[float]:
        if isinstance(data, dict):
            payload = data.get("data") if "data" in data else data
        else:
            payload = data

        if isinstance(payload, list):
            return self._parse_usdt_equity_from_list(payload)

        if isinstance(payload, dict):
            for key in ("assets", "balances", "account", "detail"):
                v = payload.get(key)
                if isinstance(v, list):
                    eq = self._parse_usdt_equity_from_list(v)
                    if eq is not None:
                        return eq

            for k in ("equity", "available", "availableBalance", "balance"):
                v = payload.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, str):
                    try:
                        return float(v)
                    except Exception:
                        pass

        return None

    def _parse_usdt_equity_from_list(self, items: list) -> Optional[float]:
        for item in items:
            if not isinstance(item, dict):
                continue
            currency = (item.get("currency") or item.get("coin") or item.get("asset") or "").upper()
            if currency and currency != "USDT":
                continue
            for k in ("equity", "available", "availableBalance", "balance", "total"):
                v = item.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, str):
                    try:
                        return float(v)
                    except Exception:
                        continue
        return None
