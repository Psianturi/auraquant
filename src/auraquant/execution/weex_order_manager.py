from __future__ import annotations

import logging
import math
import os
import re
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
    # Trailing stop state
    original_stop_loss: Optional[float] = None
    breakeven_activated: bool = False  # True once SL moved to breakeven
    trailing_activated: bool = False  # True once trailing mode active
    highest_price: Optional[float] = None  # Track best price for trailing (LONG)
    lowest_price: Optional[float] = None  # Track best price for trailing (SHORT)


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
                default_leverage = int(os.getenv("WEEX_LEVERAGE", "10"))
            except Exception:
                default_leverage = 10
        self.default_leverage = max(1, min(int(default_leverage), 20))
        
        try:
            self.breakeven_trigger_pct = float(os.getenv("BREAKEVEN_TRIGGER_PCT", "0.005"))  # 0.5%
        except Exception:
            self.breakeven_trigger_pct = 0.005
        try:
            self.trailing_trigger_pct = float(os.getenv("TRAILING_TRIGGER_PCT", "0.008"))  # 0.8%
        except Exception:
            self.trailing_trigger_pct = 0.008
        try:
            self.trailing_distance_pct = float(os.getenv("TRAILING_DISTANCE_PCT", "0.004"))  # 0.4%
        except Exception:
            self.trailing_distance_pct = 0.004
        self.margin_mode = int(margin_mode)

        # Optional: send SL/TP to exchange so they trigger server-side.
        self.use_preset_sltp = os.getenv("WEEX_USE_PRESET_SLTP", "0") == "1"
        self._preset_sltp_supported: Optional[bool] = None

        self._starting_equity: Optional[float] = None
        self._equity: float = 0.0
        self._position: Optional[WeexLivePosition] = None
        self._positions_opened: int = 0
        self._trades_closed: int = 0
        # Cache: {weex_symbol: (min_order_size, step_size)}
        self._contract_rules_cache: Dict[str, Tuple[Optional[float], float]] = {}
        # Track last close attempt to prevent spam
        self._last_close_attempt: float = 0.0

        try:
            self._close_retry_cooldown = float(os.getenv("WEEX_CLOSE_RETRY_COOLDOWN_SECONDS", "60"))
        except Exception:
            self._close_retry_cooldown = 60.0

        self._position_sync_block_until: float = 0.0
        self._position_sync_block_seconds: float = 60.0

        # If the assets endpoint is flaky (e.g. HTTP 5xx/521), avoid spamming it.
        self._assets_sync_block_until: float = 0.0
        self._assets_sync_backoff_seconds: float = 2.0
        try:
            self._assets_sync_backoff_max_seconds = float(os.getenv("WEEX_ASSETS_BACKOFF_MAX_SECONDS", "60"))
        except Exception:
            self._assets_sync_backoff_max_seconds = 60.0

        self._cleanup_stale_orders_on_start()
        
        self.reconcile(now=datetime.utcnow())
    
    def _cleanup_stale_orders_on_start(self) -> None:
        """Cancel all pending orders on startup to free locked margin."""
        cleanup_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LTC/USDT", "BNB/USDT"]
        logger.info("[WEEX] 🧹 Cleaning up stale orders on startup...")
        for symbol in cleanup_symbols:
            try:
                weex_symbol = self._resolve_weex_symbol(symbol)
                self._cancel_all_orders(weex_symbol)
            except Exception as e:
                logger.debug(f"[WEEX] Cleanup {symbol} skipped: {e}")

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

        attempted_preset = False
        if self.use_preset_sltp and self._preset_sltp_supported is not False:
            attempted_preset = True
            body["presetTakeProfitPrice"] = self._format_price(take_profit, symbol)
            body["presetStopLossPrice"] = self._format_price(stop_loss, symbol)

        resp = self.client.signed_post("/capi/v2/order/placeOrder", body)
        if resp.status_code != 200:
            msg = (resp.text or "")[:500]
            if attempted_preset and self._looks_like_open_sltp_invalid(msg):
                self._preset_sltp_supported = False
                logger.warning("[WEEX] placeOrder rejected preset SL/TP; retrying without preset SL/TP (auto-disable).")
                body.pop("presetTakeProfitPrice", None)
                body.pop("presetStopLossPrice", None)
                resp = self.client.signed_post("/capi/v2/order/placeOrder", body)

            if resp.status_code != 200:
                raise RuntimeError(f"WEEX placeOrder failed HTTP {resp.status_code}: {(resp.text or '')[:200]}")

        if attempted_preset and self._preset_sltp_supported is None:
            self._preset_sltp_supported = True

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
            # Initialize trailing stop state
            original_stop_loss=float(stop_loss),
            breakeven_activated=False,
            trailing_activated=False,
            highest_price=float(entry_price) if side == "LONG" else None,
            lowest_price=float(entry_price) if side == "SHORT" else None,
        )
        self._position = pos
        self._positions_opened += 1
        return pos

    @staticmethod
    def _looks_like_open_sltp_invalid(msg: str) -> bool:
        """Heuristic for WEEX rejecting preset SL/TP params on order open."""
        s = (msg or "").lower()
        if "invalid_argument" in s and ("tp" in s or "take" in s or "stop" in s):
            return True
        if "order open" in s and ("tp" in s or "take" in s or "stop" in s):
            return True

        if "\"code\":\"40015\"" in s and "invalid" in s:
            return True
        return False

    def on_price_tick(self, symbol: str, price: float, now: datetime) -> Optional[TradeResult]:
        pos = self.position()
        if not pos or pos.symbol != symbol:
            return None

        price = float(price)
        
        self._update_trailing_stop(pos, price)
        
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

        # Use safe close method that doesn't raise on failure.
        close_result = self._safe_close_position(pos=pos, reason="SL/TP hit", price_hint=float(exit_price))
        if close_result is None:
            return None
        if close_result is False:
            logger.warning(f"[WEEX] Close attempt failed for {symbol}, will retry")
            return None

        try:
            self.reconcile(now=now)
            pnl = float(self._equity - equity_before)
        except Exception:
            pnl = self._pnl_proxy(pos, float(exit_price))

        pos.is_open = False
        self._trades_closed += 1
        return TradeResult(symbol=pos.symbol, pnl_usdt=float(pnl), closed_at=now, order_id=pos.order_id)

    def _update_trailing_stop(self, pos: WeexLivePosition, current_price: float) -> None:
      
        entry = pos.entry_price
        current_price = float(current_price)
        
        if pos.side == "LONG":
            # Track highest price
            if pos.highest_price is None or current_price > pos.highest_price:
                pos.highest_price = current_price
            
            price_move_pct = (current_price - entry) / entry
            best_move_pct = (pos.highest_price - entry) / entry if pos.highest_price else 0.0
            
            # Step 1: Breakeven activation (0.5% profit)
            if not pos.breakeven_activated and price_move_pct >= self.breakeven_trigger_pct:
                # Move SL to breakeven (entry price + small buffer for fees ~0.05%)
                breakeven_sl = entry * 1.0005  # Tiny buffer above entry
                if breakeven_sl > pos.stop_loss:
                    logger.info(
                        f"[TRAILING] {pos.symbol} LONG: Breakeven activated @ {current_price:.4f} "
                        f"(moved +{price_move_pct*100:.2f}%). SL: {pos.stop_loss:.4f} -> {breakeven_sl:.4f}"
                    )
                    pos.stop_loss = breakeven_sl
                    pos.breakeven_activated = True
            
            # Step 2: Trailing activation (0.8% profit)
            if not pos.trailing_activated and best_move_pct >= self.trailing_trigger_pct:
                logger.info(
                    f"[TRAILING] {pos.symbol} LONG: Trailing mode activated @ best={pos.highest_price:.4f} "
                    f"(moved +{best_move_pct*100:.2f}%)"
                )
                pos.trailing_activated = True
            
            # Step 3: Trail the stop loss
            if pos.trailing_activated and pos.highest_price:
                # New SL = highest price - trailing distance
                new_sl = pos.highest_price * (1 - self.trailing_distance_pct)
                if new_sl > pos.stop_loss:
                    logger.info(
                        f"[TRAILING] {pos.symbol} LONG: Trailing SL updated. "
                        f"Best={pos.highest_price:.4f}, SL: {pos.stop_loss:.4f} -> {new_sl:.4f}"
                    )
                    pos.stop_loss = new_sl
        
        else:  # SHORT
            # Track lowest price
            if pos.lowest_price is None or current_price < pos.lowest_price:
                pos.lowest_price = current_price
            
            price_move_pct = (entry - current_price) / entry
            best_move_pct = (entry - pos.lowest_price) / entry if pos.lowest_price else 0.0
            
            # Step 1: Breakeven activation (0.5% profit)
            if not pos.breakeven_activated and price_move_pct >= self.breakeven_trigger_pct:
                breakeven_sl = entry * 0.9995  # Tiny buffer below entry
                if breakeven_sl < pos.stop_loss:
                    logger.info(
                        f"[TRAILING] {pos.symbol} SHORT: Breakeven activated @ {current_price:.4f} "
                        f"(moved +{price_move_pct*100:.2f}%). SL: {pos.stop_loss:.4f} -> {breakeven_sl:.4f}"
                    )
                    pos.stop_loss = breakeven_sl
                    pos.breakeven_activated = True
            
            # Step 2: Trailing activation (0.8% profit)
            if not pos.trailing_activated and best_move_pct >= self.trailing_trigger_pct:
                logger.info(
                    f"[TRAILING] {pos.symbol} SHORT: Trailing mode activated @ best={pos.lowest_price:.4f} "
                    f"(moved +{best_move_pct*100:.2f}%)"
                )
                pos.trailing_activated = True
            
            # Step 3: Trail the stop loss
            if pos.trailing_activated and pos.lowest_price:
                # New SL = lowest price + trailing distance
                new_sl = pos.lowest_price * (1 + self.trailing_distance_pct)
                if new_sl < pos.stop_loss:
                    logger.info(
                        f"[TRAILING] {pos.symbol} SHORT: Trailing SL updated. "
                        f"Best={pos.lowest_price:.4f}, SL: {pos.stop_loss:.4f} -> {new_sl:.4f}"
                    )
                    pos.stop_loss = new_sl

    def close_open_position_best_effort(self) -> bool:
        """Attempt to close the currently tracked open position.

        Returns True if a close order was sent successfully.
        Does nothing (and returns False) if there is no open position.
        """
        pos = self.position()
        if pos is None:
            return False
        
        return bool(self._safe_close_position(pos=pos, reason="best_effort"))

    def _safe_close_position(
        self,
        pos: WeexLivePosition,
        reason: str = "",
        price_hint: Optional[float] = None,
    ) -> Optional[bool]:

        now_ts = time.time()
        if now_ts < self._position_sync_block_until:
            logger.warning(f"[WEEX] Position sync degraded; skipping close attempt ({reason})")
            return None

        # Check cooldown to prevent close spam
        if now_ts - self._last_close_attempt < self._close_retry_cooldown:
            logger.debug(f"[WEEX] Close cooldown active, skipping ({reason})")
            return None
        self._last_close_attempt = now_ts
        
        weex_symbol = pos.weex_symbol or self._resolve_weex_symbol(pos.symbol)
        
        # Step 1: Cancel any pending orders first (root cause of 40015)
        try:
            self._cancel_all_orders(weex_symbol)
        except Exception as e:
            logger.warning(f"[WEEX] Cancel orders failed (non-fatal): {e}")
        
        # Step 2: Attempt close
        try:
            self._close_position_market(pos=pos, price_hint=price_hint)

            sync = self._sync_position_from_weex(weex_symbol=weex_symbol)
            if sync is True:
                logger.info(f"[WEEX] Position closed successfully ({reason})")
                return True
            if sync is False:
                logger.warning(f"[WEEX] Close submitted but position still open; will retry ({reason})")
                return False
            logger.warning(f"[WEEX] Close submitted but position status unknown; will retry ({reason})")
            return False
        except RuntimeError as exc:
            msg = str(exc)
            
            # Handle 40015: position side invalid / pending order conflict
            if "40015" in msg or "position side invalid" in msg.lower():
                logger.warning(f"[WEEX] Close rejected 40015 ({reason}). Details: {msg[:200]}")

                blocking_order_id: Optional[str] = None
                m = re.search(r"\border\s+(\d+)", msg)
                if m:
                    blocking_order_id = m.group(1)

                if blocking_order_id:
                    try:
                        self._cancel_order_by_id(weex_symbol=weex_symbol, order_id=blocking_order_id)
                    except Exception as cancel_err:
                        logger.warning(f"[WEEX] Failed to cancel blocking order {blocking_order_id}: {cancel_err}")

                    # Retry close once after cancelling the blocking order.
                    try:
                        self._close_position_market(pos=pos, price_hint=price_hint)
                    except Exception as retry_err:
                        logger.warning(f"[WEEX] Close retry after cancel failed ({reason}): {str(retry_err)[:200]}")

                try:
                    sync = self._sync_position_from_weex(weex_symbol=weex_symbol)
                except Exception as sync_err:
                    logger.warning(f"[WEEX] Sync failed (will retry): {sync_err}")
                    sync = None
                
                if sync is True or self._position is None or not self._position.is_open:
                    logger.info("[WEEX] Sync shows position closed.")
                    return True 
                else:
                    logger.warning("[WEEX] Position still open, will retry next tick.")
                    return False
            
            # Handle other errors gracefully
            logger.error(f"[WEEX] Close error ({reason}): {msg[:150]}")
            return False

        except Exception as exc:
            logger.error(f"[WEEX] Close exception ({reason}): {exc}")
            return False
    
    def _cancel_all_orders(self, weex_symbol: str) -> None:
        """Cancel all pending orders for a symbol to prevent 40015 conflicts."""
        body = {"symbol": weex_symbol}
        try:
            resp = self.client.signed_post("/capi/v2/order/cancelAll", body)
            if resp.status_code == 200:
                logger.info(f"[WEEX] Cancelled pending orders for {weex_symbol}")
            else:
                # 404 or no orders is fine
                logger.debug(f"[WEEX] Cancel orders response: {resp.status_code}")
        except Exception as e:
            logger.debug(f"[WEEX] Cancel orders exception: {e}")

    def _cancel_order_by_id(self, weex_symbol: str, order_id: str) -> None:
        """Best-effort cancel for a single blocking order.

        WEEX error 40015 sometimes includes an order id that must be cancelled
        before a close is allowed.
        """

        order_id = str(order_id).strip()
        if not order_id:
            return

        endpoints = (
            "/capi/v2/order/cancelOrder",
            "/capi/v2/order/cancel",
            "/capi/v2/order/cancelById",
        )
        payloads = (
            {"symbol": weex_symbol, "orderId": order_id},
            {"symbol": weex_symbol, "order_id": order_id},
            {"orderId": order_id},
            {"order_id": order_id},
        )

        for ep in endpoints:
            for body in payloads:
                try:
                    resp = self.client.signed_post(ep, body)
                except Exception as e:
                    logger.debug(f"[WEEX] Cancel order {order_id} failed calling {ep}: {e}")
                    continue

                if resp.status_code == 200:
                    logger.info(f"[WEEX] Cancelled blocking order {order_id} via {ep}")
                    return

                logger.debug(
                    f"[WEEX] Cancel order {order_id} via {ep} returned HTTP {resp.status_code}: {resp.text[:120]}"
                )

    def _sync_position_from_weex(self, weex_symbol: Optional[str] = None) -> Optional[bool]:
        """Sync local position state from WEEX.

        Returns:
          - True  => confirmed no open position for this symbol
          - False => confirmed an open position still exists
          - None  => unable to confirm (transient API / parse failure)
        """

    
        mode = os.getenv("WEEX_POSITION_SYNC_MODE", "auto").strip().lower()
        if weex_symbol is None and self._position is not None:
            weex_symbol = self._position.weex_symbol

        def _parse_positions(payload: object) -> Optional[list]:
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                data_val = payload.get("data")
                if isinstance(data_val, list):
                    return data_val
                if any(k in payload for k in ("symbol", "side", "size", "total", "position")):
                    return [payload]
            return None

        def _signed_get_first_ok(paths: tuple[str, ...], query: str = ""):
            last_resp = None
            for p in paths:
                try:
                    resp = self.client.signed_get(p, query=query)
                except Exception as e:
                    last_resp = e
                    continue
                last_resp = resp
                if getattr(resp, "status_code", None) == 200:
                    return resp
                if resp.status_code in (400, 404):
                    continue
            return last_resp

        try:
            resp = None
            # Try singlePosition first if allowed.
            if mode in ("auto", "single") and weex_symbol:
                single_paths = ("/capi/v2/account/position/singlePosition",)
                query = f"?symbol={weex_symbol}"
                resp = _signed_get_first_ok(single_paths, query=query)

            # Fallback to allPosition if allowed or if single failed.
            if resp is None or not hasattr(resp, "status_code"):
                all_paths = (
                    "/capi/v2/account/position/allPosition", 
                    "/capi/v2/position/allPosition",          
                )
                if mode in ("auto", "all", "single"):
                    resp = _signed_get_first_ok(all_paths)

            if not hasattr(resp, "status_code"):
                logger.warning(f"[WEEX] Position query failed (exception): {resp}")
                self._position_sync_block_until = time.time() + float(self._position_sync_block_seconds)
                return None

            if resp.status_code != 200:
                logger.warning(f"[WEEX] Position query failed HTTP {resp.status_code}: {resp.text[:200]}")
                if resp.status_code >= 500 or resp.status_code in (429, 408):
                    self._position_sync_block_until = time.time() + float(self._position_sync_block_seconds)
                return None

            data = resp.json()
            positions = _parse_positions(data)
            if positions is None:
                return None

            def _matches_symbol(p: dict) -> bool:
                if not weex_symbol:
                    return True
                sym = p.get("symbol") or p.get("contractCode") or p.get("instrument") or p.get("instrumentId")
                if sym is None:
                    return False
                return str(sym).strip().lower() == str(weex_symbol).strip().lower()

            def _position_size(p: dict) -> float:
                for k in ("total", "available", "size", "holdVol", "pos", "position"):
                    v = p.get(k)
                    if v is None:
                        continue
                    try:
                        return float(v)
                    except Exception:
                        continue
                return 0.0

            relevant = [p for p in positions if isinstance(p, dict) and _matches_symbol(p)]
            has_open = any(_position_size(p) > 0 for p in relevant) if relevant else False

            if not has_open:
                if self._position is not None:
                    logger.info("[WEEX] No active position found. Clearing local state and incrementing trades_closed.")
                    self._position.is_open = False
                    self._trades_closed += 1  # Increment when position is confirmed closed
                    self._position = None
                return True

            logger.info("[WEEX] Active position still exists on exchange.")
            return False

        except Exception as e:
            logger.error(f"[WEEX] Position sync error: {e}")
            self._position_sync_block_until = time.time() + float(self._position_sync_block_seconds)
            return None

    def reconcile(self, now: Optional[datetime] = None) -> None:
        _ = now
        # Keep startup strict (so init/health checks fail fast), but when the bot
        # is already running, avoid hammering the assets endpoint during exchange
        # instability (e.g. HTTP 521/5xx).
        strict = self._starting_equity is None

        if not strict:
            now_ts = time.time()
            if now_ts < self._assets_sync_block_until:
                return

        resp = self.client.signed_get("/capi/v2/account/assets")
        if resp.status_code != 200:
            if not strict and (resp.status_code >= 500 or resp.status_code in (429, 408)):
                logger.warning(
                    f"[WEEX] Assets query failed HTTP {resp.status_code}: {resp.text[:200]}"
                )
                now_ts = time.time()
                delay = float(max(1.0, self._assets_sync_backoff_seconds))
                self._assets_sync_block_until = now_ts + delay
                self._assets_sync_backoff_seconds = min(
                    float(self._assets_sync_backoff_seconds) * 2.0, float(self._assets_sync_backoff_max_seconds)
                )
                return

            raise RuntimeError(f"WEEX assets failed HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            data = resp.json()
        except Exception as e:
            if not strict:
                logger.warning(f"[WEEX] Assets returned non-JSON; backing off: {e}")
                now_ts = time.time()
                delay = float(max(1.0, self._assets_sync_backoff_seconds))
                self._assets_sync_block_until = now_ts + delay
                self._assets_sync_backoff_seconds = min(
                    float(self._assets_sync_backoff_seconds) * 2.0, float(self._assets_sync_backoff_max_seconds)
                )
                return
            raise RuntimeError("WEEX assets returned non-JSON") from e


        self._assets_sync_block_until = 0.0
        self._assets_sync_backoff_seconds = 2.0

        equity = self._parse_usdt_equity(data)
        if equity is None:
            return

        self._equity = float(equity)
        if self._starting_equity is None:
            self._starting_equity = float(equity)

        # Also sync position state from exchange to detect server-side SL/TP closes
        # This is crucial when WEEX_USE_PRESET_SLTP=1 (exchange handles SL/TP)
        if self._position is not None and self._position.is_open:
            weex_symbol = self._position.weex_symbol
            if weex_symbol:
                try:
                    self._sync_position_from_weex(weex_symbol=weex_symbol)
                except Exception as e:
                    logger.debug(f"[WEEX] Position sync during reconcile failed: {e}")

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

    def _close_position_market(self, pos: WeexLivePosition, price_hint: Optional[float] = None) -> None:
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
            # Prefer a fresh price hint when available (SL/TP exit), otherwise fall back.
            "price": str(float(price_hint) if price_hint is not None else float(pos.entry_price)),
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

    def _format_price(self, price: float, symbol: str = "") -> str:
        """Format price with appropriate decimal places per symbol."""
        p = float(price)
        sym_lower = symbol.lower()
        # Determine decimal places based on price magnitude and symbol
        if "btc" in sym_lower or p >= 1000:
            # BTC, BNB: 1-2 decimals
            decimals = 2 if p < 10000 else 1
        elif "eth" in sym_lower or p >= 100:
            # ETH, SOL, LTC: 2 decimals
            decimals = 2
        elif p >= 1:
            # XRP: 4 decimals
            decimals = 4
        elif p >= 0.01:
            # DOGE, ADA: 5 decimals
            decimals = 5
        else:
            # Very small prices: 6 decimals
            decimals = 6
        formatted = f"{p:.{decimals}f}"
        return formatted

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
