#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
# Allow running from repo root without installing.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from auraquant.weex.private_client import WeexPrivateRestClient
from auraquant.weex.symbols import to_weex_contract_symbol


ALLOWED_SPOT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "XRP/USDT",
    "LTC/USDT",
]


def _signed_get_first_ok(client: WeexPrivateRestClient, paths: Iterable[str], query: str = ""):
    last: Any = None
    for p in paths:
        try:
            resp = client.signed_get(p, query=query)
        except Exception as e:
            last = e
            continue
        last = resp
        if getattr(resp, "status_code", None) == 200:
            return resp
        if resp.status_code in (400, 404):
            continue
    return last


def _public_last_price(base_url: str, weex_symbol: str) -> Optional[float]:
    try:
        url = base_url.rstrip("/") + "/capi/v2/market/ticker"
        resp = requests.get(url, params={"symbol": weex_symbol}, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        last = float(data.get("last") or 0.0)
        return last if last > 0 else None
    except Exception:
        return None


def _parse_positions_payload(payload: object) -> Optional[List[Dict[str, Any]]]:
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        data_val = payload.get("data")
        if isinstance(data_val, list):
            return [p for p in data_val if isinstance(p, dict)]
        # Sometimes the API returns a single object.
        if any(k in payload for k in ("symbol", "side", "size", "total", "position")):
            return [payload]
    return None


def _extract_symbol(p: Dict[str, Any]) -> Optional[str]:
    for k in ("symbol", "contractCode", "instrument", "instrumentId"):
        v = p.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _extract_size(p: Dict[str, Any]) -> float:
    for k in ("total", "available", "size", "holdVol", "pos", "position"):
        v = p.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            continue
    return 0.0


def _extract_side(p: Dict[str, Any]) -> Optional[str]:

    for k in ("holdSide", "posSide", "positionSide", "side", "direction"):
        v = p.get(k)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            v = str(int(v))
        if not isinstance(v, str):
            continue
        s = v.strip().lower()
        if s in ("long", "buy", "1"):
            return "long"
        if s in ("short", "sell", "2"):
            return "short"
    return None


def fetch_open_positions(client: WeexPrivateRestClient) -> List[Tuple[str, str, float]]:
    """Returns list of (weex_symbol, side[long|short], size)."""

    paths = (
        "/capi/v2/account/position/allPosition",
        "/capi/v2/position/allPosition",
    )
    resp = _signed_get_first_ok(client, paths)
    if not hasattr(resp, "status_code"):
        raise RuntimeError(f"Position query failed (exception): {resp}")
    if resp.status_code != 200:
        raise RuntimeError(f"Position query failed HTTP {resp.status_code}: {(resp.text or '')[:250]}")

    positions = _parse_positions_payload(resp.json()) or []
    out: List[Tuple[str, str, float]] = []
    for p in positions:
        sym = _extract_symbol(p)
        if not sym:
            continue
        size = _extract_size(p)
        if size <= 0:
            continue
        side = _extract_side(p)
        if side not in ("long", "short"):
            # If side is missing, we cannot safely close.
            continue
        out.append((sym, side, float(size)))
    return out


def cancel_all_orders(client: WeexPrivateRestClient, weex_symbol: str, execute: bool) -> None:
    if not execute:
        print(f"[DRY] cancelAll symbol={weex_symbol}")
        return
    resp = client.signed_post("/capi/v2/order/cancelAll", {"symbol": weex_symbol})
    if resp.status_code != 200:
        # 404 / no orders is fine; don’t crash the cleanup.
        print(f"[WARN] cancelAll {weex_symbol} -> HTTP {resp.status_code}: {(resp.text or '')[:200]}")


def close_position_market(
    client: WeexPrivateRestClient,
    weex_symbol: str,
    side: str,
    size: float,
    price_hint: float,
    margin_mode: int,
    execute: bool,
) -> None:
    def _format_size(x: float) -> str:
        # Best-effort formatting: trim float artifacts while keeping precision.
        s = f"{float(x):.8f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s

    type_code = "3" if side == "long" else "4"  # 3 close long, 4 close short
    body = {
        "symbol": weex_symbol,
        "client_oid": f"aura_closeall_{int(time.time() * 1000)}",
        "size": _format_size(float(size)),
        "type": type_code,
        "order_type": "0",
        "match_price": "1",
        "price": str(float(price_hint)),
        "marginMode": int(margin_mode),
    }

    if not execute:
        print(f"[DRY] placeOrder(close) symbol={weex_symbol} side={side} size={size} price={price_hint}")
        return

    resp = client.signed_post("/capi/v2/order/placeOrder", body)
    if resp.status_code != 200:
        raise RuntimeError(f"close placeOrder failed HTTP {resp.status_code}: {(resp.text or '')[:250]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="WEEX: close all positions + cancel pending orders")
    ap.add_argument("--execute", action="store_true", help="Actually send cancel/close requests")
    ap.add_argument(
        "--symbols",
        default=",".join(ALLOWED_SPOT_SYMBOLS),
        help="Comma-separated symbols like BTC/USDT,ETH/USDT (defaults to allowed list)",
    )
    ap.add_argument("--margin-mode", type=int, default=int(os.getenv("WEEX_MARGIN_MODE", "1")))
    ap.add_argument("--base-url", default=os.getenv("WEEX_BASE_URL", "https://api-contract.weex.com"))
    ap.add_argument("--sleep", type=float, default=2.0, help="Seconds to sleep between close and recheck")
    args = ap.parse_args()

    # Require explicit env for real action.
    client = WeexPrivateRestClient()
    client.require_env()

    execute = bool(args.execute)
    if execute and os.getenv("WEEX_EXECUTE_ORDER", "0") != "1":
        print("[SAFEGUARD] Refusing to execute: set WEEX_EXECUTE_ORDER=1 and pass --execute")
        return 2

    raw_symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    weex_symbols = [to_weex_contract_symbol(s) for s in raw_symbols]

    print(f"[INFO] execute={execute} margin_mode={args.margin_mode} base_url={args.base_url}")

    # Step A: cancel pending orders for all target symbols.
    for ws in weex_symbols:
        cancel_all_orders(client, ws, execute=execute)

    # Step B: close any open positions (for any symbols returned by API).
    positions = fetch_open_positions(client)
    if not positions:
        print("[INFO] No open positions reported by API.")
    else:
        print("[INFO] Open positions:")
        for sym, side, size in positions:
            print(f"  - {sym} {side} size={size}")

        for sym, side, size in positions:
            # Best-effort: cancel pending orders for that symbol again.
            cancel_all_orders(client, sym, execute=execute)

            price_hint = _public_last_price(args.base_url, sym) or 1.0
            close_position_market(
                client,
                weex_symbol=sym,
                side=side,
                size=size,
                price_hint=price_hint,
                margin_mode=int(args.margin_mode),
                execute=execute,
            )

    time.sleep(float(args.sleep))
    remaining = fetch_open_positions(client)
    if remaining:
        print("[WARN] Remaining open positions after cleanup:")
        for sym, side, size in remaining:
            print(f"  - {sym} {side} size={size}")
        return 1

    print("[OK] No open positions remain (per API).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
