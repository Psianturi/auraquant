#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List
.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from auraquant.weex.private_client import WeexPrivateRestClient
from auraquant.weex.symbols import to_weex_contract_symbol
from auraquant.weex.emergency_close import emergency_close, fetch_open_positions


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

def main() -> int:
    ap = argparse.ArgumentParser(description="WEEX: close all positions + cancel pending orders")
    ap.add_argument("--execute", action="store_true", help="Actually send cancel/close requests")
    ap.add_argument(
        "--symbols",
        default=",".join(ALLOWED_SPOT_SYMBOLS),
        help="Comma-separated symbols like BTC/USDT,ETH/USDT (defaults to allowed list)",
    )
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

    print("[INFO] execute=%s" % execute)

    if not execute:

        for ws in weex_symbols:
            print(f"[DRY] cancelAllOrders(normal) symbol={ws}")
            print(f"[DRY] cancelAllOrders(plan) symbol={ws}")

        positions = fetch_open_positions(client)
        if positions:
            print("[INFO] Open positions:")
            for sym, side, size in positions:
                print(f"  - {sym} {side} size={size}")
            for sym, side, size in positions:
                print(f"[DRY] closePositions symbol={sym} (side={side} size={size})")
        else:
            print("[INFO] No open positions reported by API.")

        time.sleep(float(args.sleep))
        remaining = fetch_open_positions(client)
        if remaining:
            print("[WARN] Remaining open positions after cleanup:")
            for sym, side, size in remaining:
                print(f"  - {sym} {side} size={size}")
            return 1

        print("[OK] No open positions remain (per API).")
        return 0

    # Execute: cancel orders + close positions using official endpoints.
    result = emergency_close(client=client, weex_symbols=weex_symbols, sleep_seconds=float(args.sleep))
    remaining = result.get("remaining_positions") or []
    if remaining:
        print("[WARN] Remaining open positions after cleanup:")
        for r in remaining:
            print(f"  - {r}")
        return 1

    print("[OK] No open positions remain (per API).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
