#!/usr/bin/env python3
"""WEEX account snapshot (safe)

Usage:
  python src/weex_account_snapshot.py

Optional env:
  WEEX_BASE_URL (default https://api-contract.weex.com)
  WEEX_TARGET_QUOTE or WEEX_PRODUCT_SYMBOL (default cmt_btcusdt)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("WEEX_BASE_URL", "https://api-contract.weex.com").rstrip("/")
API_KEY = os.getenv("WEEX_API_KEY")
SECRET_KEY = os.getenv("WEEX_SECRET_KEY")
PASSPHRASE = os.getenv("WEEX_PASSPHRASE")
SYMBOL = (os.getenv("WEEX_PRODUCT_SYMBOL") or os.getenv("WEEX_TARGET_QUOTE") or "cmt_btcusdt").strip()


def _require_env() -> None:
    missing: list[str] = []
    if not API_KEY:
        missing.append("WEEX_API_KEY")
    if not SECRET_KEY:
        missing.append("WEEX_SECRET_KEY")
    if not PASSPHRASE:
        missing.append("WEEX_PASSPHRASE")
    if missing:
        raise SystemExit(f"Missing required .env variables: {', '.join(missing)}")


def _json_pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def generate_signature(timestamp: str, method: str, request_path: str, query_string: str = "", body: str = "") -> str:
    message = timestamp + method.upper() + request_path + query_string + body
    signature = hmac.new(
        SECRET_KEY.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.b64encode(signature).decode("utf-8")


def get_headers(timestamp: str, signature: str) -> dict[str, str]:
    return {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json",
        "locale": "en-US",
    }


def signed_get(path: str, query: str = "") -> requests.Response:
    ts = str(int(time.time() * 1000))
    sig = generate_signature(ts, "GET", path, query)
    headers = get_headers(ts, sig)
    return requests.get(f"{BASE_URL}{path}{query}", headers=headers, timeout=15)


def _extract_usdt_asset(data: Any) -> dict[str, Any] | None:
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict) and row.get("coinName") == "USDT":
                return row
    return None


def main() -> None:
    _require_env()

    print("=" * 70)
    print("WEEX ACCOUNT SNAPSHOT (SAFE)")
    print("=" * 70)
    print(f"Base URL: {BASE_URL}")
    print(f"Symbol:   {SYMBOL}")
    print("(No credentials will be printed.)")
    print()

    # Assets
    print("[1] /account/assets")
    resp = signed_get("/capi/v2/account/assets")
    print(f"Status: {resp.status_code}")
    try:
        assets = resp.json()
    except Exception:
        print(resp.text)
        assets = None

    if assets is not None:
        usdt = _extract_usdt_asset(assets)
        if usdt:
            print("USDT summary:")
            for k in ("available", "equity", "frozen", "unrealizePnl"):
                if k in usdt:
                    print(f"  {k}: {usdt.get(k)}")
        else:
            print(_json_pretty(assets)[:2000])
    print()

    # Positions (best-effort; WEEX may vary)
    print("[2] /account/position/allPosition")
    resp = signed_get("/capi/v2/account/position/allPosition")
    print(f"Status: {resp.status_code}")
    text = resp.text
    try:
        data = resp.json()
    except Exception:
        print(text[:2000])
        data = None

    if data is not None:
        rows: list[Any] = []
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and isinstance(data.get("data"), list):
            rows = data["data"]

        match = None
        for row in rows:
            if isinstance(row, dict) and str(row.get("symbol", "")).lower() == SYMBOL.lower():
                match = row
                break

        if match is None:
            print("No position row matched symbol (may be empty/no open position or different response shape).")
            print(_json_pretty(data)[:2000])
        else:
            print("Position summary (best-effort fields):")
            keys = [
                "symbol",
                "positionSide",
                "side",
                "holdSide",
                "posSide",
                "holdVol",
                "size",
                "positionSize",
                "avgOpenPrice",
                "openPrice",
                "priceAvg",
                "avgPrice",
                "markPrice",
                "unrealizedPnl",
                "unrealisePnl",
                "realizedPnl",
                "realizePnl",
                "margin",
                "leverage",
            ]
            printed = False
            for k in keys:
                if k in match:
                    print(f"  {k}: {match.get(k)}")
                    printed = True
            if not printed:
                print(_json_pretty(match)[:2000])
    print()

    print("[3] /account/position/singlePosition?symbol=...")
    q = f"?symbol={SYMBOL}"
    resp = signed_get("/capi/v2/account/position/singlePosition", q)
    print(f"Status: {resp.status_code}")
    try:
        data = resp.json()
        print(_json_pretty(data)[:2000])
    except Exception:
        print(resp.text[:2000])


if __name__ == "__main__":
    main()
