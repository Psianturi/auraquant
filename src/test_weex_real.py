#!/usr/bin/env python3
"""
Test WEEX API Real Connection
Verifies API credentials and completes the official API test flow.

References:
- API Test Guide (process): https://www.weex.com/news/detail/ai-wars-weex-alpha-awakens-weex-global-hackathon-api-test-process-guide-264335
- API Testing Task: https://www.weex.com/events/promo/apitesting

Task requirement (apitesting):
- Execute an order via API with ~10 USDT notional on the BTCUSDT trading pair.

Safety:
- By default, this script does NOT place an order.
- Set WEEX_EXECUTE_ORDER=1 to actually place an order.
"""

import os
import time
import hmac
import hashlib
import base64
import json
import math
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("WEEX_BASE_URL", "https://api-contract.weex.com").rstrip("/")

API_KEY = os.getenv("WEEX_API_KEY")
SECRET_KEY = os.getenv("WEEX_SECRET_KEY")
PASSPHRASE = os.getenv("WEEX_PASSPHRASE")

# Match the task page defaults
TARGET_QUOTE = os.getenv("WEEX_TARGET_QUOTE", "BTCUSDT")
TARGET_NOTIONAL_USDT = float(os.getenv("WEEX_TARGET_NOTIONAL_USDT", "10"))
DEFAULT_LEVERAGE = int(os.getenv("WEEX_LEVERAGE", "2"))
EXECUTE_ORDER = os.getenv("WEEX_EXECUTE_ORDER", "0") == "1"


def _require_env() -> None:
    missing = []
    if not API_KEY:
        missing.append("WEEX_API_KEY")
    if not SECRET_KEY:
        missing.append("WEEX_SECRET_KEY")
    if not PASSPHRASE:
        missing.append("WEEX_PASSPHRASE")
    if missing:
        raise SystemExit(f"Missing required .env variables: {', '.join(missing)}")


def _json_dumps_compact(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def generate_signature(timestamp, method, request_path, query_string="", body=""):
    """Signature per WEEX docs: ts + METHOD + path + query + body(JSON string for POST)."""
    message = timestamp + method.upper() + request_path + query_string + body
    signature = hmac.new(
        SECRET_KEY.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).digest()
    return base64.b64encode(signature).decode('utf-8')


def get_headers(timestamp, signature):
    return {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json",
        "locale": "en-US"
    }


def signed_get(path: str, query: str = ""):
    timestamp = str(int(time.time() * 1000))
    signature = generate_signature(timestamp, "GET", path, query)
    headers = get_headers(timestamp, signature)
    url = f"{BASE_URL}{path}{query}"
    return requests.get(url, headers=headers, timeout=15)


def signed_post(path: str, body_obj: dict, query: str = ""):
    body = _json_dumps_compact(body_obj)
    timestamp = str(int(time.time() * 1000))
    signature = generate_signature(timestamp, "POST", path, query, body)
    headers = get_headers(timestamp, signature)
    url = f"{BASE_URL}{path}{query}"
    return requests.post(url, headers=headers, data=body, timeout=15)


def public_get(path: str, params: dict | None = None):
    # Some WEEX endpoints behave better with explicit headers.
    # Keep this lightweight; auth headers are not required for public endpoints.
    public_headers = {
        "locale": "en-US",
        "Content-Type": "application/json",
        "User-Agent": "AuraQuant-WeexTester/1.0",
    }
    url = f"{BASE_URL}{path}"
    return requests.get(url, params=params, headers=public_headers, timeout=15)


def _extract_products_symbols(products_json) -> list[str]:
    """Best-effort extraction of symbol strings from /capi/v2/products."""
    if isinstance(products_json, list):
        items = products_json
    elif isinstance(products_json, dict):
        data = products_json.get("data")
        items = data if isinstance(data, list) else []
    else:
        items = []

    symbols: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        sym = item.get("symbol") or item.get("Symbol")
        if isinstance(sym, str) and sym:
            symbols.append(sym)
    return symbols


def choose_btcusdt_symbol() -> str:
    """Choose a BTCUSDT symbol strictly from /products as required by apitesting FAQ."""
    # If user pinned an exact product symbol, trust it.
    pinned = os.getenv("WEEX_PRODUCT_SYMBOL")
    if pinned:
        return pinned

    fallback = os.getenv("WEEX_FALLBACK_SYMBOL", "cmt_btcusdt")

    resp = public_get("/capi/v2/products")
    if resp.status_code != 200:
        # We have seen /products intermittently return 521 even when other endpoints work.
        # Do not hard-fail the whole test flow; proceed with a known contract symbol.
        print(f"[WARN] /products returned {resp.status_code}. Proceeding with fallback symbol: {fallback}")
        return fallback

    try:
        data = resp.json()
    except Exception:
        print("[WARN] /products returned non-JSON. Proceeding with fallback symbol: " + fallback)
        return fallback

    symbols = _extract_products_symbols(data)

    # Find symbols that contain the target quote string (case-insensitive)
    target = TARGET_QUOTE.upper()
    candidates = [s for s in symbols if target in s.upper()]
    if not candidates:
        print(
            f"[WARN] No product symbol matched {TARGET_QUOTE!r}. "
            f"Set WEEX_PRODUCT_SYMBOL explicitly. Proceeding with fallback symbol: {fallback}"
        )
        return fallback

    # Prefer ones that look like futures contract pairs
    preferred = [s for s in candidates if "CMT" in s.upper()] or candidates
    return preferred[0]


def floor_to_decimals(value: float, decimals: int) -> float:
    factor = 10 ** decimals
    return math.floor(value * factor) / factor


def safe_int_leverage(value: int) -> int:
    if value < 1:
        return 1
    if value > 20:
        return 20
    return value

print("=" * 70)
print("WEEX API REAL CONNECTION TEST")
print("=" * 70)
print(f"Base URL: {BASE_URL}")
print(f"Target:   {TARGET_QUOTE} (~{TARGET_NOTIONAL_USDT} USDT notional)")
print(f"Leverage: {safe_int_leverage(DEFAULT_LEVERAGE)}x")
print(f"Execute:  {'YES' if EXECUTE_ORDER else 'NO (set WEEX_EXECUTE_ORDER=1 to execute)'}")
print(f"API Key: {API_KEY[:20]}..." if API_KEY else "API Key: <missing>")
print("=" * 70)
print()

try:
    _require_env()
except SystemExit as e:
    print(f"[FATAL] {e}")
    raise

# Test 0: Get products and select correct BTCUSDT symbol
print("[TEST 0] Products - Select BTCUSDT Symbol")
print("-" * 70)

PRODUCT_SYMBOL = None
try:
    PRODUCT_SYMBOL = choose_btcusdt_symbol()
    print(f"[OK] Selected product symbol: {PRODUCT_SYMBOL}")
except Exception as e:
    # Should be rare after fallback; keep running with a safe default.
    PRODUCT_SYMBOL = os.getenv("WEEX_FALLBACK_SYMBOL", "cmt_btcusdt")
    print(f"[WARN] Products auto-detect failed: {e}")
    print(f"[WARN] Proceeding with fallback symbol: {PRODUCT_SYMBOL}")

print()

# Test 1: Get price ticker for the chosen symbol
print("[TEST 1] Market Ticker")
print("-" * 70)
last_price = None
try:
    response = public_get("/capi/v2/market/ticker", params={"symbol": PRODUCT_SYMBOL})
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        last_price = float(data.get("last")) if data.get("last") is not None else None
        print(f"[OK] Symbol: {data.get('symbol', PRODUCT_SYMBOL)}")
        print(f"[OK] Last:   {data.get('last', 'N/A')}")
        print(f"[OK] Mark:   {data.get('markPrice', 'N/A')}")
    else:
        print(f"[FAILED] {response.text}")
except Exception as e:
    print(f"[ERROR] {e}")

print()

# Test 2: Account assets (balance)
print("[TEST 2] Account Assets")
print("-" * 70)
try:
    response = signed_get("/capi/v2/account/assets")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("[OK] API Access Successful!")
        print(f"Response: {json.dumps(data, indent=2)}")
    elif response.status_code == 521:
        print("[BLOCKED] Error 521 - IP not whitelisted")
        print("Action: Contact WEEX admin to whitelist your VPS IP")
    else:
        print(f"[FAILED] {response.text}")
except Exception as e:
    print(f"[ERROR] {e}")

print()

# Test 3: Set leverage (required by the official test flow)
print("[TEST 3] Set Leverage")
print("-" * 70)
lev = safe_int_leverage(DEFAULT_LEVERAGE)
leverage_body = {
    "symbol": PRODUCT_SYMBOL,
    "marginMode": 1,
    "longLeverage": str(lev),
    "shortLeverage": str(lev),
}
try:
    response = signed_post("/capi/v2/account/leverage", leverage_body)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("[OK] Leverage set request accepted")
        print(f"Response: {response.text}")
    else:
        print(f"[FAILED] {response.text}")
except Exception as e:
    print(f"[ERROR] {e}")

print()

# Test 4: Place order (~10 USDT notional) on BTCUSDT (apitesting task)
print("[TEST 4] Place Order (~10 USDT notional, BTCUSDT)")
print("-" * 70)

order_id = None
client_oid = f"auraquant_{int(time.time() * 1000)}"

if last_price is None:
    print("[SKIP] Missing last price from ticker; cannot compute size")
else:
    computed_size = TARGET_NOTIONAL_USDT / last_price
    computed_size = floor_to_decimals(computed_size, 6)
    order_size = os.getenv("WEEX_ORDER_SIZE", f"{computed_size:.6f}")
    try:
        notional_est = float(order_size) * float(last_price)
    except Exception:
        notional_est = None

    print(f"Computed size (approx): {order_size}")
    if notional_est is not None:
        print(f"Estimated notional:     {notional_est:.4f} USDT")

    order_body = {
        "symbol": PRODUCT_SYMBOL,
        "client_oid": client_oid,
        "size": str(order_size),
        "type": "1",  # 1: open long
        "order_type": "0",
        "match_price": "1",  # market
        "price": str(last_price),
        "marginMode": 1,
    }

    if not EXECUTE_ORDER:
        print("[DRY RUN] Order placement is disabled.")
        print("Set WEEX_EXECUTE_ORDER=1 to execute the real order.")
        print(f"Payload: {json.dumps(order_body, indent=2)}")
    else:
        try:
            resp = signed_post("/capi/v2/order/placeOrder", order_body)
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                order_id = data.get("order_id") or data.get("orderId")
                print("[OK] Order placed")
                print(f"Response: {json.dumps(data, indent=2)}")
            else:
                print(f"[FAILED] {resp.text}")
        except Exception as e:
            print(f"[ERROR] {e}")

print()

# Test 5: Get current orders (optional but part of official flow)
print("[TEST 5] Get Current Orders")
print("-" * 70)
try:
    q = f"?symbol={PRODUCT_SYMBOL}"
    if order_id:
        q += f"&orderId={order_id}"
    response = signed_get("/capi/v2/order/current", q)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"[ERROR] {e}")

print()

# Test 6: Get order history
print("[TEST 6] Get Order History")
print("-" * 70)
try:
    now_ms = int(time.time() * 1000)
    day_ms = 24 * 60 * 60 * 1000
    create_date = now_ms - day_ms
    q = f"?symbol={PRODUCT_SYMBOL}&pageSize=10&createDate={create_date}"
    response = signed_get("/capi/v2/order/history", q)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"[ERROR] {e}")

print()

# Test 7: Get fills (trade details)
print("[TEST 7] Get Fills (Trade Details)")
print("-" * 70)
try:
    q = f"?symbol={PRODUCT_SYMBOL}&limit=1"
    if order_id:
        q = f"?symbol={PRODUCT_SYMBOL}&orderId={order_id}&limit=100"
    response = signed_get("/capi/v2/order/fills", q)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"[ERROR] {e}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("This test uses REAL WEEX API credentials from .env")
print("NOT mock data or paper trading")
print()
print("Official flow (high level):")
print("  1) /products -> pick correct symbol")
print("  2) /account/assets -> check 1000 USDT test funds")
print("  3) /account/leverage -> set leverage (<=20x)")
print("  4) /order/placeOrder -> execute ~10 USDT notional on BTCUSDT")
print("  5) /order/current, /order/history, /order/fills -> verify result")
print()
print("Notes from apitesting FAQ:")
print("  - For ALL symbol parameters, strictly use values returned by /products")
print("  - Symbols are case-sensitive; follow exactly what /products returns")
print("  - Max leverage via API orders is 20x")
print()
print("If you see Error 521:")
print("  - IP 157.15.124.107 needs to be whitelisted by WEEX admin")
print("  - Contact: https://t.me/weexaiwars")
print("=" * 70)
