from __future__ import annotations

"""WEEX public API smoke test (no API key required).

- Check connectivity: /capi/v2/market/time
- Check ticker: /capi/v2/market/ticker

Usage:
  python src/weex_public_smoketest.py
"""

import json
import urllib.parse
import urllib.request

BASE_URL = "https://api-contract.weex.com"


def _get(path: str, params: dict | None = None) -> dict:
    url = BASE_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        try:
            return json.loads(body)
        except Exception:
            return {"raw": body}


def main() -> None:
    print("== WEEX Public Smoke Test ==")
    t = _get("/capi/v2/market/time")
    print("market/time =>", t)

    for sym in ["cmt_btcusdt", "cmt_solusdt"]:
        tick = _get("/capi/v2/market/ticker", {"symbol": sym})
        last = tick.get("last")
        print(f"ticker {sym} last =>", last)


if __name__ == "__main__":
    main()
