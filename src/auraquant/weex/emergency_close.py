from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .private_client import WeexPrivateRestClient


@dataclass(frozen=True)
class CancelAllResult:
    cancel_order_type: str
    status_code: int
    ok: bool
    response_text: str


@dataclass(frozen=True)
class ClosePositionsResult:
    status_code: int
    ok: bool
    response_text: str


def _signed_post_first_ok(
    client: WeexPrivateRestClient,
    paths: Iterable[str],
    body: Dict[str, Any],
) -> Tuple[int, bool, str]:
    last_status = 0
    last_text = ""
    for p in paths:
        try:
            resp = client.signed_post(p, body)
        except Exception as e:
            last_status = 0
            last_text = str(e)
            continue

        last_status = int(getattr(resp, "status_code", 0) or 0)
        last_text = (getattr(resp, "text", "") or "")[:1000]
        if last_status == 200:
            return last_status, True, last_text

        # Try next fallback on 400/404 (common for path changes).
        if last_status in (400, 404):
            continue

    return last_status, False, last_text


def cancel_all_orders(
    client: WeexPrivateRestClient,
    weex_symbol: Optional[str],
    cancel_order_type: str,
) -> CancelAllResult:

    cancel_order_type = str(cancel_order_type).strip().lower()
    if cancel_order_type not in ("normal", "plan"):
        raise ValueError("cancel_order_type must be 'normal' or 'plan'")

    body: Dict[str, Any] = {"cancelOrderType": cancel_order_type}
    if weex_symbol:
        body["symbol"] = str(weex_symbol)

    paths = (
        "/capi/v2/order/cancelAllOrders",
        "/capi/v2/order/cancelAll",  # legacy fallback
    )

    status, ok, text = _signed_post_first_ok(client, paths=paths, body=body)
    return CancelAllResult(
        cancel_order_type=cancel_order_type,
        status_code=status,
        ok=ok,
        response_text=text,
    )


def close_positions(client: WeexPrivateRestClient, weex_symbol: Optional[str]) -> ClosePositionsResult:
    """Close positions using the official endpoint.

    Docs: POST /capi/v2/order/closePositions
    If symbol is omitted, closes all positions.
    """

    body: Dict[str, Any] = {}
    if weex_symbol:
        body["symbol"] = str(weex_symbol)

    paths = (
        "/capi/v2/order/closePositions",
    )

    status, ok, text = _signed_post_first_ok(client, paths=paths, body=body)
    return ClosePositionsResult(status_code=status, ok=ok, response_text=text)


def fetch_open_positions(client: WeexPrivateRestClient) -> List[Tuple[str, str, float]]:
    """Returns list of (weex_symbol, side[long|short], size)."""

    def _signed_get_first_ok(paths: Iterable[str], query: str = ""):
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

    def _parse_positions_payload(payload: object) -> Optional[List[Dict[str, Any]]]:
        if isinstance(payload, list):
            return [p for p in payload if isinstance(p, dict)]
        if isinstance(payload, dict):
            data_val = payload.get("data")
            if isinstance(data_val, list):
                return [p for p in data_val if isinstance(p, dict)]
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

    paths = (
        "/capi/v2/account/position/allPosition",
        "/capi/v2/position/allPosition",
    )
    resp = _signed_get_first_ok(paths)
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
            continue
        out.append((sym, side, float(size)))
    return out


def emergency_close(
    client: WeexPrivateRestClient,
    weex_symbols: Optional[Sequence[str]] = None,
    sleep_seconds: float = 2.0,
) -> Dict[str, Any]:
    """Best-effort emergency close.

    - Cancel normal orders (per symbol, if provided)
    - Cancel plan/trigger orders (per symbol, if provided)
    - Close positions (per symbol, if provided)
    - Recheck remaining positions

    Returns a dict with results for logging.
    """

    out: Dict[str, Any] = {
        "cancel": [],
        "close": [],
        "remaining_positions": [],
    }

    targets = list(weex_symbols or [])
    if not targets:
        targets = [None]  # type: ignore[list-item]

    for sym in targets:
        out["cancel"].append(cancel_all_orders(client, sym, cancel_order_type="normal").__dict__)
        out["cancel"].append(cancel_all_orders(client, sym, cancel_order_type="plan").__dict__)

    for sym in targets:
        out["close"].append(close_positions(client, sym).__dict__)

    time.sleep(float(sleep_seconds))
    try:
        out["remaining_positions"] = [
            {"symbol": s, "side": side, "size": size} for (s, side, size) in fetch_open_positions(client)
        ]
    except Exception as e:
        out["remaining_positions"] = [{"error": str(e)}]

    return out
