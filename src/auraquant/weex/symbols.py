from __future__ import annotations

from typing import Set

WEEX_ALLOWED_CONTRACT_SYMBOLS: Set[str] = {
    "cmt_btcusdt",
    "cmt_ethusdt",
    "cmt_solusdt",
    "cmt_dogeusdt",
    "cmt_xrpusdt",
    "cmt_adausdt",
    "cmt_bnbusdt",
    "cmt_ltcusdt",
}


def to_weex_contract_symbol(symbol: str) -> str:
    """Map internal symbol like 'BTC/USDT' -> 'cmt_btcusdt'."""

    s = symbol.replace("/", "").lower()
    return f"cmt_{s}"


def is_allowed_contract_symbol(contract_symbol: str) -> bool:
    return contract_symbol.lower() in WEEX_ALLOWED_CONTRACT_SYMBOLS
