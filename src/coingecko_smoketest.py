#!/usr/bin/env python3
"""CoinGecko integration smoke test.

Purpose:
- Prove meaningful CoinGecko usage for the CoinGecko API Track
- Demonstrate market discovery / universe selection for WEEX allowed pairs

This script does NOT place any WEEX orders.
"""

from __future__ import annotations

import logging

from auraquant.data.coingecko_client import CoinGeckoClient, pick_weex_contract_symbol_by_liquidity
from auraquant.weex.symbols import WEEX_ALLOWED_CONTRACT_SYMBOLS


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    client = CoinGeckoClient()
    picked = pick_weex_contract_symbol_by_liquidity(
        client=client,
        allowed_contract_symbols=sorted(WEEX_ALLOWED_CONTRACT_SYMBOLS),
        ttl_seconds=300.0,
    )

    print("Allowed WEEX symbols:")
    for s in sorted(WEEX_ALLOWED_CONTRACT_SYMBOLS):
        print(" -", s)
    print()
    print("Picked symbol (CoinGecko liquidity/move heuristic):", picked)


if __name__ == "__main__":
    main()
