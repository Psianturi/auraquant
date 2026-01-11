from __future__ import annotations

import logging
import sys
from pathlib import Path

#   python src/demo_risk_engine.py
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auraquant.risk import RiskEngine, TradeIntent, TradeResult


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("AuraQuant")

    engine = RiskEngine(logger=logger)

    equity = 1000.0
    intent = TradeIntent(
        symbol="SOL/USDT",
        side="LONG",
        entry_price=100.0,
        atr=2.0,
        confidence=0.9,
        requested_leverage=15,
    )

    decision = engine.validate_intent(intent, equity_now=equity)
    print("Decision:", decision.decision, decision.reason)

    # Simulate 3 losing trades to trigger cooldown
    for i in range(3):
        equity -= 10
        engine.update_account_state(equity_now=equity, trade_result=TradeResult(symbol=intent.symbol, pnl_usdt=-10.0))
        d = engine.validate_intent(intent, equity_now=equity)
        print(f"After loss {i+1}:", d.decision, d.reason)


if __name__ == "__main__":
    main()
