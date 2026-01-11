from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auraquant.core import Orchestrator, OrchestratorConfig
from auraquant.correlation import CorrelationTrigger
from auraquant.data import StaticMultiPriceProvider
from auraquant.execution import PaperOrderManager
from auraquant.risk import RiskEngine
from auraquant.sentiment import NewsItem, SentimentProcessor, StaticNewsProvider
from auraquant.util.ai_log import AiLogStore
from auraquant.learning import TradePolicyLearner


def _resolve_seed(seed_arg: int | None) -> int:
    if seed_arg is not None:
        return int(seed_arg)
    env = os.environ.get("AURAQUANT_SEED")
    if env is not None and env.strip() != "":
        try:
            return int(env)
        except ValueError:
            pass
    # Default: time-based so repeated runs differ.
    return int(time.time_ns() % 2_147_483_647)


def _make_correlated_series(
    *,
    points: int,
    seed: int,
    sol_start: float = 100.0,
    btc_start: float = 50000.0,
    rho: float = 0.85,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Generates (SOL_series, BTC_series) as [(price, atr), ...].

    Goal: keep demo judge-friendly but more realistic than a monotonic ramp:
    - Prices evolve via a correlated random walk with mild drift.
    - ATR scales with price (simple proxy), producing non-identical PnL per run.
    """

    points = max(int(points), 50)
    rng = random.Random(int(seed))

    btc_price = float(btc_start)
    sol_price = float(sol_start)

    # Per-tick drift/vol (1-minute ticks). Tuned to close trades reasonably fast.
    btc_drift = 0.00015
    btc_vol = 0.0025
    sol_drift = 0.00025
    sol_vol = 0.006

    rho = max(min(float(rho), 0.99), -0.99)
    rho_ortho = math.sqrt(max(1.0 - rho * rho, 0.0))

    sol_series: list[tuple[float, float]] = []
    btc_series: list[tuple[float, float]] = []

    for i in range(points):
        if i > 0 and i % 180 == 0:
            btc_vol = max(0.001, min(btc_vol * (0.8 + 0.4 * rng.random()), 0.01))
            sol_vol = max(0.002, min(sol_vol * (0.75 + 0.5 * rng.random()), 0.02))

        z1 = rng.gauss(0.0, 1.0)
        z2 = rng.gauss(0.0, 1.0)

        btc_ret = btc_drift + btc_vol * z1
        sol_ret = sol_drift + sol_vol * (rho * z1 + rho_ortho * z2)

        btc_price = max(10.0, btc_price * (1.0 + btc_ret))
        sol_price = max(0.1, sol_price * (1.0 + sol_ret))

        # ATR proxy (absolute USDT move). Keep SOL ATR relatively small so TP/SL can be hit.
        sol_atr = max(0.5, sol_price * 0.005)  # ~0.5% of price, min 0.5
        btc_atr = max(100.0, btc_price * 0.002)  # ~0.2% of price, min 100

        sol_series.append((float(sol_price), float(sol_atr)))
        btc_series.append((float(btc_price), float(btc_atr)))

    return sol_series, btc_series


def main() -> None:
    parser = argparse.ArgumentParser(description="AuraQuant orchestrator demo")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (int). If omitted, uses time-based seed.")
    parser.add_argument("--points", type=int, default=1200, help="Number of synthetic ticks to pre-generate")
    parser.add_argument(
        "--reset-learner",
        action="store_true",
        help="Delete models/dev_trade_policy.json before running (fresh learning)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("AuraQuant")

    seed = _resolve_seed(args.seed)
    # Market data (price, atr) sequence for multiple symbols.
    sol_series, btc_series = _make_correlated_series(points=int(args.points), seed=seed)
    prices = StaticMultiPriceProvider(series_by_symbol={"SOL/USDT": sol_series, "BTC/USDT": btc_series})

    now = datetime.now(timezone.utc)

    # Static news that tends to generate LONG bias in our heuristic.
    news_items = [
        NewsItem(
            title="Institutional inflows into BTC ETFs surge",
            published_at=now - timedelta(minutes=5),
            source="demo",
            url="https://example.com/a",
        ),
        NewsItem(
            title="Solana TVL hits new ATH as ecosystem growth accelerates",
            published_at=now - timedelta(minutes=12),
            source="demo",
            url="https://example.com/c",
        ),
    ]
    provider = StaticNewsProvider(items=news_items)
    sentiment = SentimentProcessor(logger=logger, provider=provider)

    execution = PaperOrderManager(starting_equity=1000.0)
    risk = RiskEngine(logger=logger)
    ai_log_store = AiLogStore(path="ai_logs/demo_ai_log.ndjson")

    model_path = Path("models/dev_trade_policy.json")
    if args.reset_learner and model_path.exists():
        model_path.unlink()
    learner = TradePolicyLearner(model_path="models/dev_trade_policy.json")

    cfg = OrchestratorConfig(
        symbol="SOL/USDT",
        lead_symbol="BTC/USDT",
        min_confidence=0.3,
        min_entry_interval_seconds=60,
        correlation_window=12,
        correlation_threshold=0.25,
        correlation_max_lag=3,
    )

    corr = CorrelationTrigger(
        logger=logger,
        lead_symbol=cfg.lead_symbol,
        window=cfg.correlation_window,
        max_lag=cfg.correlation_max_lag,
        corr_threshold=cfg.correlation_threshold,
    )
    bot = Orchestrator(
        logger=logger,
        config=cfg,
        sentiment=sentiment,
        correlation=corr,
        risk=risk,
        prices=prices,
        execution=execution,
        learner=learner,
        ai_log_store=ai_log_store,
    )

    target_closed_trades = 10
    max_ticks = 5000

    t = now
    for _ in range(max_ticks):
        bot.step(now=t)
        if execution.trades_closed() >= target_closed_trades:
            break
        t = t + timedelta(seconds=cfg.tick_seconds)

    equity_start = execution.starting_equity()
    equity_now = execution.equity()
    print("Equity start:", round(equity_start, 4))
    print("Final equity:", round(equity_now, 4))
    print("PnL total:", round(equity_now - equity_start, 4))
    print("Seed:", seed)
    print("Positions opened:", execution.positions_opened())
    print("Trades closed:", execution.trades_closed())
    print("Trade count:", execution.trade_count())
    print("AI logs written to: ai_logs/demo_ai_log.ndjson")
    print("Learner model path:", "models/dev_trade_policy.json")
    print("Learner seen trades:", learner.seen())


if __name__ == "__main__":
    main()
