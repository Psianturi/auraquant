#!/usr/bin/env python3


import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv

from auraquant.core.orchestrator import Orchestrator
from auraquant.core.types import OrchestratorConfig
from auraquant.correlation.correlation_trigger import CorrelationTrigger
from auraquant.data.weex_rest_price_provider import WeexRestMultiPriceProvider
from auraquant.execution.weex_order_manager import WeexOrderManager
from auraquant.risk.risk_engine import RiskEngine
from auraquant.sentiment.providers import CryptoPanicProvider, NewsProvider
from auraquant.sentiment.sentiment_processor import SentimentProcessor
from auraquant.sentiment.types import NewsItem
from auraquant.util.ai_log.store import AiLogStore
from auraquant.util.ai_log.realtime_uploader import make_uploader_from_env

from auraquant.weex.private_client import WeexPrivateRestClient
from auraquant.weex.symbols import to_weex_contract_symbol


REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=REPO_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class PriceMomentumNewsProvider(NewsProvider):

    prices: WeexRestMultiPriceProvider

    def fetch_latest(self, symbol: str, limit: int = 5):

        internal = f"{symbol}/USDT"
        now = datetime.utcnow()
        recent = self.prices.get_recent_prices(internal, window=12)
        bias = "flat"
        ret = 0.0
        if len(recent) >= 2 and recent[0] > 0:
            ret = (recent[-1] - recent[0]) / max(recent[0], 1e-12)
            if ret > 0.0015:
                bias = "bullish"
            elif ret < -0.0015:
                bias = "bearish"
        titles = [
            f"{symbol} momentum {bias} (ret={ret:+.4%})",
            f"{symbol} volatility check based on recent ticks",
        ]
        items = [
            NewsItem(title=t, published_at=now - timedelta(seconds=30), source="weex_ticker", url=None)
            for t in titles
        ]
        return items[:limit]


class AutonomousOrchestratorTest:
    """Run real orchestrator for N minutes with live ticks + AI logging."""

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        duration_seconds: int = 600,
        min_trades: int = 10,
        log_file: str = "ai_logs/test_demo_orchestrator_realtime.ndjson",
    ):
        self.symbol = symbol
        self.duration_seconds = duration_seconds
        self.min_trades = min_trades
        self.log_file = log_file

        self.start_time = None
        self.end_time = None
        self.tick_count = 0
        self.trade_count = 0
        self.ai_log_count = 0
        self.upload_success = 0
        self.upload_failed = 0

        self.store = AiLogStore(log_file)
        self.uploader = make_uploader_from_env(queue_dir="ai_logs/.upload_queue")
        if self.uploader is not None:
            self.uploader.start()
            logger.info("[INIT] Real-time uploader enabled")
        else:
            logger.warning("[INIT] Real-time uploader disabled (WEEX_AI_LOG_UPLOAD_URL not set)")

        # Monkeypatch store.append to also push to uploader, without changing core orchestrator.
        orig_append = self.store.append

        def _append_and_upload(event):
            orig_append(event)
            self.ai_log_count += 1
            if self.uploader is not None:
                ok = bool(self.uploader.upload(event.to_payload()))
                if ok:
                    self.upload_success += 1
                else:
                    self.upload_failed += 1

        self.store.append = _append_and_upload  # type: ignore[method-assign]

        # Build real orchestrator components
        bot_logger = logging.getLogger("auraquant.orch_test")
        prices = WeexRestMultiPriceProvider()

        # Prefer real news if CRYPTOPANIC_API_TOKEN is set; else fall back to real-price momentum.
        provider: NewsProvider
        if os.getenv("CRYPTOPANIC_API_TOKEN"):
            try:
                provider = CryptoPanicProvider()
            except Exception:
                provider = PriceMomentumNewsProvider(prices=prices)
        else:
            provider = PriceMomentumNewsProvider(prices=prices)

        sentiment = SentimentProcessor(logger=bot_logger, provider=provider)
        sentiment.news_cache_ttl_minutes = 0.0
        correlation = CorrelationTrigger(logger=bot_logger)
        risk = RiskEngine(logger=bot_logger)
  
        risk.sl_atr_mult = 0.75
        risk.tp_atr_mult = 1.25
        client = WeexPrivateRestClient()
        execution = WeexOrderManager(client=client)

        # Make it possible to hit >=10 entries in 10 minutes.
        config = OrchestratorConfig(
            symbol=self.symbol,
            tick_seconds=20,
            min_entry_interval_seconds=20,
            enforce_weex_allowlist=True,
        )

        correlation.window = 10

        self.execution = execution
        self.orchestrator = Orchestrator(
            logger=bot_logger,
            config=config,
            sentiment=sentiment,
            correlation=correlation,
            risk=risk,
            prices=prices,
            execution=execution,
            ai_log_store=self.store,
            ai_log_uploader=self.uploader,
        )

        # Official guide connectivity checks (no secrets printed)
        self._weex_connectivity_checks(symbol=config.symbol)

    def _weex_connectivity_checks(self, symbol: str) -> None:
        weex_symbol = to_weex_contract_symbol(symbol)
        logger.info("[WEEX] Connectivity checks per official guide")
        logger.info(f"[WEEX] Target symbol: {symbol} -> {weex_symbol}")

        # 1) Public ticker
        try:
            prices = WeexRestMultiPriceProvider()
            tick = prices.get_tick([symbol], now=datetime.utcnow())
            last, _atr = tick.get(symbol, (0.0, 0.0))
            logger.info(f"[WEEX] Market ticker OK (last={last})")
        except Exception as e:
            raise RuntimeError(f"WEEX public ticker failed: {e}")

        # 2) Private assets
        try:
            self.execution.reconcile(now=datetime.utcnow())
            logger.info(f"[WEEX] Account assets OK (equity={self.execution.equity():.4f} USDT)")
        except Exception as e:
            raise RuntimeError(f"WEEX account assets failed: {e}")

        # 3) Set leverage
        try:
            if hasattr(self.execution, "set_leverage"):
                self.execution.set_leverage(symbol=symbol, leverage=int(os.getenv("WEEX_LEVERAGE", "2")))
            logger.info("[WEEX] Set leverage OK")
        except Exception as e:
            raise RuntimeError(f"WEEX set leverage failed: {e}")

    def run(self):
        """Run autonomous orchestrator test."""
        self.start_time = datetime.now(timezone.utc)
        logger.info(f"[START] Running {self.duration_seconds}s autonomous orchestrator test")
        logger.info(f"[START] Min trades required: {self.min_trades}")
        logger.info(f"[START] Start time: {self.start_time.isoformat()}")

        tick_interval = int(getattr(self.orchestrator.config, "tick_seconds", 20) or 20)

        while True:
            self.tick_count += 1
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()

            if elapsed >= self.duration_seconds:
                logger.info(f"[END] Duration limit ({elapsed:.1f}s >= {self.duration_seconds}s)")
                break

            try:
                now = datetime.utcnow()
                self.orchestrator.step(now=now)

                # Track trades as "positions opened" (entries)
                self.trade_count = int(self.execution.positions_opened())
                logger.info(
                    f"[TICK {self.tick_count}] positions_opened={self.trade_count} equity={self.execution.equity():.2f}"
                )

            except Exception as e:
                logger.error(f"[ERROR] Tick {self.tick_count}: {e}", exc_info=True)

            # Wait before next tick
            time.sleep(tick_interval)

        self.end_time = datetime.now(timezone.utc)
        self._print_report()

    def _print_report(self):
        """Print final report."""
        if not self.end_time:
            return

        total_time = (self.end_time - self.start_time).total_seconds()

        status = "PASS" if self.trade_count >= self.min_trades else "CHECK"
        report = (
            "\n"
            "============================================================\n"
            "AUTONOMOUS ORCHESTRATOR TEST REPORT\n"
            "============================================================\n"
            f"[TEST PARAMETERS]\n"
            f"  Duration: {total_time:.1f}s (target: {self.duration_seconds}s)\n"
            f"  Min Trades Required: {self.min_trades}\n"
            f"  Actual Trades Executed (positions_opened): {self.trade_count}\n"
            f"  Ticks Completed: {self.tick_count}\n"
            "\n"
            f"[AI LOGGING METRICS]\n"
            f"  AI Log Events Generated: {self.ai_log_count}\n"
            f"  Successfully Uploaded: {self.upload_success}\n"
            f"  Queued (failed initially): {self.upload_failed}\n"
            f"  Local Store: {self.log_file}\n"
            "\n"
            f"[STATUS]\n"
            f"  {status}\n"
            "============================================================\n"
        )
        print(report)
        logger.info(report)

    def cleanup(self):
        """Graceful shutdown."""
        try:
            pos = self.execution.position() if hasattr(self, "execution") else None
            if pos is not None:
                logger.warning("[CLEANUP] Detected an open position; attempting best-effort close")
                closer = getattr(self.execution, "close_open_position_best_effort", None)
                if callable(closer):
                    closer()
        except Exception:
            logger.exception("[CLEANUP] Failed to close open position (continuing shutdown)")

        if self.uploader:
            logger.info("[CLEANUP] Stopping uploader (flushing queued events)...")
            self.uploader.stop()
        logger.info("[CLEANUP] Test completed")


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous orchestrator real-time test (10-12 min with AI logging)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=360,
        help="Duration in seconds (default: 360 = 6 min)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=os.getenv("WEEX_SYMBOL", "BTC/USDT"),
        help="Trading symbol (default: BTC/USDT). Also supports env WEEX_SYMBOL.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=10,
        help="Minimum trades required (default: 10)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="ai_logs/test_demo_orchestrator_realtime.ndjson",
        help="AI log file",
    )
    args = parser.parse_args()

    try:
        test = AutonomousOrchestratorTest(
            symbol=args.symbol,
            duration_seconds=args.duration,
            min_trades=args.min_trades,
            log_file=args.log_file,
        )
    except Exception as e:
        logger.error("[FATAL] Init failed. ", exc_info=True)
        raise SystemExit(1) from e

    try:
        test.run()
    except KeyboardInterrupt:
        logger.info("[INTERRUPT] Test interrupted")
    except Exception as e:
        logger.error(f"[ERROR] {e}", exc_info=True)
    finally:
        test.cleanup()


if __name__ == "__main__":
    main()
