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
from auraquant.core.types import BotPhase, OrchestratorConfig
from auraquant.correlation.correlation_trigger import CorrelationTrigger
from auraquant.data.multi_price_provider import MultiPriceProvider
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

    prices: MultiPriceProvider

    def fetch_latest(self, symbol: str, limit: int = 5):

        # `symbol` is expected to be the internal pair format (e.g. "BTC/USDT").
        internal = symbol if "/" in symbol else f"{symbol}/USDT"
        now = datetime.utcnow()
        recent = self.prices.get_recent_prices(internal, window=12)
        bias = "flat"
        ret = 0.0
        if len(recent) >= 2 and recent[0] > 0:
            ret = (recent[-1] - recent[0]) / max(recent[0], 1e-12)
            # Lowered to 0.04% for more signals during very flat markets
            threshold = float(os.getenv("MOMENTUM_THRESHOLD", "0.0004"))
            if ret > threshold:
                bias = "bullish"
            elif ret < -threshold:
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


class CachedTickPriceProvider:
    """Wraps a MultiPriceProvider and caches per-symbol ticks briefly.

    This prevents duplicate REST calls when we pre-warm multiple symbols and
    then the orchestrator immediately fetches the active symbol again.
    """

    def __init__(self, inner: WeexRestMultiPriceProvider, ttl_seconds: float = 2.0) -> None:
        self._inner = inner
        self._ttl = float(ttl_seconds)
        self._cache: dict[str, tuple[datetime, tuple[float, float]]] = {}

    def get_tick(self, symbols: list[str], now: datetime) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        to_fetch: list[str] = []

        for sym in symbols:
            cached = self._cache.get(sym)
            if cached is None:
                to_fetch.append(sym)
                continue
            ts, val = cached
            if (now - ts).total_seconds() <= self._ttl:
                out[sym] = val
            else:
                to_fetch.append(sym)

        if to_fetch:
            fresh = self._inner.get_tick(to_fetch, now=now)
            for sym, val in fresh.items():
                self._cache[sym] = (now, val)
                out[sym] = val

        return out

    def get_recent_prices(self, symbol: str, window: int) -> list[float]:
        return self._inner.get_recent_prices(symbol, window)


@dataclass
class FallbackNewsProvider(NewsProvider):
    primary: NewsProvider
    fallback: NewsProvider

    def fetch_latest(self, symbol: str, limit: int = 5):
        try:
            items = self.primary.fetch_latest(symbol, limit=limit)
            if items:
                return items
        except Exception:
            pass
        return self.fallback.fetch_latest(symbol, limit=limit)


class AutonomousOrchestratorTest:
    """Run real orchestrator for N minutes with live ticks + AI logging."""

    def __init__(
        self,
        symbols: list[str],
        duration_seconds: int = 900,
        min_trades: int = 6,
        log_file: str = "ai_logs/test_demo_orchestrator_realtime.ndjson",
    ):
        if not symbols:
            raise ValueError("symbols list cannot be empty")

        # Normalize, dedupe, and keep stable order.
        seen: set[str] = set()
        normalized: list[str] = []
        for s in symbols:
            s = str(s).strip()
            if not s:
                continue
            if s not in seen:
                seen.add(s)
                normalized.append(s)
        if not normalized:
            raise ValueError("symbols list cannot be empty")

        self.symbols = normalized
        self._scan_index = 0
        self._active_symbol = self.symbols[0]
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

        bot_logger = logging.getLogger("auraquant.orch_test")
        prices = CachedTickPriceProvider(WeexRestMultiPriceProvider())
        self.prices = prices

        # Prefer real news if CRYPTOPANIC_API_TOKEN is set; else fall back to real-price momentum.
        provider: NewsProvider
        using_cryptopanic = False
        if os.getenv("CRYPTOPANIC_API_TOKEN"):
            try:
                cp = CryptoPanicProvider()
                provider = FallbackNewsProvider(primary=cp, fallback=PriceMomentumNewsProvider(prices=prices))
                using_cryptopanic = True
            except Exception:
                provider = PriceMomentumNewsProvider(prices=prices)
        else:
            provider = PriceMomentumNewsProvider(prices=prices)

        sentiment = SentimentProcessor(logger=bot_logger, provider=provider)
      
        try:
            sentiment.long_threshold = float(os.getenv("SENTIMENT_LONG_THRESHOLD", "0.05"))
        except Exception:
            sentiment.long_threshold = 0.05
        try:
            sentiment.short_threshold = float(os.getenv("SENTIMENT_SHORT_THRESHOLD", "-0.05"))
        except Exception:
            sentiment.short_threshold = -0.05
        # CryptoPanic dev plan is quota-limited and 24h delayed; cache by default.
        if using_cryptopanic:
            try:
                sentiment.news_cache_ttl_minutes = float(os.getenv("SENTIMENT_NEWS_CACHE_TTL_MINUTES", "60"))
            except Exception:
                sentiment.news_cache_ttl_minutes = 60.0
        else:
            sentiment.news_cache_ttl_minutes = 0.0
        correlation = CorrelationTrigger(logger=bot_logger)
        # Lower correlation threshold → more trades approved (default 0.25 → 0.15)
        correlation.corr_threshold = float(os.getenv("CORR_THRESHOLD", "0.15"))
        risk = RiskEngine(logger=bot_logger)
        
        # Cooldown after consecutive losses: 5 min for testing, 45 min for production
        risk.circuit_breaker.cooldown_minutes = int(os.getenv("COOLDOWN_MINUTES", "4"))

        risk.sl_atr_mult = float(os.getenv("SL_ATR_MULT", "3.0"))

        risk.tp_atr_mult = float(os.getenv("TP_ATR_MULT", "6.0"))
        client = WeexPrivateRestClient()
        execution = WeexOrderManager(client=client)

        config = OrchestratorConfig(
            symbol=self.symbols[0],
            tick_seconds=20,
            min_entry_interval_seconds=20,
            enforce_weex_allowlist=True,
        )

        # Match short-run testing needs; can be overridden via env.
        try:
            config.min_confidence = float(os.getenv("MIN_CONFIDENCE", "0.05"))
        except Exception:
            config.min_confidence = 0.05

        correlation.window = 10
        # Ensure we always keep BTC history available (correlation lead).
        correlation.lead_symbol = os.getenv("CORR_LEAD_SYMBOL", correlation.lead_symbol)

        self.execution = execution
        self.correlation = correlation
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
        self._weex_connectivity_checks(symbols=self.symbols)

    def _weex_connectivity_checks(self, symbols: list[str]) -> None:
        logger.info("[WEEX] Connectivity checks per official guide")
        logger.info(f"[WEEX] Symbols: {', '.join(symbols)}")

        # 1) Public ticker
        try:
            base_prices = WeexRestMultiPriceProvider()
            now = datetime.utcnow()
            for s in symbols:
                tick = base_prices.get_tick([s], now=now)
                last, _atr = tick.get(s, (0.0, 0.0))
                logger.info(f"[WEEX] Market ticker OK for {s} (last={last})")
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
                self.execution.set_leverage(symbol=symbols[0], leverage=int(os.getenv("WEEX_LEVERAGE", "2")))
            logger.info("[WEEX] Set leverage OK")
        except Exception as e:
            raise RuntimeError(f"WEEX set leverage failed: {e}")

    def _pick_active_symbol(self) -> str:
        # IMPORTANT:
        # Orchestrator is a multi-phase state machine (SCAN -> QUALIFY -> ENTER -> RECONCILE ...).
        # If we rotate the symbol every tick, we end up doing SCAN on symbol A and then
        # QUALIFY/ENTER on symbol B (because phase advances on each step call).
        # That makes entries extremely unlikely.
        #
        # Fix: only rotate symbols when we're in SCAN *and* flat. Otherwise keep the
        # last chosen active symbol stable until we return to SCAN again.
        pos = self.execution.position()
        if pos is not None and getattr(pos, "symbol", None):
            self._active_symbol = str(pos.symbol)
            return self._active_symbol

        if getattr(self.orchestrator, "phase", None) == BotPhase.SCAN:
            self._active_symbol = self.symbols[self._scan_index % len(self.symbols)]
            self._scan_index += 1

        return self._active_symbol

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

                # Pre-warm history for ALL symbols each tick so correlation can work.
                warm_symbols = list(dict.fromkeys([self.correlation.lead_symbol, *self.symbols]))
                self.prices.get_tick(warm_symbols, now=now)

                active_symbol = self._pick_active_symbol()
                self.orchestrator.config.symbol = active_symbol
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
        default=900,
        help="Duration in seconds (default: 900 = 15 min)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=os.getenv(
            "WEEX_SYMBOLS",
            "BTC/USDT,ETH/USDT,BNB/USDT,XRP/USDT,SOL/USDT,ADA/USDT,DOGE/USDT,LTC/USDT",
        ),
        help="Comma-separated symbols list. Also supports env WEEX_SYMBOLS.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=os.getenv("WEEX_SYMBOL", ""),
        help="(Deprecated) Single symbol override. Prefer --symbols.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=6,
        help="Minimum trades required (default: 6)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="ai_logs/test_demo_orchestrator_realtime.ndjson",
        help="AI log file",
    )
    args = parser.parse_args()

    symbols: list[str]
    if args.symbol:
        symbols = [str(args.symbol).strip()]
    else:
        symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]

    try:
        test = AutonomousOrchestratorTest(
            symbols=symbols,
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
