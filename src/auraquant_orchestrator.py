#!/usr/bin/env python3


import argparse
import logging
import os
import sys
import time
import atexit
import signal
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
from auraquant.learning import TradePolicyLearner
from auraquant.risk.risk_engine import RiskEngine
from auraquant.sentiment.providers import CoinGeckoMomentumProvider, CoinGeckoTrendingProvider, CryptoPanicProvider, NewsProvider
from auraquant.sentiment.sentiment_processor import SentimentProcessor
from auraquant.sentiment.types import NewsItem
from auraquant.util.ai_log.store import AiLogEvent, AiLogStore
from auraquant.util.ai_log.realtime_uploader import make_uploader_from_env

from auraquant.weex.private_client import WeexPrivateRestClient
from auraquant.weex.symbols import to_weex_contract_symbol
from auraquant.weex.emergency_close import emergency_close


REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=REPO_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


PIDFILE_PATH = REPO_ROOT / "runtime_cache" / "orchestrator.pid"


def _write_pidfile() -> None:
    PIDFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE_PATH.write_text(str(os.getpid()), encoding="utf-8")


def _remove_pidfile() -> None:
    try:
        if PIDFILE_PATH.exists():
            PIDFILE_PATH.unlink()
    except Exception:
        pass


def _stop_running_bot_best_effort(timeout_seconds: float = 5.0) -> bool:
    """Stop a previously started runner instance (best-effort).

    Uses a pidfile written by this runner during normal runs.
    """
    try:
        if not PIDFILE_PATH.exists():
            return False
        raw = PIDFILE_PATH.read_text(encoding="utf-8").strip()
        if not raw:
            return False
        pid = int(raw)
        if pid <= 0 or pid == os.getpid():
            return False

        try:
            os.kill(pid, 0)
        except Exception:
            return False

        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            return False

        deadline = time.time() + float(timeout_seconds)
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except Exception:
                return True
            time.sleep(0.1)

        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
        return True
    except Exception:
        return False


@dataclass
class PriceMomentumNewsProvider(NewsProvider):

    prices: MultiPriceProvider

    def fetch_latest(self, symbol: str, limit: int = 5):

        # `symbol` is expected to be the internal pair format (e.g. "BTC/USDT").
        internal = symbol if "/" in symbol else f"{symbol}/USDT"
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        recent = self.prices.get_recent_prices(internal, window=12)
        bias = "flat"
        ret = 0.0
        if len(recent) >= 2 and recent[0] > 0:
            ret = (recent[-1] - recent[0]) / max(recent[0], 1e-12)
            # Aggressive: 0.03% threshold for more trade signals
            threshold = float(os.getenv("MOMENTUM_THRESHOLD", "0.0003"))
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
        log_file: str = "ai_logs/auraquant_orchestrator.ndjson",
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

        self._hold_anchor_at: datetime | None = None
        self._hold_anchor_symbol: str | None = None
        self._hold_limit_seconds: int | None = None

        self.store = AiLogStore(log_file)
        self.uploader = make_uploader_from_env(queue_dir="ai_logs/.upload_queue")
        if self.uploader is not None:
            self.uploader.start()
            logger.info("[INIT] Real-time uploader enabled")
        else:
            logger.info("[INIT] Real-time uploader disabled (WEEX_AI_LOG_UPLOAD_URL not set)")

        # Monkeypatch store.append to also push to uploader, without changing core orchestrator.
        orig_append = self.store.append

        try:
            ai_log_throttle_seconds = float(os.getenv("AI_LOG_THROTTLE_SECONDS", "120"))
        except Exception:
            ai_log_throttle_seconds = 120.0
        throttle_stages_env = os.getenv("AI_LOG_THROTTLE_STAGES", "SCAN,QUALIFY")
        throttle_stages = {s.strip().upper() for s in str(throttle_stages_env).split(",") if s.strip()}
        _last_stage_emit_ts: dict[str, float] = {}

        def _append_and_upload(event):
            try:
                stage = str(getattr(event, "stage", "")).upper().strip()
                if ai_log_throttle_seconds > 0 and stage in throttle_stages:
                    ts = getattr(event, "timestamp", None)
                    now_ts = time.time()
                    if ts is not None:
                        try:
                            now_ts = float(ts.timestamp())
                        except Exception:
                            now_ts = time.time()
                    last_ts = float(_last_stage_emit_ts.get(stage, 0.0))
                    if last_ts > 0 and (now_ts - last_ts) < float(ai_log_throttle_seconds):
                        return
                    _last_stage_emit_ts[stage] = float(now_ts)
            except Exception:
                pass

            orig_append(event)
            self.ai_log_count += 1
            if self.uploader is not None:
                ok = bool(self.uploader.upload(event.to_payload()))
                if ok:
                    self.upload_success += 1
                else:
                    self.upload_failed += 1

        self.store.append = _append_and_upload  

        bot_logger = logging.getLogger("auraquant.orch_test")
        prices = CachedTickPriceProvider(WeexRestMultiPriceProvider())
        self.prices = prices

        # Prefer real news if CRYPTOPANIC_API_TOKEN is set; fall back to CoinGecko Trending; then Momentum; then real-price.
        provider: NewsProvider
        using_cryptopanic = False
        using_coingecko = False
        provider_chain_desc: list[str] = []

        # Build fallback chain from bottom up:
        # PriceMomentum -> CoinGeckoMomentum -> CoinGeckoTrending -> CryptoPanic
        base_fallback: NewsProvider = PriceMomentumNewsProvider(prices=prices)
        provider_chain_desc.append("PriceMomentum")
        
        cg_api_key = os.getenv("COINGECKO_API_KEY")
        if cg_api_key or os.getenv("COINGECKO_BASE_URL"):
            # CoinGecko Momentum as fallback
            base_fallback = FallbackNewsProvider(primary=CoinGeckoMomentumProvider(), fallback=base_fallback)
            provider_chain_desc.insert(0, "CoinGeckoMomentum")
            # CoinGecko Trending as higher priority fallback (requires API key)
            if cg_api_key:
                base_fallback = FallbackNewsProvider(primary=CoinGeckoTrendingProvider(), fallback=base_fallback)
                provider_chain_desc.insert(0, "CoinGeckoTrending")
            using_coingecko = True

        if os.getenv("CRYPTOPANIC_API_TOKEN"):
            try:
                cp = CryptoPanicProvider()
                provider = FallbackNewsProvider(primary=cp, fallback=base_fallback)
                provider_chain_desc.insert(0, "CryptoPanic")
                using_cryptopanic = True
            except Exception:
                provider = base_fallback
        else:
            provider = base_fallback

        logger.info(f"[INIT] News provider chain: {' -> '.join(provider_chain_desc)}")
        sentiment = SentimentProcessor(logger=bot_logger, provider=provider)
      
        try:
            sentiment.long_threshold = float(os.getenv("SENTIMENT_LONG_THRESHOLD", "0.05"))
        except Exception:
            sentiment.long_threshold = 0.05
        try:
            sentiment.short_threshold = float(os.getenv("SENTIMENT_SHORT_THRESHOLD", "-0.05"))
        except Exception:
            sentiment.short_threshold = -0.05
        # Cache network-backed sentiment sources to avoid API spam.
        if using_cryptopanic:
            try:
                sentiment.news_cache_ttl_minutes = float(os.getenv("SENTIMENT_NEWS_CACHE_TTL_MINUTES", "60"))
            except Exception:
                sentiment.news_cache_ttl_minutes = 60.0
        elif using_coingecko:
            try:
                sentiment.news_cache_ttl_minutes = float(os.getenv("SENTIMENT_NEWS_CACHE_TTL_MINUTES", "5"))
            except Exception:
                sentiment.news_cache_ttl_minutes = 5.0
        else:
            sentiment.news_cache_ttl_minutes = 0.0
        correlation = CorrelationTrigger(logger=bot_logger)
       
        correlation.corr_threshold = float(os.getenv("CORR_THRESHOLD", "0.50"))
        risk = RiskEngine(logger=bot_logger)
        
        risk.circuit_breaker.cooldown_minutes = int(os.getenv("COOLDOWN_MINUTES", "1"))
        try:
            risk.max_position_notional_pct = float(os.getenv("MAX_POSITION_NOTIONAL_PCT", "4.0"))
        except Exception:
            risk.max_position_notional_pct = 4.0
        risk.sl_atr_mult = float(os.getenv("SL_ATR_MULT", "2.85"))  
        risk.tp_atr_mult = float(os.getenv("TP_ATR_MULT", "2.4"))
        client = WeexPrivateRestClient()
        execution = WeexOrderManager(client=client)

        config = OrchestratorConfig(
            symbol=self.symbols[0],
            tick_seconds=20,
            min_entry_interval_seconds=int(os.getenv("MIN_ENTRY_INTERVAL_SECONDS", "10")),
            enforce_weex_allowlist=True,
        )

        # Match short-run testing needs; can be overridden via env.
        try:
 
            config.min_confidence = float(os.getenv("MIN_CONFIDENCE", "0.06"))
        except Exception:
            config.min_confidence = 0.06

        correlation.window = 10
        # Ensure we always keep BTC history available (correlation lead).
        correlation.lead_symbol = os.getenv("CORR_LEAD_SYMBOL", correlation.lead_symbol)

        self.execution = execution
        self.correlation = correlation
        learner = None
        if os.getenv("ENABLE_LEARNER", "0") == "1":
            model_path = os.getenv("LEARNER_MODEL_PATH", "models/trade_policy.json")
            learner = TradePolicyLearner(model_path=model_path)
            logger.info(f"[INIT] Online learner enabled (model={model_path})")

        self.orchestrator = Orchestrator(
            logger=bot_logger,
            config=config,
            sentiment=sentiment,
            correlation=correlation,
            risk=risk,
            prices=prices,
            execution=execution,
            learner=learner,
            ai_log_store=self.store,
            ai_log_uploader=self.uploader,
        )

        self._weex_connectivity_checks(symbols=self.symbols)

    def _weex_connectivity_checks(self, symbols: list[str]) -> None:
        logger.info("[WEEX] Connectivity checks per official guide")
        logger.info(f"[WEEX] Symbols: {', '.join(symbols)}")

        # 1) Public ticker
        try:
            base_prices = WeexRestMultiPriceProvider()
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            for s in symbols:
                tick = base_prices.get_tick([s], now=now)
                last, _atr = tick.get(s, (0.0, 0.0))
                logger.info(f"[WEEX] Market ticker OK for {s} (last={last})")
        except Exception as e:
            raise RuntimeError(f"WEEX public ticker failed: {e}")

        # 2) Private assets
        try:
            self.execution.reconcile(now=datetime.now(timezone.utc).replace(tzinfo=None))
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
        if self.duration_seconds and self.duration_seconds > 0:
            logger.info(f"[START] Running {self.duration_seconds}s autonomous orchestrator test")
        else:
            logger.info("[START] Running autonomous orchestrator (no duration limit)")
        logger.info(f"[START] Min trades required: {self.min_trades}")
        logger.info(f"[START] Start time: {self.start_time.isoformat()}")

        tick_interval = int(getattr(self.orchestrator.config, "tick_seconds", 20) or 20)

        last_progress_log = 0  # Track last progress log time (minutes)
        last_max_hold_close_attempt_ts = 0.0
        try:
            max_hold_retry_interval_seconds = float(
                os.getenv("MAX_HOLD_RETRY_INTERVAL_SECONDS", "60")
            )
        except Exception:
            max_hold_retry_interval_seconds = 60.0
        while True:
            self.tick_count += 1
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            elapsed_minutes = int(elapsed // 60)

            # Log progress every 15 minutes
            if elapsed_minutes > 0 and elapsed_minutes % 15 == 0 and elapsed_minutes != last_progress_log:
                logger.info(f"[PROGRESS] Agent running {elapsed_minutes} minutes")
                last_progress_log = elapsed_minutes

            if self.duration_seconds and self.duration_seconds > 0 and elapsed >= self.duration_seconds:
                logger.info(f"[END] Duration limit ({elapsed:.1f}s >= {self.duration_seconds}s)")
                break

            try:
                now = datetime.now(timezone.utc).replace(tzinfo=None) 

                warm_symbols = list(dict.fromkeys([self.correlation.lead_symbol, *self.symbols]))
                warm_ticks = self.prices.get_tick(warm_symbols, now=now)

                pos = self.execution.position()
                max_hold_seconds = int(os.getenv("MAX_HOLD_SECONDS", "1800"))
                if pos is not None and max_hold_seconds > 0:

                    if self._hold_anchor_symbol != str(pos.symbol) or self._hold_anchor_at is None:
                        self._hold_anchor_symbol = str(pos.symbol)
                        self._hold_anchor_at = pos.opened_at
                        self._hold_limit_seconds = None

                    anchor = self._hold_anchor_at or pos.opened_at
                    held_for = (now - anchor).total_seconds()

                    hold_limit_seconds = int(self._hold_limit_seconds) if self._hold_limit_seconds else int(max_hold_seconds)

                    last_price = None
                    try:
                        last_price = float((warm_ticks.get(str(pos.symbol)) or (0.0, 0.0))[0])
                    except Exception:
                        last_price = None

                    pnl_pct = None
                    try:
                        entry = max(float(getattr(pos, "entry_price", 0.0) or 0.0), 1e-12)
                        if last_price is not None and last_price > 0 and entry > 0:
                            if str(getattr(pos, "side", "")).upper() == "SHORT":
                                pnl_pct = (entry - float(last_price)) / entry * 100.0
                            else:
                                pnl_pct = (float(last_price) - entry) / entry * 100.0
                    except Exception:
                        pnl_pct = None

                    # Optional: close immediately on any profit (can be noisy due to fees).
                    close_on_any_profit = os.getenv("CLOSE_ON_ANY_PROFIT", "0") == "1"
                    if close_on_any_profit and pnl_pct is not None and pnl_pct > 0:
                        held_for = max(held_for, float(hold_limit_seconds))

                    if held_for >= hold_limit_seconds:
                        max_hold_loss_extend = os.getenv("MAX_HOLD_LOSS_EXTEND", "0") == "1"
                        max_hold_profit_force_close = os.getenv("MAX_HOLD_PROFIT_FORCE_CLOSE", "1") == "1"

                        if pnl_pct is not None and pnl_pct <= 0 and max_hold_loss_extend:
                            # Extend hold window: wait another configured window before trying again.
                            try:
                                extend_seconds = int(os.getenv("MAX_HOLD_LOSS_EXTEND_SECONDS", "600"))
                            except Exception:
                                extend_seconds = 600
                            if extend_seconds <= 0:
                                extend_seconds = 600
                            self._hold_anchor_at = now
                            self._hold_limit_seconds = int(extend_seconds)
                            logger.warning(
                                f"[MANAGE] Max hold exceeded but in loss (pnl_pct={pnl_pct:.3f}%). "
                                f"Extending hold window by {extend_seconds}s."
                            )
                            continue

                        close_reason = "MAX_HOLD"
                        if pnl_pct is not None and pnl_pct > 0 and max_hold_profit_force_close:
                            close_reason = "MAX_HOLD_PROFIT"

                        if hasattr(self.execution, "close_open_position_best_effort"):
                            logger.warning(
                                f"[MANAGE] Max hold exceeded ({held_for:.0f}s >= {hold_limit_seconds}s). "
                                "Attempting best-effort close."
                            )
                            equity_before = None
                            try:
                                equity_before = float(self.execution.equity())
                            except Exception:
                                equity_before = None
                            now_ts = time.time()
                            if now_ts - last_max_hold_close_attempt_ts >= max_hold_retry_interval_seconds:
                                last_max_hold_close_attempt_ts = now_ts
                                try:
                                    self.execution.close_open_position_best_effort()
                                except Exception as exc:
                                    logger.warning(f"[MANAGE] Max-hold close failed: {exc}")
                                pnl = None
                                try:
                                    self.execution.reconcile(now=now)
                                    if equity_before is not None:
                                        pnl = float(self.execution.equity()) - float(equity_before)
                                except Exception:
                                    pnl = None

                                try:
                                    if self.store is not None:
                                        self.store.append(
                                            AiLogEvent(
                                                stage="TRADE_CLOSED",
                                                model=f"AuraQuant.{type(self.execution).__name__}",
                                                input={
                                                    "symbol": str(getattr(pos, "symbol", "")),
                                                    "pnl_usdt": None if pnl is None else round(float(pnl), 6),
                                                    "reason": close_reason,
                                                    "held_for_seconds": round(float(held_for), 2),
                                                    "unrealized_pnl_pct": None if pnl_pct is None else round(float(pnl_pct), 4),
                                                    "hold_limit_seconds": int(hold_limit_seconds),
                                                },
                                                output={
                                                    "equity_now": round(float(self.execution.equity()), 6),
                                                    "trade_count": int(self.execution.trade_count()),
                                                },
                                                explanation=(
                                                    "Force close due to MAX_HOLD_SECONDS." if close_reason == "MAX_HOLD" else
                                                    "Force close on profit at MAX_HOLD_SECONDS."
                                                ),
                                                timestamp=now,
                                            )
                                        )
                                except Exception as e:
                                    logger.debug(f"[AI LOG] Failed to append max-hold TRADE_CLOSED: {e}")
                            else:
                                logger.warning(
                                    "[MANAGE] Max-hold close throttled; waiting before retry"
                                )
                        else:
                            logger.warning("[MANAGE] Max hold exceeded but execution has no close method.")
                else:

                    self._hold_anchor_at = None
                    self._hold_anchor_symbol = None
                    self._hold_limit_seconds = None

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

        status = "PASS" if self.trade_count >= self.min_trades else "FAIL"
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
                    equity_before = None
                    try:
                        equity_before = float(self.execution.equity())
                    except Exception:
                        equity_before = None
                    closer()

                    pnl = None
                    try:
                        self.execution.reconcile(now=datetime.now(timezone.utc).replace(tzinfo=None))
                        if equity_before is not None:
                            pnl = float(self.execution.equity()) - float(equity_before)
                    except Exception:
                        pnl = None

                    try:
                        now_ts = datetime.now(timezone.utc).replace(tzinfo=None)
                        if self.store is not None:
                            self.store.append(
                                AiLogEvent(
                                    stage="TRADE_CLOSED",
                                    model=f"AuraQuant.{type(self.execution).__name__}",
                                    input={
                                        "symbol": str(getattr(pos, "symbol", "")),
                                        "pnl_usdt": None if pnl is None else round(float(pnl), 6),
                                        "reason": "CLEANUP",
                                    },
                                    output={
                                        "equity_now": round(float(self.execution.equity()), 6),
                                        "trade_count": int(self.execution.trade_count()),
                                    },
                                    explanation="Force close during runner cleanup.",
                                    timestamp=now_ts,
                                )
                            )
                    except Exception as e:
                        logger.debug(f"[AI LOG] Failed to append cleanup TRADE_CLOSED: {e}")
        except Exception:
            logger.exception("[CLEANUP] Failed to close open position (continuing shutdown)")

        if self.uploader:
            logger.info("[CLEANUP] Stopping uploader (flushing queued events)...")
            self.uploader.stop()
        logger.info("[CLEANUP] Test completed")


def main():
    parser = argparse.ArgumentParser(
        description="AuraQuant orchestrator real-time runner (with AI logging)"
    )
    parser.add_argument(
        "--emergency-close",
        action="store_true",
        help="Stop any running bot (via pidfile), disable Gemini calls, and force close positions/cancel orders, then exit.",
    )
    parser.add_argument(
        "--forever",
        action="store_true",
        help="Run without duration limit (equivalent to --duration 0).",
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
        default="ai_logs/auraquant_orchestrator.ndjson",
        help="AI log file",
    )
    args = parser.parse_args()

    if args.forever:
        args.duration = 0

    symbols: list[str]
    if args.symbol:
        symbols = [str(args.symbol).strip()]
    else:
        symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]

    if args.emergency_close:
        # Ensure we do not call Gemini while emergency handling.
        os.environ["DISABLE_GEMINI_API"] = "1"
        os.environ["USE_GEMINI_SENTIMENT"] = "0"

        stopped = _stop_running_bot_best_effort(timeout_seconds=8.0)
        logger.warning(f"[EMERGENCY] stop_running_bot={stopped} pidfile={str(PIDFILE_PATH)}")

        client = WeexPrivateRestClient()
        client.require_env()
        weex_symbols = [to_weex_contract_symbol(s) for s in symbols]
        result = emergency_close(client=client, weex_symbols=weex_symbols, sleep_seconds=2.0)
        remaining = result.get("remaining_positions") or []

        if remaining:
            logger.error(f"[EMERGENCY] Remaining positions after close: {remaining}")
            raise SystemExit(2)

        logger.warning("[EMERGENCY] OK: no open positions remain (per API)")
        raise SystemExit(0)

    try:
        _write_pidfile()
        atexit.register(_remove_pidfile)
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
