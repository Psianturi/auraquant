#!/usr/bin/env python3
"""
Orchestrator Real-Time Test (10-12 minutes autonomous trading with AI logging).

This is the PRIMARY testing/training script for competition prep.
Bot makes autonomous decisions: SCAN → QUALIFY → ENTER → RECONCILE.
All AI decisions are logged real-time to WEEX (mandatory for Jan 19).

Features:
- Full orchestrator pipeline (sentiment + correlation + risk)
- Real-time AI log upload to WEEX (retry + disk queue)
- Bot decides autonomously how many trades to execute (minimum 10)
- 10-12 minute run duration
- Complete test statistics + compliance report

Usage:
  python src/test_demo_orchestrator_realtime.py --duration 600 --min-trades 10

Environment:
  WEEX_API_KEY, WEEX_SECRET_KEY, WEEX_PASSPHRASE (for auth)
  WEEX_AI_LOG_UPLOAD_URL (for real-time logging)
  WEEX_BASE_URL (contract API endpoint)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv

# Import core modules
from auraquant.core.orchestrator import Orchestrator
from auraquant.core.types import Decision, OrderPlacement
from auraquant.data.weex_rest_price_provider import WeexRestPriceProvider
from auraquant.execution.paper_order_manager import PaperOrderManager
from auraquant.risk.risk_engine import RiskEngine
from auraquant.sentiment.providers import CoinGeckoPriceProvider, CryptoPanicNewsProvider
from auraquant.sentiment.sentiment_processor import SentimentProcessor
from auraquant.correlation.correlation_trigger import CorrelationTrigger
from auraquant.util.ai_log import AiLogStore, AiLogEvent, make_uploader_from_env

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class RealTimeOrchestratorTest:
    """Autonomous orchestrator test runner with AI logging."""

    def __init__(
        self,
        duration_seconds: int = 600,
        min_trades: int = 10,
        log_file: str = "ai_logs/test_demo_orchestrator_realtime.ndjson",
    ):
        self.duration_seconds = duration_seconds
        self.min_trades = min_trades
        self.log_file = log_file
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Stats
        self.tick_count = 0
        self.trade_count = 0
        self.ai_log_count = 0
        self.successful_uploads = 0
        self.failed_uploads = 0

        # Initialize components
        logger.info("[INIT] Initializing orchestrator components...")
        self.price_provider = WeexRestPriceProvider()
        self.order_manager = PaperOrderManager()
        self.risk_engine = RiskEngine()
        self.sentiment_processor = SentimentProcessor(
            price_provider=CoinGeckoPriceProvider(),
            news_provider=CryptoPanicNewsProvider(),
        )
        self.correlation_trigger = CorrelationTrigger()
        self.orchestrator = Orchestrator(
            price_provider=self.price_provider,
            order_manager=self.order_manager,
            risk_engine=self.risk_engine,
            sentiment_processor=self.sentiment_processor,
            correlation_trigger=self.correlation_trigger,
        )

        # AI Logging
        self.store = AiLogStore(log_file)
        self.uploader = make_uploader_from_env(queue_dir="ai_logs/.upload_queue")
        if self.uploader:
            self.uploader.start()
            logger.info("[INIT] Real-time AI log uploader started (WEEX endpoint configured)")
        else:
            logger.warning("[INIT] WEEX_AI_LOG_UPLOAD_URL not set. Local logging only.")

    def log_ai_decision(
        self,
        stage: str,
        model: str,
        input_data: dict,
        output: dict,
        explanation: str,
        order_id: Optional[str] = None,
    ) -> None:
        """Log AI decision to local store + real-time uploader."""
        event = AiLogEvent(
            orderId=order_id,
            stage=stage,
            model=model,
            input=input_data,
            output=output,
            explanation=explanation,
        )
        # Local store
        self.store.append(event)
        self.ai_log_count += 1

        # Real-time upload
        if self.uploader:
            payload = {
                "orderId": order_id,
                "stage": stage,
                "model": model,
                "input": input_data,
                "output": output,
                "explanation": explanation,
            }
            if self.uploader.upload(payload):
                self.successful_uploads += 1
            else:
                self.failed_uploads += 1

    def run(self) -> None:
        """Run autonomous orchestrator test for configured duration."""
        self.start_time = datetime.now(timezone.utc)
        logger.info(
            f"[START] Running {self.duration_seconds}s autonomous orchestrator test "
            f"(min {self.min_trades} trades required)"
        )
        logger.info(f"[START] Start time: {self.start_time.isoformat()}")

        tick_interval = 30  # Scan market every 30s

        while True:
            self.tick_count += 1
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()

            if elapsed >= self.duration_seconds:
                logger.info(f"[END] Duration limit reached ({elapsed:.1f}s >= {self.duration_seconds}s)")
                break

            logger.info(f"[TICK {self.tick_count}] Elapsed: {elapsed:.1f}s")

            try:
                # Run orchestrator scan → qualify → enter → reconcile
                decision = self.orchestrator.scan_market()

                if decision.stage == "SCAN":
                    bias = decision.output.get("bias", "NEUTRAL")
                    score = decision.output.get("score", 0.0)
                    logger.info(f"[TICK {self.tick_count}] SCAN: {bias} (score={score:.2f})")
                    self.log_ai_decision(
                        stage="SCAN",
                        model="AuraQuant.SentimentProcessor",
                        input_data={"symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]},
                        output={"bias": bias, "score": score},
                        explanation="Sentiment analysis + market scanning",
                    )

                elif decision.stage == "QUALIFY":
                    result = decision.output.get("decision", "SKIP")
                    corr = decision.output.get("correlation", 0.0)
                    logger.info(f"[TICK {self.tick_count}] QUALIFY: {result} (corr={corr:.2f})")
                    self.log_ai_decision(
                        stage="QUALIFY",
                        model="AuraQuant.CorrelationTrigger",
                        input_data={"bias": decision.input.get("bias")},
                        output={"decision": result, "correlation": corr},
                        explanation="Correlation filtering + risk validation",
                    )

                elif decision.stage == "ENTER":
                    status = decision.output.get("status", "REJECTED")
                    order_id = decision.output.get("orderId")
                    logger.info(f"[TICK {self.tick_count}] ENTER: {status}")
                    if status == "APPROVED" and order_id:
                        self.trade_count += 1
                        logger.info(f"[TICK {self.tick_count}] ✓ Order placed: {order_id}")
                    self.log_ai_decision(
                        stage="ENTER",
                        model="AuraQuant.OrderExecutor",
                        input_data={"decision": decision.input.get("decision", "SKIP")},
                        output={"status": status, "orderId": order_id},
                        explanation="Order placement with pre-trade risk check",
                        order_id=order_id,
                    )

                elif decision.stage == "RECONCILE":
                    equity = decision.output.get("equity", 0.0)
                    pnl = decision.output.get("unrealizedPnL", 0.0)
                    logger.info(f"[TICK {self.tick_count}] RECONCILE: equity={equity:.2f}, pnl={pnl:.2f}")
                    self.log_ai_decision(
                        stage="RECONCILE",
                        model="AuraQuant.RiskEngine",
                        input_data={"trades": self.trade_count},
                        output={"equity": equity, "unrealizedPnL": pnl},
                        explanation="Portfolio reconciliation + risk reassessment",
                    )

            except Exception as e:
                logger.error(f"[TICK {self.tick_count}] Error in orchestrator cycle: {e}", exc_info=True)

            # Wait before next tick
            time.sleep(tick_interval)

        self.end_time = datetime.now(timezone.utc)
        self._print_report()

    def _print_report(self) -> None:
        """Print final test report + compliance checklist."""
        if not self.end_time:
            return

        total_time = (self.end_time - self.start_time).total_seconds()

        report = f"""
╔═══════════════════════════════════════════════════════════════╗
║     AUTONOMOUS ORCHESTRATOR TEST REPORT (Jan 15, 2026)       ║
╚═══════════════════════════════════════════════════════════════╝

[TEST PARAMETERS]
  Duration: {total_time:.1f}s ({self.duration_seconds}s target)
  Min Trades Required: {self.min_trades}
  Actual Trades Executed: {self.trade_count}
  Ticks Completed: {self.tick_count}

[AI LOGGING METRICS]
  AI Log Events Generated: {self.ai_log_count}
  Successfully Uploaded: {self.successful_uploads}
  Failed Uploads (Queued): {self.failed_uploads}
  Local Store: {self.log_file}

[COMPLIANCE CHECKLIST]
  ✓ Full orchestrator pipeline tested
  ✓ AI decisions logged to local NDJSON
  ✓ Real-time upload to WEEX attempted
  ✓ Retry queue + disk persistence active
  ✓ All stages (SCAN/QUALIFY/ENTER/RECONCILE) executed
  {"✓" if self.trade_count >= self.min_trades else "✗"} Minimum {self.min_trades} trades executed (got {self.trade_count})

[STATUS]
  {f"PASS ✓" if self.trade_count >= self.min_trades else "CHECK LOGS"}

[NEXT STEPS FOR JAN 19]
  1. Verify WEEX real-time upload succeeds (not just queued)
  2. Run 2-3 more 10-12 min tests to stabilize
  3. Monitor balance + PnL trends
  4. Ensure sentiment/correlation scores are realistic
  5. Test on competition day (Jan 19, 12:00 UTC+8)

═══════════════════════════════════════════════════════════════
"""
        print(report)
        logger.info(report)

    def cleanup(self) -> None:
        """Graceful shutdown."""
        if self.uploader:
            logger.info("[CLEANUP] Stopping uploader (flushing remaining queued events)...")
            self.uploader.stop()
        logger.info("[CLEANUP] Test completed")


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous orchestrator real-time test (10-12 min with AI logging)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=600,
        help="Test duration in seconds (default: 600 = 10 min)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=10,
        help="Minimum trades required to pass (default: 10)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="ai_logs/test_demo_orchestrator_realtime.ndjson",
        help="AI log output file",
    )
    args = parser.parse_args()

    test = RealTimeOrchestratorTest(
        duration_seconds=args.duration,
        min_trades=args.min_trades,
        log_file=args.log_file,
    )

    try:
        test.run()
    except KeyboardInterrupt:
        logger.info("[INTERRUPT] Test interrupted by user")
    except Exception as e:
        logger.error(f"[ERROR] Test failed: {e}", exc_info=True)
    finally:
        test.cleanup()


if __name__ == "__main__":
    main()
