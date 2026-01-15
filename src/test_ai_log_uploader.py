#!/usr/bin/env python3
"""Test AI log real-time uploader (dry-run, no actual trades).

Usage:
  python src/test_ai_log_uploader.py --help

Environment:
  WEEX_AI_LOG_UPLOAD_URL (required, e.g., https://api.weex.com/.../uploadAiLog)
  WEEX_AI_LOG_AUTH_HEADER (optional, e.g., "Authorization:Bearer TOKEN")
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auraquant.util.ai_log import AiLogStore, AiLogEvent, RealTimeAiLogUploader, make_uploader_from_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test real-time AI log uploader")
    parser.add_argument(
        "--num-events",
        type=int,
        default=5,
        help="Number of test events to upload",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Delay between uploads (seconds)",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Only test local storage (no real upload)",
    )
    args = parser.parse_args()

    # Create local store (always)
    store_path = "ai_logs/test_uploader.ndjson"
    store = AiLogStore(store_path)

    # Create uploader (if WEEX_AI_LOG_UPLOAD_URL is set and not --skip-upload)
    uploader: RealTimeAiLogUploader | None = None
    if not args.skip_upload:
        uploader = make_uploader_from_env()
        if uploader is None:
            logger.warning("[TEST] WEEX_AI_LOG_UPLOAD_URL not set. Skipping real-time upload (local storage only).")
        else:
            uploader.start()
            logger.info("[TEST] Real-time uploader started")

    try:
        logger.info(f"[TEST] Generating {args.num_events} test events...")

        for i in range(args.num_events):
            now = datetime.now(timezone.utc)

            # Generate synthetic event (simulating bot decision)
            event = AiLogEvent(
                stage="QUALIFY" if i % 2 == 0 else "ENTER",
                model="AuraQuant.TestCorrelation",
                input={
                    "symbol": "SOL/USDT",
                    "event_index": i,
                    "test_timestamp": now.isoformat(),
                },
                output={
                    "decision": "APPROVED" if i % 3 == 0 else "REJECTED",
                    "confidence": round(0.5 + (i % 5) * 0.1, 2),
                    "event_index": i,
                },
                explanation=f"Test event #{i} for compliance validation",
                order_id=None,
                timestamp=now,
            )

            # Store locally
            store.append(event)
            logger.info(f"[TEST] Stored locally: {event.stage} event #{i}")

            # Push to uploader if available
            if uploader is not None:
                payload = event.to_payload()
                success = uploader.upload(payload)
                status = "pushed (will retry if needed)" if not success else "uploaded"
                logger.info(f"[TEST] {status}: {event.stage} event #{i}")

            if i < args.num_events - 1:
                time.sleep(args.delay_seconds)

        logger.info(f"[TEST] All {args.num_events} events generated")
        logger.info(f"[TEST] Local storage: {store_path}")

        # Show local records
        print("\n" + "=" * 70)
        print("LOCAL RECORDS (ai_logs/test_uploader.ndjson)")
        print("=" * 70)
        try:
            with open(store_path, "r") as f:
                for line_no, line in enumerate(f, 1):
                    try:
                        rec = json.loads(line)
                        print(f"[{line_no}] {json.dumps(rec, ensure_ascii=False)[:120]}...")
                    except Exception as e:
                        logger.warning(f"Failed to parse line {line_no}: {e}")
        except FileNotFoundError:
            logger.error(f"Store file not found: {store_path}")

        if uploader is not None:
            print("\n" + "=" * 70)
            print("REAL-TIME UPLOAD STATUS")
            print("=" * 70)
            print(f"Upload URL: {uploader.weex_upload_url}")
            print(f"Max retries: {uploader.max_retries}")
            print(f"Retry delay: {uploader.retry_delay_seconds}s")
            print(f"Flush interval: {uploader.flush_interval_seconds}s")

            # Wait a bit for background flushes
            print("\n[TEST] Waiting for background retry thread to process...")
            time.sleep(5)

            # Check if queue has failed entries
            failed_count = len(uploader.queue.pop_all())
            if failed_count > 0:
                logger.warning(f"[TEST] {failed_count} events still in retry queue (check network)")
            else:
                logger.info("[TEST] All events processed (or uploaded)")

    finally:
        if uploader is not None:
            logger.info("[TEST] Stopping uploader...")
            uploader.stop(timeout_seconds=5)
            logger.info("[TEST] Uploader stopped")

    print("\n" + "=" * 70)
    print("COMPLIANCE NOTES")
    print("=" * 70)
    print("""
1. Local store (ai_logs/) proves AI decisions are captured.
2. Real-time uploader (if configured) pushes to WEEX with <1min delay.
3. On network error, events are queued to ai_logs/.upload_queue/ for retry.
4. Before competition, verify:
   - WEEX_AI_LOG_UPLOAD_URL is set correctly
   - Auth header (if needed) is valid
   - No credentials leaked in logs
   - All required fields present (stage, model, input, output, explanation)

5. During competition:
   - Real orchestrator uses _push_ai_log() to auto-upload all decisions
   - WEEX dashboard should show real-time event stream
""")


if __name__ == "__main__":
    sys.exit(main())
