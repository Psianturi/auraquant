#!/usr/bin/env python3
"""Simple test for AI log real-time upload flow."""

import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from auraquant.util.ai_log.realtime_uploader import make_uploader_from_env

load_dotenv()


def main():
    uploader = make_uploader_from_env()
    if uploader is None:
        print("⚠️  WEEX_AI_LOG_UPLOAD_URL not set. Skipping upload test.")
        return

    print("Starting real-time uploader...")
    uploader.start()

    # Simulate AI decisions
    events = [
        {
            "stage": "SCAN",
            "model": "SentimentProcessor",
            "input": {"symbol": "SOL/USDT"},
            "output": {"bias": "LONG", "confidence": 0.85},
            "explanation": "Positive sentiment detected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        {
            "stage": "QUALIFY",
            "model": "CorrelationTrigger",
            "input": {"symbol": "SOL/USDT", "bias": "LONG"},
            "output": {"signal": "APPROVED", "correlation": 0.92},
            "explanation": "BTC lead-lag confirms entry",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        {
            "stage": "ENTER",
            "model": "RiskEngine",
            "input": {"symbol": "SOL/USDT", "side": "BUY"},
            "output": {"approved": True, "position_size": 0.5},
            "explanation": "Risk gate passed",
            "orderId": 12345,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    ]

    print(f"Pushing {len(events)} AI log events...")
    for event in events:
        uploader.upload(event)
        print(f"  ✓ {event['stage']}")
        time.sleep(0.2)

    print("Waiting for flush (30s max)...")
    time.sleep(35)

    uploader.stop()
    print("✅ Upload test complete. Check server logs for success/failure.")


if __name__ == "__main__":
    main()
