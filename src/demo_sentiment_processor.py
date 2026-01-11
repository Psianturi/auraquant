from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auraquant.sentiment import NewsItem, SentimentProcessor, StaticNewsProvider


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("AuraQuant")

    now = datetime.utcnow()
    items = [
        NewsItem(
            title="Institutional inflows into BTC ETFs surge",
            published_at=now - timedelta(minutes=5),
            source="demo",
            url="https://example.com/a",
        ),
        NewsItem(
            title="Major exchange suffers exploit; users fear liquidation cascade",
            published_at=now - timedelta(minutes=25),
            source="demo",
            url="https://example.com/b",
        ),
        NewsItem(
            title="Solana TVL hits new ATH as ecosystem growth accelerates",
            published_at=now - timedelta(minutes=12),
            source="demo",
            url="https://example.com/c",
        ),
    ]

    provider = StaticNewsProvider(items=items)
    sp = SentimentProcessor(logger=logger, provider=provider, half_life_minutes=30.0)

    report = sp.analyze(symbol="BTC", limit=5, now=now)
    print("Bias:", report.bias)
    print("Score:", round(report.score, 4))
    print("Top headline:", report.top_headline)


if __name__ == "__main__":
    main()
