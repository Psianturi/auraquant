# AuraQuant

**A Hybrid Multi-Agent AI Trading System**

AuraQuant is an AI-driven trading agent integrating NLP sentiment analysis, cross-asset correlation modeling, and dynamic risk management.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                           │
│         (State Machine: SCAN→QUALIFY→ENTER→MANAGE→          │
│                    EXIT→RECONCILE)                          │
└──────────────┬──────────────┬──────────────┬───────────────┘
               │              │              │
    ┌──────────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
    │   LAYER A       │ │  LAYER B  │ │   LAYER C   │
    │   Sentiment     │ │Correlation│ │ RiskEngine  │
    │   (The Brain)   │ │(Trigger)  │ │(Guardian)   │
    └─────────────────┘ └───────────┘ └─────────────┘
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                    ┌─────────────────┐
                    │    EXECUTION    │
                    │  (Paper / WEEX) │
                    └─────────────────┘
```

### Three Pillars

| Layer | Role | Output |
|-------|------|--------|
| **A - Sentiment** | Analyze news with half-life decay | `LONG / SHORT / NEUTRAL` |
| **B - Correlation** | Confirm entry via BTC lead-lag | `CorrelationSignal` |
| **C - Risk** | Final gate: sizing, SL/TP, circuit breaker | `APPROVED / DENIED` |

## Key Features

- **State Machine** - Prevents order spam and race conditions
- **Circuit Breaker** - Auto-stop on daily drawdown or loss streak
- **ATR-based SL/TP** - Dynamic stops based on volatility
- **Online Learning** - Adaptive P(win) estimation from trade outcomes
- **AI Log Store** - NDJSON format for audit trail

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and configure your API credentials:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
- WEEX API credentials (API key, secret, passphrase)
- CryptoPanic API token for news sentiment
- Google Gemini API key for NLP analysis (optional)

### Running the System

**Paper Trading Demo:**
```bash
python src/demo_orchestrator.py
```

**Test WEEX API Connection:**
```bash
python src/test_weex_real.py
```

**Test CryptoPanic News Integration:**
```bash
python src/cryptopanic_tester.py
```

**Mock Trading Simulation:**
```bash
python src/mock_weex_tester.py
```

### System Validation

```bash
python src/system_validation.py
```

This will run comprehensive tests on all components and generate performance reports.

## Project Structure

```
src/auraquant/
├── core/           # Orchestrator, state machine
├── sentiment/      # SentimentProcessor, NewsProvider
├── correlation/    # CorrelationTrigger (BTC lead-lag)
├── risk/           # RiskEngine, CircuitBreaker
├── execution/      # PaperOrderManager
├── learning/       # OnlineLogistic P(win) model
├── data/           # Price providers (Static, WEEX REST)
├── weex/           # Symbol allowlist
└── util/           # JSON logging, AI log store
```


## Configuration

### Trading Parameters

**OrchestratorConfig:**
- `symbol` - Target trading pair (e.g., "SOL/USDT")
- `default_leverage` - Leverage multiplier (default: 10.0)
- `min_confidence` - Minimum confidence threshold for trade entry
- `min_entry_interval_seconds` - Cooldown between trades
- `tick_seconds` - Time interval for each trading cycle

**RiskEngine:**
- `daily_drawdown_limit_pct` - Maximum daily loss before circuit breaker
- `max_consecutive_losses` - Loss streak limit before cooldown
- `risk_per_trade_pct` - Position size as percentage of equity
- `sl_atr_mult` - Stop loss multiplier based on ATR
- `tp_atr_mult` - Take profit multiplier based on ATR

### Environment Variables

See `.env.example` for complete list of configuration options.

## AI Decision Logging

All AI decisions are logged in NDJSON format for transparency and audit:

```json
{
  "stage": "QUALIFY",
  "model": "AuraQuant.CorrelationTrigger",
  "input": {"symbol": "SOL/USDT", "bias": "LONG"},
  "output": {"correlation": 0.93, "signal": "APPROVED"},
  "explanation": "Cross-asset correlation confirmation",
  "timestamp": "2026-01-13T14:00:00Z"
}
```

Logs are stored in `ai_logs/` directory.

## How CoinGecko API Is Used (CoinGecko Track)

AuraQuant can optionally use CoinGecko API to improve **market discovery** and **strategy workflow**
without changing WEEX execution correctness.

What we use CoinGecko for:
- **Universe selection** across the 8 allowed WEEX competition pairs (BTC/ETH/SOL/DOGE/XRP/ADA/BNB/LTC).
- **Liquidity-aware selection** using CoinGecko market snapshots (24h volume + 24h move) to pick the most tradable asset.
- **Research-ready snapshots** cached to disk so the bot can survive CoinGecko rate limits/outages (stale cache fallback).

Implementation:
- CoinGecko REST client + disk cache: [src/auraquant/data/coingecko_client.py](src/auraquant/data/coingecko_client.py)
- Smoke test / demo script: [src/coingecko_smoketest.py](src/coingecko_smoketest.py)

Endpoints used:
- `GET /api/v3/coins/markets` (CoinGecko) for market discovery & selection

Configuration (never commit keys):
- `COINGECKO_API_KEY` (Analyst plan key if issued)
- `COINGECKO_BASE_URL` (optional override; if `COINGECKO_API_KEY` is set, defaults to Pro `https://pro-api.coingecko.com/api/v3`)
- `COINGECKO_CACHE_DIR` (defaults to `runtime_cache/coingecko`)

Run locally:
- `python src/coingecko_smoketest.py`

## WEEX API Testing Checklist (Official Flow)

Use this script for the official API test flow + evidence logging:

- `python src/test_weex_real.py`


### 1) Dry-run first (no real orders)

```bash
python src/test_weex_real.py --log-file ai_logs/weex_api_test.log
```

### 2) Execute the required trades (opt-in)

Default order mode is `market_notional` and targets ~10 USDT notional.

Linux/macOS:

```bash
WEEX_EXECUTE_ORDER=1 WEEX_TRADE_REPEAT=10 WEEX_TRADE_SLEEP=1 \
python src/test_weex_real.py --log-file ai_logs/weex_api_test_exec.log
```

Windows PowerShell:

```powershell
$env:WEEX_EXECUTE_ORDER="1"
$env:WEEX_TRADE_REPEAT="10"
$env:WEEX_TRADE_SLEEP="1"
python src/test_weex_real.py --log-file ai_logs/weex_api_test_exec.log
```

### 3) What to verify in the log

- `/capi/v2/products` succeeded (or fallback symbol used)
- `/capi/v2/account/assets` returns OK (test funds visible)
- `/capi/v2/account/leverage` returns OK
- `/capi/v2/order/placeOrder` returns `200` and includes `order_id`
- `/capi/v2/order/fills` shows fills for each `order_id`

Tip: if a symbol is rejected, follow the script guidance to use the exact symbol returned by `/products`.

## Performance Monitoring

The system tracks:
- Equity curve and PnL
- Win rate and Sharpe ratio
- Drawdown metrics
- Trade execution statistics
- Model learning progress

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.




