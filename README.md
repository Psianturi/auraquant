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
- CoinGecko API key for market data fallback
- Google Gemini API key for NLP analysis (optional)

### API Testing (Required for Competition Entry)

**Competition Requirement:** Execute a ~10 USDT trade on `cmt_btcusdt` via API.

```bash
# Dry-run (verify API connectivity without placing orders)
python src/test_weex_real.py --log-file ai_logs/weex_api_test.log

# Execute the required API test trade (~10 USDT on BTC)
WEEX_EXECUTE_ORDER=1 python src/test_weex_real.py --log-file ai_logs/weex_api_test_exec.log
```

### Running the System

**Paper Trading Demo:**
```bash
python src/demo_orchestrator.py
```

**Live WEEX Trading (VPS Recommended):**
```bash
# Set required env vars
export WEEX_EXECUTE_ORDER=1
export ENABLE_LEARNER=1

python src/auraquant_orchestrator.py --duration 600 --min-trades 4 --symbols BTC/USDT,ETH/USDT,SOL/USDT
```

**Test WEEX API Connection:**
```bash
python src/test_weex_real.py
```

### System Validation

```bash
python src/system_validation.py
```

This will run comprehensive tests on all components and generate performance reports.

## Position Management & SL/TP

**How positions are closed:**

1. **Preset SL/TP (Optional):** When `WEEX_USE_PRESET_SLTP=1`, SL/TP prices are sent with the opening order (`presetStopLossPrice` / `presetTakeProfitPrice`). If WEEX rejects those parameters, the bot retries without preset SL/TP and auto-disables this mode.

2. **Server-side TP/SL After Open (Optional):** When `WEEX_USE_SERVER_TPSL_AFTER_OPEN=1`, the bot places TP/SL as plan orders after the entry is opened (`/capi/v2/order/placeTpSlOrder`). This provides server-side protection even if the bot process stops.

3. **Local Monitoring (Fallback):** If both server-side modes are off, the bot monitors price locally via `on_price_tick()` and sends market close orders when SL/TP is hit. This requires the bot to be continuously running.

4. **Max Hold Timeout:** Positions held longer than `MAX_HOLD_SECONDS` (default: 1800s)

5. **No Single-Position Close Endpoint:** WEEX doesn't have a dedicated single-position close endpoint. Closing is done via `placeOrder` with type=3 (close long) or type=4 (close short).

**Environment Variables:**
```bash
WEEX_USE_PRESET_SLTP=0         
WEEX_USE_SERVER_TPSL_AFTER_OPEN=1  # Place TP/SL plan orders after open 
MAX_HOLD_SECONDS=1800          
WEEX_CLOSE_RETRY_COOLDOWN_SECONDS=60  # Cooldown between close retries
```

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

## WEEX AI Log Compliance (Real-time Upload)

**WEEX Requirements:**
- Every AI decision must upload to WEEX within **1 minute** (real-time, not batch).
- All orders (open/close/conditional) logged with `order_id`.
- Retry + disk queue on network failures.
- No sampling; 100% logging required.

**Configuration (.env):**
```bash
WEEX_AI_LOG_UPLOAD_URL="https://www.weex.com/api/ai/log/submit"
WEEX_AI_LOG_AUTH_HEADER="Authorization:Bearer <token>"
```

**Status:**
- ✅ Orchestrator integrates RealTimeAiLogUploader
- ✅ Async background thread + 60s flush interval
- ✅ Network failure → disk queue + exponential retry

## WEEX AI Log Compliance

**Mandatory for competition:** All AI logs must be uploaded in real-time to WEEX for verification and ranking.

### Real-Time Upload

Configure the orchestrator with real-time uploader to push logs automatically (<1 minute delay):

1. Set environment variables:
   ```bash
   export WEEX_AI_LOG_UPLOAD_URL="https://api.weex.com/... (provided by WEEX)"
   export WEEX_AI_LOG_AUTH_HEADER="Authorization:Bearer YOUR_TOKEN"
   ```

2. The orchestrator auto-starts the background uploader thread when `ai_log_uploader` is initialized.

3. Each AI decision (SCAN, QUALIFY, ENTER, MANAGE, EXIT, RECONCILE) is pushed real-time.

### Network Error Handling

If upload fails due to network error:
- Event is persisted to disk queue (`ai_logs/.upload_queue/`)
- Background retry thread attempts re-upload every 30 seconds
- Max retries: 3 attempts (configurable)

If all retries fail:
- Log locally and save proof of network error
- Submit error logs + proof after competition ends

### Required Fields

All uploads include these mandatory fields:
- `stage` (str): Bot phase (SCAN, QUALIFY, ENTER, MANAGE, EXIT, RECONCILE)
- `model` (str): Model name (e.g., "AuraQuant.CorrelationTrigger")
- `input` (JSON): Model inputs
- `output` (JSON): Model outputs
- `explanation` (str): Human-readable summary (≤1000 chars)
- `orderId` (int, optional): Order ID if trade was placed

### Compliance Checklist

| Item | Requirement | Status |
|------|-------------|--------|
| Real-time upload | <1 min delay, every AI decision | ✅ Implemented |
| All orders logged | Market, limit, conditional + order_id | ✅ Via orchestrator |
| Network retry | Local queue + background retry | ✅ Implemented |
| Exact field names | Request body matches WEEX spec | ✅ Validated |
| No credential leak | API keys never in logs | ✅ .env ignored |
| AI participation | Must prove AI involvement in strategy | ✅ Sentiment/Correlation/Risk logs |
| Minimum 10 trades | Prevent "idle participation" | ✅ Enforced at competition |
| Leverage ≤20x | Anti-gambling rule | ✅ Configurable in RiskEngine |

### Testing Upload (Before Competition)

Run a dry-run with logging enabled:

```bash
WEEX_AI_LOG_UPLOAD_URL="https://..." python src/demo_orchestrator.py
```

Check `/ai_logs/` for local records, and verify WEEX dashboard shows events received.

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




