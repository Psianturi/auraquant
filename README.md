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
python src/weex_api_tester.py
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




