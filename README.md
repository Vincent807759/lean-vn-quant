# Lean VN-Quant

Backtesting and paper-trading engine for Vietnamese equities on HOSE (VN30 basket).

[![CI](https://github.com/Vincent807759/lean-vn-quant/actions/workflows/quant_audit_ci.yml/badge.svg)](https://github.com/Vincent807759/lean-vn-quant/actions/workflows/quant_audit_ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Generic backtesters usually assume T+0 settlement, no price limits and unlimited liquidity. None of these hold on HOSE. This project simulates the exchange rules that change a strategy's real P&L:

- **T+2.5 settlement.** A ledger keeps sellable and locked shares (`locked_t1`, `locked_t2`) apart from pending cash receivables.
- **Execution at the open.** Target weights from session *t* are executed at the open of *t+1*.
- **±7% daily price limits.** Buys are blocked at the ceiling, and positions can be trapped at the floor.
- **100-share round lots.**
- **Costs:**
  - Brokerage commission of 0.15% on each side, plus 0.10% tax on sales.
  - Square-root market-impact slippage with a 12 bps floor.
  - A cap on participation as a share of ADV20.
  - Advance-borrowing interest of 0.035% per calendar day, with a 50,000 VND minimum.
- **Corporate actions** credited on the ex-date.
- **Point-in-time liquidity mask** (ADV20 ≥ 10 bn VND) and a turnover dampener.
- **Overfitting checks:**
  - Deflated Sharpe Ratio computed from a trial registry (`experiments/trial_registry.jsonl`).
  - A kill-switch flag when max drawdown exceeds 25%.

The simulator core is compiled with Numba. Data handling uses Polars.

> **Signals are not included.** The public code runs a textbook placeholder signal (SMA-40/50 breakout z-score). Backtests of that signal test the engine; they are not evidence of alpha. See [Limitations](#limitations).

## Quick start

```bash
git clone https://github.com/Vincent807759/lean-vn-quant.git
cd lean-vn-quant
pip install -r requirements.txt

python -m pytest tests/        # unit tests
python run_verification.py     # 8-stage end-to-end verification
```

Both commands run without market data. Missing data is replaced by a synthetic generator, which is also what CI uses.

## Market data

Market data is not distributed with the repository (`data/` is git-ignored). To download daily VN30 bars from the public Entrade API:

```python
from engine.data_pipeline import fetch_vn_ohlcv

fetch_vn_ohlcv(start_date="2023-01-01", end_date="2026-10-24",
               save_parquet="data/vn30_production.parquet")
```

The published results were produced from one fixed snapshot. To check whether your file matches that snapshot:

```bash
python scripts/verify_data.py data/vn30_production.parquet
```

Upstream prices are back-adjusted, so a fresh download can differ slightly from the snapshot. The snapshot's SHA256 and the reproduction steps are in [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) §5.5.

## Using your own signal

Create a local package `private_alphas/`. It is git-ignored, so it is never pushed. The package must expose:

```python
# private_alphas/__init__.py
def compute_custom_alpha(raw_close, tradable_mask, df_market_history=None):
    """Return a (T x N) score matrix; higher = more attractive."""
```

`run_verification.py` and `services/daily_paper_trader.py` import this package automatically. If it is missing, they fall back to the placeholder signal.

## Paper trading

`services/daily_paper_trader.py` runs a forward test with 100M VND of starting capital. Each run does the following:

1. Updates market data.
2. Fills the previous session's orders at the open, subject to HOSE ticks and price limits.
3. Advances the T+2.5 ledger.
4. Marks the portfolio to market against an equal-weight VN30 benchmark.
5. Writes the next session's orders.

Runs are idempotent per date, so running twice on the same day does nothing.

```bash
python services/daily_paper_trader.py                    # default run, after the close
python services/daily_paper_trader.py --mode pre-open    # show staged orders only
python services/daily_paper_trader.py --test-replay 10   # replay last 10 sessions on separate files
python services/dashboard.py --serve                     # HTML dashboard on port 8050
```

State and output files are written to `data/`:

| File | Contents |
| :--- | :--- |
| `paper_portfolio_state.json` | Ledger state |
| `paper_nav_history.csv` | Daily NAV and benchmark |
| `paper_fill_log.csv` | Per-order fills |

`services/setup_autostart.ps1` registers a Windows Task Scheduler job that runs Monday to Friday.

**Telegram (optional).** Copy `.env.example` to `.env` and set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

- `--send-telegram` sends the daily report.
- `--bot-daemon` starts a bot that answers the following commands, and only from that chat id:
  - `/status`
  - `/holdings`
  - `/orders`
  - `/sync`

## Project layout

```
engine/      simulator (vn_simulator.py), data pipeline, alpha operators, multi-factor combiner
services/    paper trader, Telegram bot, dashboard, Task Scheduler setup
scripts/     data verification, backtest comparison, power test, audit scripts
experiments/ data manifest, trial registry, pre-registered hashes and results
fingraph/    optional knowledge-graph factor bridge (falls back safely if absent)
tests/       pytest suite
docs/        technical report
```

Some modules are tested but not used in the backtest P&L loop:

- The intraday order-slicing helpers in `engine/alpha_ops.py`.
- The VN30F hedge calculators in `engine/vn_simulator.py`.

See [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) §2.3.

## Limitations

- **Survivorship bias.** The default basket is the 2026 VN30 constituents. Stocks that left the index earlier are missing.
- **Short sample.** The sample covers about 3.7 years (947 sessions) in a single market regime. Confidence intervals for active Sharpe include zero.
- **Uncalibrated impact model.** Market-impact parameters are literature defaults. They are not calibrated to order-book data, because only daily bars are available.
- **Concentrated, long-only portfolio.** It holds 2 to 4 names, and short selling is not modelled.

## Further reading

[docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) contains:

- The settlement and cost formulas.
- Factor orthogonalisation (ridge projection and Löwdin).
- The verification stages.
- All backtest tables: realistic vs naive, tie-breaking, warmup, cash sweep and DSR.
- The planted-alpha power test.
- The permutation placebo test.
- Data disclosures.

## Security

`python scripts/security_scan.py` checks the git index for secrets, data files and private directories. CI runs it on every push, together with the tests.

## License

MIT. See [LICENSE](LICENSE).
