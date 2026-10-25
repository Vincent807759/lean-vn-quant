# Lean VN-Quant
### Institutional Systematic Execution & Microstructure Engine for Vietnam Equities (HOSE)

[![CI Quantitative Audit](https://github.com/Vincent807759/lean-vn-quant/actions/workflows/quant_audit_ci.yml/badge.svg)](https://github.com/Vincent807759/lean-vn-quant/actions/workflows/quant_audit_ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Numba JIT](https://img.shields.io/badge/accelerated-Numba%20JIT-orange.svg)](https://numba.pydata.org/)
[![Polars Engine](https://img.shields.io/badge/storage-Polars%20DataFrame-navy.svg)](https://pola.rs/)
[![Telegram Bot Control](https://img.shields.io/badge/interactive-Telegram%20Bot%20v2.0-blueviolet.svg)](https://core.telegram.org/bots)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Overview & Architecture

Standard quantitative backtesting frameworks (Backtrader, Zipline, VectorBT) assume continuous frictionless execution, instantaneous $T+0$ settlement, and infinite liquidity. When deployed on emerging markets such as the **Ho Chi Minh Stock Exchange (HOSE)**, quantitative strategies frequently break down due to unique market microstructural frictions:
- **Strict $T+2.5$ Settlement Cycles:** Equities purchased on day $t$ remain locked across $T+1$ and $T+2$ cycles before becoming tradable in the afternoon session of $T+2$. Morning ATO selling before settlement is legally restricted.
- **Asymmetric Transaction Frictions:** Independent accounting for brokerage commission ($0.15\%$ on both buy and sell legs), personal income transfer tax on gross sales ($0.10\%$), half-spread and market impact slippage ($\ge 12\text{ bps}$), and dynamic calendar-day advance borrowing interest ($0.035\%/\text{day}$ with a $50,000\text{ VND}$ minimum floor fee).
- **Exchange Price Limits & Floor Locks:** Strict daily price limits ($\pm 7\%$) with zero-bid floor-locked liquidation penalties and zero-ask ceiling-locked execution blocks.
- **Round-Lot Discretization:** Compulsory round-lot execution ($100$ shares), producing uninvested cash drag.

**Lean VN-Quant** is an open-source, high-performance quantitative simulation and execution engine engineered in Python, Numba C-speed JIT, and Polars. It enforces strict temporal causality ($0\%$ look-ahead bias), provides permutation-invariant factor orthogonalization via **Ridge Factor Projection** and **Löwdin Symmetric Orthogonalization**, models non-linear market impact via a **3-Tier LOB Sweep Model**, and executes dynamic risk mitigation via a **Regime-Switching VN30F1M Futures Beta Hedge**.

```mermaid
flowchart TD
    subgraph Data_Layer ["1. Data Ingestion & Causal Masking (engine/data_pipeline.py)"]
        API["Public REST OHLCV Stream\n(DNSE / Entrade)"] --> Fetcher["Data Pipeline & Point-in-Time Filter"]
        Fetcher --> Mask["Point-in-Time Constituent Masking\n(adv_20d_lag1 >= 10B VND, min_samples=15)"]
        Mask --> Matrix["Causal 2D Matrices (T x N)\nraw_open, raw_close, ceiling, floor, ADV20"]
    end

    subgraph Alpha_Layer ["2. Mathematical Engine & Orthogonalization (engine/)"]
        Matrix --> Ops["alpha_ops.py: Causal Operators\nts_mean, ts_rank, ts_decay_linear, Van der Waerden"]
        Ops --> Plugin["Dynamic Alpha Loader\n(private_alphas/ or Benchmark Placeholder)"]
        Plugin --> Ortho["Ridge Factor Projection & Löwdin Orthogonalization\n(Cross-Correlation Purging < 0.05)"]
        Ortho --> Dampener["Turnover Dampener Engine\n(Rank Hysteresis + EMA Smoothing <= 1.5%/day)"]
    end

    subgraph Sim_Layer ["3. Microstructure Simulation & Risk Management (engine/)"]
        Dampener --> Simulator["vn_simulator.py: @numba.njit Core Engine"]
        Simulator --> LOB["3-Tier LOB Sweep & Square-Root Impact\n(Floor >= 12 bps, 25% ATO - 55% TWAP - 20% ATC)"]
        Simulator --> Ledger["SettlementLedger T+2.5 State Machine\nsellable_qty, locked_t1, locked_t2\navailable_cash, receivable_t1, receivable_t2\nDual Ledger Corporate Actions"]
        Simulator --> Hedge["Regime-Switching VN30F1M Beta Hedge\n(Contagion Sensitivity, Deadband delta=0.20, Margin <= 18% NAV)"]
    end

    subgraph Ops_Layer ["4. Automated Production Operations (services/)"]
        Ledger --> StateJSON[("paper_portfolio_state.json\nIdempotent Ledger")]
        Ledger --> Trader["services/daily_paper_trader.py\n(15:15 EOD Reconciliation & 08:15 Pre-Open Staging)"]
        Trader --> Telegram["services/telegram_bot_service.py\n2-Way Telegram Bot Daemon"]
    end
```

---

## 2. Microstructure Specifications

### 2.1. HOSE $T+2.5$ Dual Settlement State Machine
The core ledger state machine (`engine/vn_simulator.py`) maintains explicit separation between settled tradable assets and pending receivables:
- **Cash Accounts:** `available_cash`, `receivable_cash_t1`, `receivable_cash_t2`, `dividend_cash_receivable`.
- **Share Accounts:** `sellable_qty`, `locked_t1`, `locked_t2`, `locked_corp_action_shares`.
- **Calendar-Day Advance Borrowing:** When purchasing power requires pre-settlement funds, interest is charged over calendar days (e.g., Thursday buy borrows over 4 calendar days through Monday settlement; Friday buy borrows over 3-4 calendar days):
  $$\text{Fee}_{\text{adv}} = \max\left(50,000\text{ VND}, \; V_{\text{borrowed}} \times 0.035\% \times \text{cal\_days}\right)$$
- **Corporate Action Ex-Date Preservation:** Stock dividend and cash dividend rights are credited to escrow receivable registers on Ex-Date ($t_{\text{ex}}$) rather than market payout date ($t_{\text{pay}}$), ensuring zero artificial NAV drop:
  $$\Delta \text{NAV}(t_{\text{ex}}) \equiv 0.0000\%$$

### 2.2. Vectorized Point-in-Time Universe Masking
To prevent Survivorship Bias and Look-Ahead Bias, historical constituents are filtered via Polars Lazy API:
```python
adv_20d_lag1 = (raw_close * volume).rolling_mean(window_size=20, min_samples=15).shift(1).over(ticker)
valid_alpha_mask = in_index_pit & (adv_20d_lag1 >= 10_000_000_000.0)
```
Setting `min_samples=15` guarantees newly listed or transferred stocks cannot bypass the 20-day liquidity filter during initial IPO/listing spikes until stable multi-week volume is established. Non-constituents receive `null` masked values, ensuring complete perturbation invariance ($\Delta \text{Signal} \equiv 0.000000$).

### 2.3. 3-Tier Order Book Sweep & Intraday Slicing
Large institutional orders incur realistic non-linear market impact:
$$IS(Q) = \frac{S_t}{2 P_{\text{mid}}} + \text{SweepCost}_{\text{LOB}} + \mathbb{I}(Q > Q_{\text{vis}}) \left[ \eta \cdot \sigma_{\text{daily}} \left(\frac{Q - Q_{\text{vis}}}{\text{ADV}_{20}}\right)^\delta + \gamma_{\text{perm}} \cdot \sigma_{\text{daily}} \left(\frac{Q}{\text{ADV}_{20}}\right) \right]$$
- Half-spread floor: $12\text{ bps} = 0.0012$.
- Intraday order slicing rule: If order $Q > 0.5\% \text{ ADV}_{20}$, execution is automatically partitioned into **25% ATO** (09:00 - 09:15), **55% TWAP** (09:30 - 14:15), and **20% ATC** (14:30 - 14:45), capped at $3\% \text{ ADV}_{20}$.

### 2.4. Permutation-Invariant Factor Orthogonalization
Standard Gram-Schmidt QR decomposition suffers from column-order dependence. Lean VN-Quant implements **Ridge Factor Projection**:
$$\alpha_{\text{ortho}}(t) = \left[ I_N - X_t (X_t^T W_t X_t + \gamma I_K)^{-1} X_t^T W_t \right] \tilde{S}(t)$$
with shrinkage parameter $\gamma = \gamma_{\text{shrink}} \times \frac{\text{trace}(X_t^T W_t X_t)}{K}$. This achieves mathematical permutation invariance ($\Delta = 4.44 \times 10^{-16}$) while crushing cross-factor correlation from $> 0.85$ down to $< 0.05$. For multi-alpha blending, **Löwdin Symmetric Orthogonalization** is applied:
$$F_{\text{Löwdin}}(t) = F_t (F_t^T W_t F_t)^{-1/2}$$

---

## 3. Dynamic Alpha Plugin Interface (`private_alphas/`)

To protect proprietary research and prevent alpha leakage in open-source environments, the engine cleanly decouples technical infrastructure from trading signals:

```
lean-vn-quant/
├── private_alphas/           <-- Local only (in .gitignore). Put proprietary alphas here.
│   ├── __init__.py
│   └── custom_strategy.py    <-- Implements: compute_custom_alpha(raw_close, mask, ...)
├── engine/                   <-- Public open-source quantitative engine
├── services/                 <-- Operational execution services
└── run_verification.py       <-- Audit test runner
```

### Dynamic Loading Mechanism
Both [`run_verification.py`](run_verification.py) and [`services/daily_paper_trader.py`](services/daily_paper_trader.py) query the local environment dynamically:
1. **Local Deployment with `private_alphas/`:** Automatically imports your custom alpha functions and generates live trading allocations.
2. **Open-Source / CI/CD (No `private_alphas/`):** Automatically falls back to `_default_textbook_placeholder_signal` (SMA-40/50 Breakout Z-Score), allowing the entire 8-stage verification test suite to run and pass with $100\%$ determinism.

---

## 4. Verification & Audit Test Suite

Execute the comprehensive 8-stage audit suite:
```bash
python run_verification.py
```

### Test Suite Coverage:
| Test Stage | Verification Target | Acceptance Criteria |
| :--- | :--- | :--- |
| **TEST 1** | 0% Look-Ahead Bias | All 11 operators & PIT join pass perturbation invariance ($\Delta = 0.0$). |
| **TEST 2** | Ridge Factor Projection | Permutation invariance confirmed ($\Delta = 4.44 \times 10^{-16}$), Correlation $< 0.05$. |
| **TEST 3** | Turnover Dampener | Suppresses raw daily turnover from $> 45\%/\text{day}$ to $\le 5.0\%/\text{day}$. |
| **TEST 4** | Full Backtest Engine | Physically consistent shortfall ($0\% \le IS \le 5\%$), Automated kill-switch SAFE. |
| **TEST 5** | Phase 1 Microstructure | Buy fee (0.15%), sell fee+tax (0.25%), calendar-day 50k floor, Ex-Date NAV delta 0.0%. |
| **TEST 6** | Phase 2 PIT & Impact | `min_samples=15` blocks unseasoned IPOs, 3-tier LOB sweep ($\ge 12\text{ bps}$), 25/55/20 slicing. |
| **TEST 7** | Hawkes/OU & VDW | Continuous exponential decay ($\tau=5\text{d}, 42\text{d}$), Van der Waerden normal score. |
| **TEST 8** | Dynamic Hedge & DSR | Dynamic VN30F1M switch, margin cap $\le 18\%$, calibrated Deflated Sharpe Ratio. |

---

## 5. Security & Pre-Commit Protection

Run the zero-leak security auditor:
```bash
python scripts/security_scan.py
```
The scanner enforces strict pre-commit checks:
- Blocks tracked `.env` credentials, private keys, or personal workspace file paths.
- Intercepts forbidden private directories (`private_alphas/`, `research/`, `local_docs/`).
- Blocks proprietary data files (`*.parquet`, `*.csv`) from Git index.
- Enforces Zero-Leak integrity across GitHub Actions CI/CD.

---

## 6. Daily Operations

### Production Schedule:
1. **End-of-Day Reconciliation (15:15 ICT, Post-ATC):**
   ```bash
   python services/daily_paper_trader.py --mode eod
   ```
   Reconciles ATC closing prices, marks portfolio to market, advances the $T+2.5$ settlement ledger (with idempotency lock `last_ledger_advance_date`), and stages pending orders for tomorrow morning.

2. **Pre-Open Staging (08:15 ICT, 45 mins before ATO):**
   ```bash
   python services/daily_paper_trader.py --mode pre-open
   ```
   Inspects corporate actions, checks overnight risk scores, and displays staged ATO/TWAP order tickets without altering the settlement ledger.

3. **Multi-Session Replay Test Audit:**
   ```bash
   python services/daily_paper_trader.py --test-replay 10
   ```

---

## 7. Project Structure

```
lean-vn-quant/
├── .github/
│   └── workflows/
│       └── quant_audit_ci.yml       # Automated GitHub Actions CI/CD test runner
├── engine/
│   ├── __init__.py
│   ├── alpha_ops.py                 # Vectorized causal operators & PIT universe masking
│   ├── data_pipeline.py             # Entrade REST data ingestion & DuckDB engine
│   ├── ml_alpha_combiner.py         # Ridge factor projection & Löwdin orthogonalization
│   └── vn_simulator.py              # HOSE T+2.5 microstructure engine & dynamic hedge
├── fingraph/
│   ├── __init__.py
│   └── bridge_to_vnquant.py         # Hawkes/OU decay filter & Van der Waerden score
├── services/
│   ├── __init__.py
│   ├── daily_paper_trader.py        # Forward-testing engine & order ticket generator
│   ├── dashboard.py                 # Interactive Chart.js portfolio dashboard
│   ├── setup_autostart.ps1          # Task Scheduler automated service script
│   └── telegram_bot_service.py      # 2-Way Telegram Bot interactive management daemon
├── scripts/
│   └── security_scan.py             # Zero-leak secret & alpha privacy auditor
├── .gitignore                       # Strict exclusion rules for data, keys & private alphas
├── LICENSE                          # MIT License
├── pyproject.toml                   # Project metadata & build configuration
├── requirements.txt                 # Pinned dependencies (numpy, polars, numba, scipy)
├── README.md                        # Documentation
└── run_verification.py              # 8-stage end-to-end verification test suite
```

---

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
