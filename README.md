# Lean VN-Quant
### Systematic Execution & Microstructure Simulation Engine for Vietnam Equities (HOSE)

[![CI Quantitative Audit](https://github.com/Vincent807759/lean-vn-quant/actions/workflows/quant_audit_ci.yml/badge.svg)](https://github.com/Vincent807759/lean-vn-quant/actions/workflows/quant_audit_ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Numba JIT](https://img.shields.io/badge/accelerated-Numba%20JIT-orange.svg)](https://numba.pydata.org/)
[![Polars Engine](https://img.shields.io/badge/storage-Polars%20DataFrame-navy.svg)](https://pola.rs/)
[![Telegram Bot Control](https://img.shields.io/badge/interactive-Telegram%20Bot%20v2.0-blueviolet.svg)](https://core.telegram.org/bots)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Overview & Architecture

Standard quantitative backtesting frameworks (Backtrader, Zipline, VectorBT) assume continuous frictionless execution, instantaneous T+0 settlement, and infinite liquidity. When deployed on emerging markets such as the **Ho Chi Minh Stock Exchange (HOSE)**, quantitative strategies frequently break down due to unique market microstructural frictions:
- **Strict T+2.5 Settlement Cycles:** Equities purchased on day $t$ remain locked across T+1 and T+2 cycles before becoming tradable in the afternoon session of T+2. Morning ATO selling before settlement is legally restricted.
- **Asymmetric Transaction Frictions:** Independent accounting for brokerage commission (0.15% on both buy and sell legs), personal income transfer tax on gross sales (0.10%), half-spread and market impact slippage (≥ 12 bps), and dynamic calendar-day advance borrowing interest (0.035%/day with a 50,000 VND minimum floor fee).
- **Exchange Price Limits & Floor Locks:** Strict daily price limits (±7%) with zero-bid floor-locked liquidation penalties and zero-ask ceiling-locked execution blocks.
- **Round-Lot Discretization:** Compulsory round-lot execution (100 shares), producing uninvested cash drag.

**Lean VN-Quant** is an open-source, high-performance quantitative simulation engine engineered in Python, Numba C-speed JIT, and Polars. It models strict temporal causality, provides factor orthogonalization via **Ridge Factor Projection** and **Löwdin Symmetric Orthogonalization**, models non-linear market impact, and evaluates dynamic risk mitigation via a **Regime-Switching VN30F1M Futures Beta Hedge** calculation.

```mermaid
flowchart TD
    subgraph Data_Layer ["1. Data Ingestion & Causal Masking (engine/data_pipeline.py)"]
        API["Public REST OHLCV Stream<br/>(DNSE / Entrade)"] --> Fetcher["Data Pipeline & Point-in-Time Filter"]
        Fetcher --> Mask["Point-in-Time Constituent Masking<br/>(adv_20d_lag1 >= 10B VND, min_samples=15)"]
        Mask --> Matrix["Causal 2D Matrices (T x N)<br/>raw_open, raw_close, ceiling, floor, ADV20"]
    end

    subgraph Alpha_Layer ["2. Mathematical Engine & Orthogonalization (engine/)"]
        Matrix --> Ops["alpha_ops.py: Causal Operators<br/>ts_mean, ts_rank, ts_decay_linear, Van der Waerden"]
        Ops --> Plugin["Dynamic Alpha Loader<br/>(private_alphas/ or Benchmark Placeholder)"]
        Plugin --> Ortho["Ridge Factor Projection & Löwdin Orthogonalization<br/>(Cross-Correlation Purging < 0.05)"]
        Ortho --> Dampener["Turnover Dampener Engine<br/>(Rank Hysteresis + EMA Smoothing <= 1.5%/day)"]
    end

    subgraph Sim_Layer ["3. Microstructure Simulation & Risk Management (engine/)"]
        Dampener --> Simulator["vn_simulator.py: @numba.njit Core Engine"]
        Simulator --> Slippage["_calc_dynamic_slippage: Sqrt Volume Impact<br/>(Floor >= 12 bps, eta=0.75, delta=0.55)<br/>Note: vn_market.py is a Standalone Utility"]
        Simulator --> Ledger["SettlementLedger T+2.5 State Machine<br/>sellable_qty, locked_t1, locked_t2<br/>available_cash, receivable_t1, receivable_t2<br/>Dual Ledger Corporate Actions (5% PIT tax)"]
        Simulator --> Hedge["Regime-Switching VN30F1M Beta Hedge<br/>(Contagion Sensitivity, Deadband delta=0.20, Margin <= 18% NAV)"]
    end

    subgraph Ops_Layer ["4. Automated Production Operations (services/)"]
        Ledger --> StateJSON[("paper_portfolio_state.json<br/>Idempotent Ledger")]
        Ledger --> Trader["services/daily_paper_trader.py<br/>(15:15 EOD Reconciliation & 08:15 Pre-Open Staging)"]
        Trader --> Telegram["services/telegram_bot_service.py<br/>2-Way Telegram Bot Daemon"]
    end
```

---

## 2. Microstructure Specifications

### 2.1. HOSE T+2.5 Dual Settlement State Machine
The core ledger state machine (`engine/vn_simulator.py`) maintains explicit separation between settled tradable assets and pending receivables:
- **Cash Accounts:** `available_cash`, `receivable_cash_t1`, `receivable_cash_t2`, `dividend_cash_receivable`.
- **Share Accounts:** `sellable_qty`, `locked_t1`, `locked_t2`, `locked_corp_action_shares`.

**Calendar-Day Advance Borrowing:** When purchasing power requires pre-settlement funds, interest is charged over calendar days at 0.035%/day (with a 50,000 VND minimum floor fee):

$$
\text{Fee}_{\text{adv}} = \max\left(50{,}000\text{ VND}, \quad V_{\text{borrowed}} \times 0.00035 \times D_{\text{cal}}\right)
$$

**Corporate Action Ex-Date Preservation:** Stock dividend and cash dividend rights are credited to escrow receivable registers on Ex-Date ($t_{\text{ex}}$) rather than market payout date ($t_{\text{pay}}$), ensuring zero artificial NAV drop:

$$
\Delta \text{NAV}(t_{\text{ex}}) \equiv 0.0000
$$

### 2.2. Vectorized Point-in-Time Universe Masking
To prevent Survivorship Bias and Look-Ahead Bias, historical constituents are filtered via Polars Lazy API:
```python
adv_20d_lag1 = (raw_close * volume).rolling_mean(window_size=20, min_samples=15).shift(1).over(ticker)
valid_alpha_mask = in_index_pit & (adv_20d_lag1 >= 10_000_000_000.0)
```
Setting `min_samples=15` guarantees newly listed or transferred stocks cannot bypass the 20-day liquidity filter during initial IPO/listing spikes until stable multi-week volume is established. Non-constituents receive `null` masked values, ensuring complete perturbation invariance ($\Delta \text{Signal} \equiv 0.000000$).

### 2.3. Market Impact Modeling & Standalone Execution Utilities
#### Historical Simulation Engine Model:
In backtest simulations (`engine/vn_simulator.py`), execution slippage is modeled via `_calc_dynamic_slippage` combining a 12 bps half-spread floor with temporary and permanent volume impact:

$$
IS(Q) = \max\left(\text{base\_slippage}, \quad \text{half\_spread\_floor} + \eta \cdot \sigma_{\text{daily}} \left(\frac{Q}{\text{ADV}_{20}}\right)^\delta + \gamma_{\text{perm}} \cdot \sigma_{\text{daily}} \left(\frac{Q}{\text{ADV}_{20}}\right)\right)
$$

- **Parameters & Floor:** Base slippage 10 bps, half-spread floor 12 bps (0.0012), with literature parameters $\eta = 0.75$, $\delta = 0.55$, $\gamma_{\text{perm}} = 0.30$.
- **Empirical Calibration Status:** These parameters are literature defaults and **have not yet been calibrated from empirical tick/order book data**.
- **Data Limitation:** The historical dataset (`data/vn30_production.parquet`) consists solely of daily 1D OHLCV bars; full limit order book (LOB depth) data is not present.
#### Standalone Execution & Risk Management Modules:
- **LOB Sweep & Slicing:** The functions `compute_hose_market_impact` (3-tier LOB sweep) and `slice_intraday_order` (25% ATO / 55% TWAP / 20% ATC, capped at 3% ADV20) in `engine/alpha_ops.py` have been independently unit-tested (`tests/test_dynamic_execution.py`), but **are not currently called by the simulation engine** (`_simulate_vn_microstructure_core` in `engine/vn_simulator.py`).
- **VN30F Derivative Hedge:** The hedging calculators `calculate_vn30f_hedge` and `calculate_dynamic_vn30f_hedge` in `engine/vn_simulator.py` provide tested sizing formulas, but **are not currently included in the simulation P&L loop**.

### 2.4. Permutation-Invariant Factor Orthogonalization
Standard Gram-Schmidt QR decomposition suffers from column-order dependence. Lean VN-Quant implements **Ridge Factor Projection**:

$$
\alpha_{\text{ortho}}(t) = \left[ I_N - X_t (X_t^T W_t X_t + \gamma I_K)^{-1} X_t^T W_t \right] \tilde{S}(t)
$$

with shrinkage parameter:

$$
\gamma = \gamma_{\text{shrink}} \times \frac{\text{trace}(X_t^T W_t X_t)}{K}
$$

This achieves mathematical permutation invariance ($\Delta = 4.44 \times 10^{-16}$) while crushing cross-factor correlation from $> 0.85$ down to $< 0.05$. For multi-alpha blending, **Löwdin Symmetric Orthogonalization** is applied:

$$
F_{\text{Löwdin}}(t) = F_t (F_t^T W_t F_t)^{-1/2}
$$

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
2. **Open-Source / CI/CD (No `private_alphas/`):** Automatically falls back to `_default_textbook_placeholder_signal` (SMA-40/50 Breakout Z-Score), allowing verification tests to run and pass deterministically.

---

## 4. Verification & Audit Test Suite

Execute the comprehensive 8-stage audit suite:
```bash
python run_verification.py
```

### Test Suite Coverage:
| Test Stage | Verification Target | Acceptance Criteria |
| :--- | :--- | :--- |
| **TEST 1** | Temporal Causality | All 11 operators & PIT join pass perturbation invariance ($\Delta = 0.0$). |
| **TEST 2** | Ridge Factor Projection | Permutation invariance confirmed ($\Delta = 4.44 \times 10^{-16}$), Correlation $< 0.05$. |
| **TEST 3** | Turnover Dampener | Suppresses raw daily turnover from $> 45\%/\text{day}$ to $\le 5.0\%/\text{day}$. |
| **TEST 4** | Full Backtest Engine | Physically consistent shortfall (0% ≤ IS ≤ 5%), Automated kill-switch SAFE. |
| **TEST 5** | Phase 1 Microstructure | Buy fee (0.15%), sell fee+tax (0.25%), calendar-day 50k floor, Ex-Date NAV delta 0.0%. |
| **TEST 6** | Phase 2 PIT & Impact | `min_samples=15` blocks unseasoned IPOs, 3-tier LOB sweep (≥ 12 bps), 25/55/20 slicing. |
| **TEST 7** | Hawkes/OU & VDW | Continuous exponential decay ($\tau = 5\text{d}, 42\text{d}$), Van der Waerden normal score. |
| **TEST 8** | Dynamic Hedge & DSR | Dynamic VN30F1M switch, margin cap ≤ 18%, calibrated Deflated Sharpe Ratio. |

---

## 5. Quantitative Verification & Empirical Results (947 VN30 Sessions: 2023–2026)

> [!WARNING] Methodological & Statistical Disclaimer
> The empirical tables below evaluate engine mechanics and microstructural friction modeling; **they do not demonstrate statistically significant real-world alpha**. Evaluation sample: ~3.7 years (947 trading sessions from 2023-01-03 to 2026-10-23), a single market regime, a fixed VN30 basket based on 2026 constituents (survivorship bias), and concentrated holdings of only 2–4 symbols (`top_k_in=2`, `top_k_out=4`).

All empirical metrics below are evaluated directly on the complete 947-session historical dataset of the HOSE VN30 basket (`data/vn30_production.parquet`, 2023-01-03 to 2026-10-23, SHA256: `8a3bd271d5fab5750d427712c0ba5715fdc9a9f638f1195661cc5e18996eb0cb`) via `scripts/compare_backtest_b6_b11.py` (Configuration: 947 sessions, initial capital 1.0B VND, brokerage fee 0.15%, sales tax 0.10%, advance borrowing interest 0.035%/calendar day, minimum advance floor fee 50,000 VND, participation cap `max_adv_participation = 0.03`, liquidation penalty rate `penalty_rate_per_day = 0.01`). All metrics are reproducible and validated under pinned dependencies in `requirements.txt` (`numpy==1.26.4`, `numba==0.59.1`).

> [!NOTE] Experimental Configuration Differences: Test 4 (`run_verification.py`) vs. Table 5.1 (`compare_backtest_b6_b11.py`)
> Both Table 5.1 (`compare_backtest_b6_b11.py`) and `run_verification.py` employ the same underlying signal **Alpha 1: Breakout Z-Score SMA-40/50** (`_default_textbook_placeholder_signal` / `generate_five_alphas[0]`), but differ in their microstructural parameterization:
> - **Test 4 in `run_verification.py` (+94.57% Return, NAV 1,945,702,272 VND):** Discards the first `warmup = 50` sessions ($T = 897$ sessions vs 947 sessions); passes `dates` (calculating advance borrowing interest based on actual calendar days across weekends and holidays); uses fixed slippage of `0.10%` with participation cap `max_adv_participation = 0.05` (5% ADV).
> - **Table 5.1 (`scripts/compare_backtest_b6_b11.py`, +95.72% Return, NAV 1,957,198,248 VND):** Evaluates across the entire $T = 947$ sessions; uses dynamic square-root volume impact slippage (`_calc_dynamic_slippage`, 12 bps floor) with strict participation cap `max_adv_participation = 0.03` (3% ADV), standard T+2.5 / T+3 ATO settlement lock, and 5% PIT cash dividend tax.

### 5.1. 947-Session Realistic vs. Naive Simulation (Pre vs. Post B6–B11 & Benchmark VN30)

| Metric | Single-Alpha (Post B6-B11) | Multi-Alpha (Post B6-B11) | Benchmark (VN30 Equal-Weight Rebalanced) | Single-Alpha (Pre-Fix) | Multi-Alpha (Pre-Fix) | Microstructure & Economic Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Final Real NAV (VND)** | **1,957,198,248** | **1,700,542,002** | 1,611,194,562 | 2,267,693,439 | 1,637,074,421 | Initial capital 1.0B VND. Multi-Alpha outperforms the VN30 benchmark across all three profitability metrics: Return (+70.05% vs +61.12%), CAGR (15.18% vs 13.53%), and Sharpe (0.64 vs 0.59), with MaxDD controlled at (18.76% vs 17.78%). Single-Alpha Sharpe reaches 0.81 with Newey-West standard error ±0.53 vs benchmark (0.59 ± 0.54). |
| **Final Naive NAV (VND)** | **2,184,691,709** | **1,916,185,400** | N/A | 2,545,527,906 | 1,849,673,523 | Theoretical frictionless T+0 baseline (timing aligned at raw_open, zero fees, zero taxes, zero slippage). |
| **Cumulative Return** | **+95.72%** | **+70.05%** | +61.12% | +126.77% | +63.71% | Fully net of 0.15% brokerage commission, 0.10% sales tax, T+2 advance loan interest, 5% cash dividend PIT tax, and dynamic square-root volume impact (_calc_dynamic_slippage, 12 bps floor). |
| **Realized CAGR (%/yr)** | **19.57%** | **15.18%** | 13.53% | 24.34% | 14.02% | Single-Alpha CAGR adjusted after eliminating ATO look-ahead bias; Multi-Alpha CAGR reaches 15.18% post-fix. |
| **Theoretical Naive CAGR** | **23.12%** | **18.89%** | N/A | 28.23% | 17.78% | Theoretical compound annual growth rate without microstructural penalties. |
| **Annualized Sharpe (Newey-West)** | **0.81 (±0.53)** | **0.64 (±0.54)** | 0.59 (±0.54) | 1.01 (±0.54) | 0.58 (±0.53) | Newey-West HAC standard error indicates confidence intervals of ~ ±0.54 across all strategies. |
| **Sortino Ratio** | **1.04** | **0.80** | 0.70 | 1.29 | 0.73 | Downside risk-adjusted return ratio. |
| **Annualized Volatility** | **19.47%** | **18.57%** | 17.28% | 19.58% | 18.58% | Realized strategy annualized volatility is well-controlled under 20%. |
| **Maximum Drawdown (MaxDD)** | **18.10%** | **18.76%** | 17.78% | 17.83% | 19.30% | Single-Alpha MaxDD is 18.10% (comparable to benchmark 17.78%); Multi-Alpha MaxDD is 18.76%. |
| **Daily Turnover (Target Weights)** | **1.4752%** | **1.4752%** | 0.1045% | 1.4767% | 1.4767% | Target weights turnover: 1.4752%/day (constrained by dampener `max_turnover_target = 0.015`). Realized executed ledger turnover: Single 1.5144%/day, Multi 1.6077%/day. |
| **Cumulative IS (% Capital)** | **+22.75%** | **+21.56%** | N/A | +27.78% | +21.26% | (NAV_Naive - NAV_Real) / 1.0B VND: Microstructure frictions consume 21.56%–22.75% of initial capital. |
| **Annualized IS (%/yr)** | **+3.55%/yr** | **+3.72%/yr** | N/A | +3.88%/yr | +3.77%/yr | CAGR_Naive - CAGR_Real: Market frictions impose an annual drag of 3.55%–3.72%/year. |
| **Total Fees & Transfer Taxes** | **82.7M VND** | **80.6M VND** | 30.0M VND | 89.4M VND | 80.6M VND | Independent accounting of 0.15% brokerage commission and 0.10% gross sales tax. |
| **Advance Borrowing Interest** | **25.1M VND** | **17.8M VND** | 0.3M VND | 32.4M VND | 19.1M VND | 94.1%–95.2% of advance borrowing costs stem from the 50,000 VND floor fee on small lots (506 times for Single, 376 times for Multi). |

> [!NOTE] Methodological Notes on Table 5.1
> - **Implementation Shortfall (IS):** IS is measured by the NAV spread (Naive − Real) after timing alignment at `raw_open` (Phase 2), eliminating timing mismatch artifacts to isolate real market frictions (brokerage fees, sales taxes, execution slippage, advance borrowing interest).
> - **Deflated Sharpe Ratio (DSR):** DSR p-values range from 0.07 to 0.79 across Single and Multi-Alpha configurations (see Table 5.3); no regime achieves statistical significance after multiple testing haircut.
> - **Benchmark Definition:** The Benchmark column reflects an Equal-Weight daily rebalanced VN30 basket, not static Buy & Hold.

---

### 5.2. Smart Cash Sweep Optimization

Vietnam securities brokerages enforce a compulsory minimum advance borrowing fee of **50,000 VND per advance transaction**, regardless of loan notional. Small residual cash sweep orders incur an effective annualized borrowing rate exceeding 100%:

$$
\text{Break-Even Advance Notional} = \frac{50{,}000\text{ VND}}{0.00035 \times 2\text{ days}} = 71{,}428{,}571\text{ VND}
$$

By configuring `min_advance_notional = 71,428,571 VND`, the engine completely bypasses the sub-optimal borrowing trap on odd/fractional amounts:

#### Single-Alpha Cash Sweep Comparison:
| Configuration | Advance Fees Paid (VND) | Effective Advance Rate | Cumulative Return | 50k Floor Fee Incurred | Final Real NAV (VND) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Greedy Sweep (Baseline)** | 25,150,000 | ~0.62% / order (Floor penalty) | +95.72% | 506 times | 1,957,198,248 |
| **Smart Cash Sweep (`min = 71.4M`)** | **0** | **0.00% (Zero borrowing)** | **+97.42% (+1.70%)** | **0 times** | **1,974,165,840** |
| **No Advance Sweep (T+2 Closed)** | 0 | 0.00% | +97.42% (+1.70%) | 0 times | 1,974,165,840 |

#### Multi-Alpha Cash Sweep Comparison:
| Configuration | Advance Fees Paid (VND) | Effective Advance Rate | Cumulative Return | 50k Floor Fee Incurred | Final Real NAV (VND) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Greedy Sweep (Baseline)** | 17,800,000 | ~0.62% / order (Floor penalty) | +70.05% | 376 times | 1,700,542,002 |
| **Smart Cash Sweep (`min = 71.4M`)** | **0** | **0.00% (Zero borrowing)** | **+72.95% (+2.90%)** | **0 times** | **1,729,491,161** |
| **No Advance Sweep (T+2 Closed)** | 0 | 0.00% | +72.95% (+2.90%) | 0 times | 1,729,491,161 |

---

### 5.3. 3-Mode Deflated Sharpe Ratio (DSR) Analysis (Bailey & López de Prado, 2014)

*(Generated from baseline verification audit. Generator script: `scripts/audit_argon_revisions.py`)*

To mitigate data mining selection bias across multiple trial evaluations, Lean VN-Quant logs all exploration runs to `experiments/trial_registry.jsonl` and evaluates the strategy under three rigorous DSR formulations:

| Evaluation Mode | Description | N | N_eff | Avg Cross-Corr (rho_bar) | Null Threshold (SR_0) | Single DSR (p-value) | Multi DSR (p-value) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mode 1: Total-Return Registry** | Full registry correlation matrix (C_26x26). Reflects market beta co-movement. | 26 | **1.1194** | 0.9387 | 0.0436 | **0.9282** ($p=0.0718$) | **0.8472** ($p=0.1528$) |
| **Mode 2: Active-Return Core Alphas** | Active returns over VN30 benchmark (r_active = r - r_bm) across 9 core alphas. Purges market beta. | 9 | **3.5194** | 0.2793 | 0.6584 | **0.6132** ($p=0.3868$) | **0.4388** ($p=0.5612$) |
| **Mode 3: Unadjusted Conservative Bound** | Orthogonal independent assumption (I_26). Extreme upper bound on multiple-testing penalty. | 26 | **26.0000** | 0.0000 | 1.0536 | **0.3200** ($p=0.6800$) | **0.1825** ($p=0.8175$) |

---

### 5.4. Pillar 2 (VN-FinGraph Knowledge Graph) Ablation Study (6-Factor Blend)

*(Baseline research data. Requires proprietary research modules `fin_corpus_pipeline` and `knowledge_graph_engine`)*

By coupling unstructured financial news/event contagion matrices from Pillar 2 (`vn-fingraph`) into the Pillar 1 quantitative execution engine via Parquet bridge, we evaluate the impact of Graph Laplacian risk diffusion $\alpha \in \{0.15, 0.25, 0.50, 0.70\}$:

*(Note: FinGraph requires proprietary modules `fin_corpus_pipeline` and `knowledge_graph_engine` which are non-public research; `fingraph/bridge_to_vnquant.py` implements safe lazy imports with explicit warnings when these modules are absent).*

| Laplacian Propagation alpha | Rank IC (1-Day Forward) | Cumulative Return (6-Factor) | CAGR (%/yr) | Annualized Sharpe | Max Drawdown | Architectural Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **alpha = 0.15** (Fast decay) | -0.0158 | +63.14% | 13.91% | 0.58 | 19.97% | Narrow diffusion; shocks contained at epicenter. Performs on par with 5-factor baseline (+63.32%). |
| **alpha = 0.25** (Standard configuration) | -0.0137 | **+65.81%** | **14.40%** | **0.60** | **19.99%** | Standard balance: Adds +2.49% return over 5-factor baseline, Sharpe improves from 0.58 to 0.60. |
| **alpha = 0.50** (Broad network diffusion) | -0.0121 | +69.57% | 15.09% | 0.63 | 18.77% | Broad transmission; lifts CAGR to 15.09% while dampening MaxDD to 18.77%. |
| **alpha = 0.70** (High amplification) | -0.0086 | +83.00% | 17.45% | 0.73 | 18.76% | High amplification; lifts Sharpe to 0.73 but introduces cross-network systemic sensitivity. |

> [!NOTE] Methodological Limitation & Non-Public Dependencies
> - **Rank IC:** 1-day forward Rank IC values are negative across all 4 diffusion levels $\alpha$ (ranging from -0.0158 to -0.0086).
> - **Reproducibility Note:** Upstream FinGraph corpus modules (`fin_corpus_pipeline` and `knowledge_graph_engine`) represent proprietary research; this theoretical ablation table cannot be independently reproduced solely from the public open-source repository.

---

### 5.5. Empirical Reproducibility Guide

#### 1. Market Data Ingestion (Entrade REST API)
Historical market bars can be freshly ingested directly via the data pipeline:
```python
from engine.data_pipeline import fetch_vn_ohlcv

# Ingest 30 VN30 tickers and persist to Parquet format
df = fetch_vn_ohlcv(
    start_date="2023-01-01",
    end_date="2026-10-24",
    save_parquet="data/vn30_production.parquet"
)
```
*Note:* Ingested figures may exhibit minor variations across different retrieval dates due to upstream Entrade API backward adjustments when corporate stock dividends or cash distributions occur.

#### 2. Alpha Signal Lineage in Verification Tables
- All results in Tables 5.1, 5.2, and 5.3 are generated completely endogenously from public open-source code without `private_alphas/`.
- **Single-Alpha (`w_single`):** Employs Alpha 1 (SMA-40/50 Breakout Z-Score) from `generate_five_alphas` (`engine/ml_alpha_combiner.py`), stabilized via the `dampen_turnover` engine.
- **Multi-Alpha (`w_multi`):** Blends 5 baseline technical factors after QR/Ridge factor orthogonalization (`check_and_orthogonalize`) using the Purged Walk-Forward Combiner (`window=120`, `embargo=3`).

#### 3. Execution Commands for Verification & Benchmarking
```bash
# Compare realistic simulation vs Naive baseline (generates Table 5.1):
python scripts/compare_backtest_b6_b11.py

# Execute Argon 4 statistical verification experiment suite (generates Tables 5.2 & 5.3):
python scripts/audit_argon_revisions.py

# Run automated unit test suite (59 unit tests):
python -m pytest tests/ -v

# Run comprehensive 8-stage verification test runner:
python run_verification.py
```

---

## 6. Security & Pre-Commit Protection

Run the security auditor:
```bash
python scripts/security_scan.py
```
The scanner enforces strict pre-commit checks:
- Blocks tracked `.env` credentials, private keys, or personal workspace file paths.
- Intercepts forbidden private directories (`private_alphas/`, `research/`, `local_docs/`).
- Blocks proprietary data files (`*.parquet`, `*.csv`) from Git index.
- Enforces secret and sensitive data exclusion across GitHub Actions CI/CD.
- **CI/CD Automation:** GitHub Actions CI automatically runs `scripts/security_scan.py`, `pytest tests/`, and `run_verification.py` (8-stage verification suite) on every push and PR across Python 3.10 and 3.11.

---

## 7. Daily Operations

### Production Schedule:
1. **End-of-Day Reconciliation (15:15 ICT, Post-ATC):**
   ```bash
   python services/daily_paper_trader.py --mode eod
   ```
   Reconciles ATC closing prices, marks portfolio to market, advances the T+2.5 settlement ledger (with idempotency lock `last_ledger_advance_date`), and stages pending orders for tomorrow morning.

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

## 8. Project Structure

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
│   └── bridge_to_vnquant.py         # Hawkes/OU decay filter & Van der Waerden score (FinGraph bridge with safe fallback)
├── services/
│   ├── __init__.py
│   ├── daily_paper_trader.py        # Forward-testing engine & order ticket generator
│   ├── dashboard.py                 # Interactive Chart.js portfolio dashboard
│   ├── setup_autostart.ps1          # Task Scheduler automated service script
│   └── telegram_bot_service.py      # 2-Way Telegram Bot interactive management daemon
├── scripts/
│   ├── security_scan.py             # Secret & alpha privacy auditor
│   ├── compare_backtest_b6_b11.py   # Empirical backtest comparison runner
│   └── audit_argon_revisions.py     # Argon 4 statistical verification experiment suite
├── .gitignore                       # Strict exclusion rules for data, keys & private alphas
├── LICENSE                          # MIT License
├── pyproject.toml                   # Project metadata & build configuration
├── requirements.txt                 # Pinned dependencies (numpy, polars, numba, scipy)
├── README.md                        # Documentation
└── run_verification.py              # 8-stage end-to-end verification test suite
```

---

## 9. Data Integrity & Methodological Disclosures

### 9.1. Universe Selection & Survivorship Bias
- **Fixed Basket Limitation:** The default historical data ingestion (`data/vn30_production.parquet`, 2023-01-03 to 2026-10-23) utilizes the 30 constituent tickers of the VN30 index as constituted in 2026. Companies that were removed from VN30 prior to 2026 (e.g., ROS, HNG, NVL) are excluded from the default sample basket, introducing potential survivorship bias.
- **Point-in-Time Membership Support:** The pipeline interface `engine/data_pipeline.py` supports `universe_membership_path: Optional[str] = None` to dynamically filter tickers per historical HOSE index rebalancing effective dates (January and July annual cycles) when constituent membership tables are provided.

### 9.2. Price Basis & Dual-Price Simulation
- **Entrade Adjusted Data:** Prices obtained from the upstream Entrade REST API are backward-adjusted for stock splits, bonus shares, and cash dividends (`price_basis = "entrade_adjusted"`).
- **Impact on Historical Levels:** While returns and corporate-action NAV preservation remain mathematically exact, nominal historical price thresholds (such as $\pm 7\%$ ceiling/floor step levels in 2023) reflect post-split adjusted scales rather than unadjusted exchange ticker tape prices.

---

## 10. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
