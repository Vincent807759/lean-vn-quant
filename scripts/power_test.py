"""Planted-alpha power test market generator, evaluation grid, and CLI.

Simulates synthetic market environments with planted cross-sectional alpha (IC = rho)
under realistic HOSE microstructure constraints (fees, taxes, T+2.5, tick sizes, limits).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numba
import numpy as np
from scipy import stats
from scipy.signal import lfilter

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine import alpha_ops as ops, data_pipeline as dp, vn_simulator as sim

CONSTRUCTIONS: Dict[str, Tuple[int, int, int, float]] = {
    "C1": (2, 4, 20, 0.015),
    "C2": (10, 18, 10, 0.05),
    "C3": (2, 4, 10, 0.05),
}

MAIN_RHO_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]
PHI05_RHO_GRID = [0.0, 0.05, 0.10]


def round_tick_np(prices: np.ndarray) -> np.ndarray:
    """Vectorized HOSE tick rounding matching data_pipeline.round_tick_hose."""
    prices = np.asarray(prices, dtype=np.float64)
    tick = np.where(prices < 10000.0, 10.0, np.where(prices < 50000.0, 50.0, 100.0))
    return np.round(prices / tick) * tick


def hose_limits_np(ref_prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized HOSE ceiling and floor limits matching compute_hose_limits."""
    ref_prices = np.asarray(ref_prices, dtype=np.float64)
    raw_ceil = ref_prices * 1.07
    ceil_tick = np.where(raw_ceil < 10000.0, 10.0, np.where(raw_ceil < 50000.0, 50.0, 100.0))
    ceiling = np.floor(raw_ceil / ceil_tick) * ceil_tick

    raw_flr = ref_prices * 0.93
    flr_tick = np.where(raw_flr < 10000.0, 10.0, np.where(raw_flr < 50000.0, 50.0, 100.0))
    floor = np.ceil(raw_flr / flr_tick) * flr_tick

    return ceiling, floor


def get_business_dates(
    T: int = 947,
    start_date: datetime.date = datetime.date(2023, 1, 3),
) -> List[datetime.date]:
    """Generate T consecutive business days (Monday to Friday) starting from start_date."""
    dates: List[datetime.date] = []
    cur = start_date
    while len(dates) < T:
        if cur.weekday() < 5:
            dates.append(cur)
        cur += datetime.timedelta(days=1)
    return dates


def draw_latents(
    seed: int,
    rho: float,
    T: int = 947,
    N: int = 30,
    phi: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw latent market processes using 4 independent RNG streams from SeedSequence."""
    ss = np.random.SeedSequence(seed)
    rng_signal, rng_eps, rng_market, rng_gap = [
        np.random.default_rng(s) for s in ss.spawn(4)
    ]

    # 1. Signal AR(1): s[0] ~ N(0, 1), s[t] = phi*s[t-1] + sqrt(1 - phi^2)*z[t]
    s0 = rng_signal.standard_normal(N)
    if T > 1:
        z = rng_signal.standard_normal((T - 1, N))
        w = np.empty((T, N), dtype=np.float64)
        w[0] = s0
        w[1:] = np.sqrt(max(0.0, 1.0 - phi**2)) * z
        s = lfilter([1.0], [1.0, -phi], w, axis=0)
    else:
        s = s0.reshape(1, N)

    # 2. Idiosyncratic shocks: eps ~ N(0, 1)
    eps = rng_eps.standard_normal((T, N))
    u = np.empty((T, N), dtype=np.float64)
    u[0] = eps[0]
    if T > 1:
        scale_eps = np.sqrt(max(0.0, 1.0 - rho**2))
        u[1:] = rho * s[:-1] + scale_eps * eps[1:]

    # 3. Market & stock parameters
    sigma = rng_market.uniform(0.012, 0.017, size=N)
    adv = rng_market.uniform(1.5e10, 6.0e10, size=N)
    sigma_m = 0.008
    mu_f = 0.04 / 252.0
    m_raw = rng_market.standard_normal(size=T)
    m = (mu_f - 0.5 * sigma_m**2) + sigma_m * m_raw

    # 4. Overnight gap
    xi = rng_gap.standard_normal((T, N))
    gap = -0.5 * ((0.4 * sigma) ** 2) + (0.4 * sigma) * xi

    # Intraday open-to-close log return
    r_oc = m[:, None] - 0.5 * (sigma**2) + sigma * u

    return s, u, m, gap, r_oc, sigma, adv


def cross_sectional_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Mean cross-sectional Pearson correlation across rows (days)."""
    if x.shape[0] == 0:
        return 0.0
    xc = x - np.mean(x, axis=1, keepdims=True)
    yc = y - np.mean(y, axis=1, keepdims=True)
    sx = np.std(x, axis=1, ddof=0)
    sy = np.std(y, axis=1, ddof=0)
    denom = sx * sy
    valid = denom >= 1e-12
    corr = np.zeros(x.shape[0], dtype=np.float64)
    corr[valid] = np.mean(xc[valid] * yc[valid], axis=1) / denom[valid]
    return float(np.mean(corr))


def generate_planted_market(
    seed: int,
    rho: float,
    T: int = 947,
    N: int = 30,
    phi: float = 0.9,
    base_prices: Optional[Union[Dict[str, float], List[float], np.ndarray]] = None,
    dates: Optional[List[datetime.date]] = None,
    latents: Optional[Tuple[np.ndarray, ...]] = None,
) -> Dict[str, Any]:
    """Generate synthetic market prices with planted alpha under HOSE constraints."""
    if latents is None:
        s, u, m, gap, r_oc, sigma, adv = draw_latents(seed, rho, T=T, N=N, phi=phi)
    else:
        s, u, m, gap, r_oc, sigma, adv = latents

    if base_prices is None:
        symbols = dp.VN30_SYMBOLS[:N]
        base_arr = np.array([dp.BASE_PRICES.get(sym, 50000.0) for sym in symbols], dtype=np.float64)
    elif isinstance(base_prices, dict):
        symbols = list(base_prices.keys())[:N]
        base_arr = np.array([base_prices[sym] for sym in symbols], dtype=np.float64)
    else:
        symbols = [f"SYM_{i:02d}" for i in range(N)]
        base_arr = np.asarray(base_prices, dtype=np.float64)

    if dates is None:
        dates = get_business_dates(T=T)

    open_p = np.empty((T, N), dtype=np.float64)
    close_p = np.empty((T, N), dtype=np.float64)
    ceil_p = np.empty((T, N), dtype=np.float64)
    flr_p = np.empty((T, N), dtype=np.float64)

    # Day 0 initialization
    open_p[0] = close_p[0] = round_tick_np(base_arr)
    c0, f0 = hose_limits_np(close_p[0])
    ceil_p[0] = c0
    flr_p[0] = f0

    # Sequential price recursion
    for t in range(1, T):
        ref = close_p[t - 1]
        c, f = hose_limits_np(ref)
        ceil_p[t] = c
        flr_p[t] = f
        raw_op = ref * np.exp(gap[t])
        op = np.clip(round_tick_np(raw_op), f, c)
        open_p[t] = op
        raw_cl = op * np.exp(r_oc[t])
        cl = np.clip(round_tick_np(raw_cl), f, c)
        close_p[t] = cl

    # Liquidity
    adv20 = np.tile(adv, (T, 1))
    adv20_shares = adv20 / close_p
    mask = np.ones((T, N), dtype=bool)

    return {
        "open": open_p,
        "close": close_p,
        "ceiling": ceil_p,
        "floor": flr_p,
        "adv20": adv20,
        "adv20_shares": adv20_shares,
        "s": s,
        "u": u,
        "m": m,
        "gap": gap,
        "r_oc": r_oc,
        "sigma": sigma,
        "adv": adv,
        "dates": dates,
        "symbols": symbols,
        "mask": mask,
    }


def nw_se(returns: np.ndarray, rf: float = 0.04 / 252.0, lags: int = 5) -> float:
    """Newey-West standard error for annualized Sharpe ratio (lags=5)."""
    returns_arr = np.asarray(returns, dtype=np.float64)
    ex = returns_arr - rf
    mu = float(np.mean(ex))
    sig = float(np.std(ex, ddof=1)) if len(ex) > 1 else float(np.std(ex))
    sr = mu / sig if sig > 1e-12 else 0.0
    u = (ex - mu) / sig if sig > 1e-12 else np.zeros_like(ex)
    g = u - 0.5 * sr * (u**2 - 1.0)
    n = len(g)
    lr_var = float(np.var(g))
    for l in range(1, lags + 1):
        if l >= n:
            break
        w_l = 1.0 - l / (lags + 1.0)
        lr_var += 2.0 * w_l * float(np.mean(g[l:] * g[:-l]))
    return float(np.sqrt(max(lr_var, 1e-12) / n) * np.sqrt(252.0))


def run_one(
    seed: int,
    rho: float,
    construction: Union[str, Tuple[int, int, int, float]],
    T: int = 947,
    N: int = 30,
    phi: float = 0.9,
    base_prices: Optional[Union[Dict[str, float], List[float], np.ndarray]] = None,
    dates: Optional[List[datetime.date]] = None,
    market: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute one full planted-alpha simulation under production HOSE pipeline."""
    if isinstance(construction, str):
        c_name = construction
        if c_name not in CONSTRUCTIONS:
            raise ValueError(f"Unknown construction '{c_name}', must be one of {list(CONSTRUCTIONS.keys())}")
        top_k_in, top_k_out, ema_span, max_daily_turnover = CONSTRUCTIONS[c_name]
    else:
        top_k_in, top_k_out, ema_span, max_daily_turnover = construction
        c_name = f"custom({top_k_in},{top_k_out},{ema_span},{max_daily_turnover})"

    if market is None:
        market = generate_planted_market(
            seed=seed,
            rho=rho,
            T=T,
            N=N,
            phi=phi,
            base_prices=base_prices,
            dates=dates,
        )

    open_p = market["open"]
    close_p = market["close"]
    ceil_p = market["ceiling"]
    flr_p = market["floor"]
    adv20 = market["adv20"]
    adv20_shares = market["adv20_shares"]
    s = market["s"]
    u = market["u"]
    mask = market["mask"]
    market_dates = market["dates"]

    w = ops.dampen_turnover(
        s,
        top_k_in=top_k_in,
        top_k_out=top_k_out,
        ema_span=ema_span,
        max_daily_turnover=max_daily_turnover,
        mask=mask,
        tie_break_key=adv20,
        invalid_below=None,
    )

    w_exec = np.zeros_like(w)
    w_exec[1:] = w[:-1]

    res = sim.simulate_vietnam(
        open_p,
        close_p,
        ceil_p,
        flr_p,
        adv20_shares,
        w_exec,
        dates=market_dates,
        penalty_rate_per_day=0.0,
        strict_t25_ato_lock=False,
    )

    naive = sim.simulate_naive(close_p, w_exec, raw_open=open_p)

    m = sim.evaluate_strategy(res["equity"], w_exec, naive)
    mg = sim.evaluate_strategy(naive, w_exec)

    sharpe_net = float(m.get("sharpe_ratio", 0.0))
    sharpe_gross = float(mg.get("sharpe_ratio", 0.0))
    daily_turnover = float(m.get("daily_turnover", 0.0))
    dsr_p_value = float(m.get("dsr_p_value", 1.0))

    eq = res["equity"]
    returns = (eq[1:] - eq[:-1]) / np.maximum(eq[:-1], 1e-6)
    se = nw_se(returns, rf=0.04 / 252.0, lags=5)
    t_nw = sharpe_net / se if se > 1e-12 else 0.0

    ic_real = cross_sectional_ic(s[:-1], close_p[1:] / open_p[1:] - 1.0)
    ic_latent = cross_sectional_ic(s[:-1], u[1:])

    # Excess return vs EW30 benchmark
    r_ew30 = np.mean(close_p[1:] / close_p[:-1] - 1.0, axis=1)
    a = returns - r_ew30
    mean_a = float(np.mean(a) * 252.0)
    vol_a = float(np.std(a) * np.sqrt(252.0))
    sr_a = mean_a / vol_a if vol_a > 1e-12 else 0.0
    se_a = nw_se(a, rf=0.0, lags=5)
    t_nw_excess = sr_a / se_a if se_a > 1e-12 else 0.0

    # EW30 arithmetic return for G6
    ew30_annualized = float(np.mean(r_ew30) * 252.0)

    return {
        "sharpe_net": sharpe_net,
        "sharpe_gross": sharpe_gross,
        "t_nw": t_nw,
        "dsr_p_value": dsr_p_value,
        "daily_turnover": daily_turnover,
        "ic_real": ic_real,
        "ic_latent": ic_latent,
        "t_nw_excess": t_nw_excess,
        "ew30_annualized": ew30_annualized,
        "seed": seed,
        "rho": rho,
        "construction": c_name,
        "phi": phi,
    }


def run_g2_timing_control(
    seeds: Optional[List[int]] = None,
    T: int = 947,
    N: int = 30,
) -> Dict[str, Any]:
    """Gate 2 timing control check (simulate_naive only, direct weights without dampener)."""
    if seeds is None:
        seeds = [1000, 1001, 1002, 1003, 1004]

    sharpes_i: List[float] = []
    sharpes_ii: List[float] = []
    sharpes_iii: List[float] = []

    for s in seeds:
        market = generate_planted_market(seed=s, rho=0.0, T=T, N=N)
        op = market["open"]
        cl = market["close"]

        # Clairvoyant signal: s_pf[t-1, i] = cl[t, i] / op[t, i] - 1
        s_pf = np.zeros((T, N), dtype=np.float64)
        s_pf[:-1] = cl[1:] / op[1:] - 1.0

        # Direct weights: top 5 equal weight (0.2 each)
        w_pf = np.zeros((T, N), dtype=np.float64)
        w_neg = np.zeros((T, N), dtype=np.float64)
        for t in range(T - 1):
            top5 = np.argsort(s_pf[t])[-5:]
            w_pf[t, top5] = 0.2
            bot5 = np.argsort(-s_pf[t])[-5:]
            w_neg[t, bot5] = 0.2

        # (i) Clairvoyant execution w_exec[1:] = w_pf[:-1]
        w_exec_i = np.zeros((T, N), dtype=np.float64)
        w_exec_i[1:] = w_pf[:-1]
        naive_i = sim.simulate_naive(cl, w_exec_i, raw_open=op)
        eval_i = sim.evaluate_strategy(naive_i, w_exec_i)
        sharpes_i.append(float(eval_i.get("sharpe_ratio", 0.0)))

        # (iii) Negated clairvoyant execution
        w_exec_iii = np.zeros((T, N), dtype=np.float64)
        w_exec_iii[1:] = w_neg[:-1]
        naive_iii = sim.simulate_naive(cl, w_exec_iii, raw_open=op)
        eval_iii = sim.evaluate_strategy(naive_iii, w_exec_iii)
        sharpes_iii.append(float(eval_iii.get("sharpe_ratio", 0.0)))

        # (ii) Lagged 1 additional day: w_exec lagged by 2 steps
        w_exec_ii = np.zeros((T, N), dtype=np.float64)
        if T > 2:
            w_exec_ii[2:] = w_pf[:-2]
        naive_ii = sim.simulate_naive(cl, w_exec_ii, raw_open=op)
        eval_ii = sim.evaluate_strategy(naive_ii, w_exec_ii)
        sharpes_ii.append(float(eval_ii.get("sharpe_ratio", 0.0)))

    mean_i = float(np.mean(sharpes_i))
    mean_ii = float(np.mean(sharpes_ii))
    mean_iii = float(np.mean(sharpes_iii))

    passed = (
        mean_i > 5.0
        and abs(mean_ii) < 1.0
        and all(abs(x) < 2.5 for x in sharpes_ii)
        and mean_iii < -5.0
    )

    return {
        "passed": passed,
        "mean_clairvoyant": mean_i,
        "sharpes_clairvoyant": sharpes_i,
        "mean_lagged": mean_ii,
        "sharpes_lagged": sharpes_ii,
        "mean_negated": mean_iii,
        "sharpes_negated": sharpes_iii,
    }


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    z = float(stats.norm.ppf((1.0 + confidence) / 2.0))
    denom = 1.0 + z**2 / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt((p * (1.0 - p) / n) + (z**2) / (4.0 * n**2))
    low = 0.0 if k == 0 else max(0.0, float(center - margin))
    high = 1.0 if k == n else min(1.0, float(center + margin))
    return low, high


def interpolate_rho_star(
    rho_grid: List[float],
    power_values: List[float],
    threshold: float = 0.8,
) -> Union[float, str]:
    """Linear interpolation of first rho where power reaches threshold."""
    if not rho_grid or not power_values:
        return "chưa đạt trong lưới"
    if power_values[0] >= threshold:
        return float(rho_grid[0])
    for i in range(1, len(rho_grid)):
        if power_values[i] >= threshold:
            p_prev = power_values[i - 1]
            p_curr = power_values[i]
            r_prev = rho_grid[i - 1]
            r_curr = rho_grid[i]
            if p_curr == p_prev:
                return float(r_curr)
            fraction = (threshold - p_prev) / (p_curr - p_prev)
            return float(r_prev + fraction * (r_curr - r_prev))
    return "chưa đạt trong lưới"


def _eval_seed_task(
    task_args: Tuple[str, int, float, float, int, int]
) -> Tuple[str, int, float, Dict[str, Dict[str, Any]]]:
    """Worker task evaluating all constructions for a single (seed, rho) market."""
    block, seed, rho, phi, T, N = task_args
    market = generate_planted_market(seed=seed, rho=rho, T=T, N=N, phi=phi)
    results: Dict[str, Dict[str, Any]] = {}
    for c_name in CONSTRUCTIONS:
        results[c_name] = run_one(
            seed=seed,
            rho=rho,
            construction=c_name,
            T=T,
            N=N,
            phi=phi,
            market=market,
        )
    return block, seed, rho, results


def run_grid(
    blocks: List[str],
    R: int = 100,
    R0: int = 400,
    T: int = 947,
    N: int = 30,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute complete planted-alpha simulation grid and aggregate results."""
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 8)

    seed_base = 1000

    # Build unique (block, seed, rho, phi, T, N) tasks
    tasks: List[Tuple[str, int, float, float, int, int]] = []
    for block in blocks:
        phi = 0.9 if block == "main" else 0.5
        r0_count = R0 if block == "main" else R
        seeds_for_null = list(range(seed_base, seed_base + r0_count))
        for s in seeds_for_null:
            tasks.append((block, s, 0.0, phi, T, N))

        rho_list = MAIN_RHO_GRID if block == "main" else PHI05_RHO_GRID
        seeds_paired = list(range(seed_base, seed_base + R))
        for rho in rho_list:
            if rho == 0.0:
                continue
            for s in seeds_paired:
                tasks.append((block, s, rho, phi, T, N))

    raw_results_by_block_rho: Dict[Tuple[str, float], Dict[int, Dict[str, Dict[str, Any]]]] = {}
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for blk, s_val, r_val, res_dict in executor.map(_eval_seed_task, tasks):
                key = (blk, r_val)
                if key not in raw_results_by_block_rho:
                    raw_results_by_block_rho[key] = {}
                raw_results_by_block_rho[key][s_val] = res_dict
    else:
        for t in tasks:
            blk, s_val, r_val, res_dict = _eval_seed_task(t)
            key = (blk, r_val)
            if key not in raw_results_by_block_rho:
                raw_results_by_block_rho[key] = {}
            raw_results_by_block_rho[key][s_val] = res_dict

    # Reconstruct raw_cells in exact seed order
    raw_cells: Dict[str, List[Dict[str, Any]]] = {}
    q95_map: Dict[str, float] = {}

    for block in blocks:
        phi = 0.9 if block == "main" else 0.5
        r0_count = R0 if block == "main" else R
        seeds_for_null = list(range(seed_base, seed_base + r0_count))

        for c_name in CONSTRUCTIONS:
            cell_key = f"{block}:{c_name}:rho_0.0"
            results_null = [
                raw_results_by_block_rho[(block, 0.0)][s][c_name]
                for s in seeds_for_null
            ]
            raw_cells[cell_key] = results_null
            t_nw_null = np.array([r["t_nw"] for r in results_null])
            q95_val = float(np.percentile(t_nw_null, 95.0))
            q95_map[f"{block}:{c_name}"] = q95_val

        rho_list = MAIN_RHO_GRID if block == "main" else PHI05_RHO_GRID
        seeds_paired = list(range(seed_base, seed_base + R))
        for rho in rho_list:
            if rho == 0.0:
                continue
            for c_name in CONSTRUCTIONS:
                cell_key = f"{block}:{c_name}:rho_{rho}"
                results_rho = [
                    raw_results_by_block_rho[(block, rho)][s][c_name]
                    for s in seeds_paired
                ]
                raw_cells[cell_key] = results_rho

    # 3. Aggregate each cell
    aggregated_cells: Dict[str, Any] = {}
    for block in blocks:
        phi = 0.9 if block == "main" else 0.5
        rho_list = MAIN_RHO_GRID if block == "main" else PHI05_RHO_GRID

        for c_name in CONSTRUCTIONS:
            q95 = q95_map[f"{block}:{c_name}"]

            for rho in rho_list:
                cell_key = f"{block}:{c_name}:rho_{rho}"
                runs = raw_cells[cell_key]
                n_runs = len(runs)

                sn_arr = np.array([r["sharpe_net"] for r in runs])
                sg_arr = np.array([r["sharpe_gross"] for r in runs])
                ic_real_arr = np.array([r["ic_real"] for r in runs])
                ic_latent_arr = np.array([r["ic_latent"] for r in runs])
                to_arr = np.array([r["daily_turnover"] for r in runs])
                tnw_excess_arr = np.array([r["t_nw_excess"] for r in runs])
                ew30_arr = np.array([r["ew30_annualized"] for r in runs])

                # Detections
                k_nom = sum(1 for r in runs if r["t_nw"] > 1.645)
                k_size = sum(1 for r in runs if r["t_nw"] > q95)
                k_dsr = sum(1 for r in runs if r["dsr_p_value"] < 0.05)

                nom_rate = k_nom / n_runs
                nom_ci = wilson_ci(k_nom, n_runs)
                size_rate = k_size / n_runs
                size_ci = wilson_ci(k_size, n_runs)
                dsr_rate = k_dsr / n_runs
                dsr_ci = wilson_ci(k_dsr, n_runs)

                per_seed_sn = {str(r["seed"]): round(r["sharpe_net"], 4) for r in runs}

                aggregated_cells[cell_key] = {
                    "block": block,
                    "construction": c_name,
                    "rho": rho,
                    "phi": phi,
                    "n_runs": n_runs,
                    "q95": q95,
                    "sharpe_net_mean": float(np.mean(sn_arr)),
                    "sharpe_net_std": float(np.std(sn_arr, ddof=1)) if n_runs > 1 else 0.0,
                    "sharpe_gross_mean": float(np.mean(sg_arr)),
                    "sharpe_gross_std": float(np.std(sg_arr, ddof=1)) if n_runs > 1 else 0.0,
                    "ic_real_mean": float(np.mean(ic_real_arr)),
                    "ic_real_std": float(np.std(ic_real_arr, ddof=1)) if n_runs > 1 else 0.0,
                    "ic_latent_mean": float(np.mean(ic_latent_arr)),
                    "ic_latent_std": float(np.std(ic_latent_arr, ddof=1)) if n_runs > 1 else 0.0,
                    "daily_turnover_mean": float(np.mean(to_arr)),
                    "t_nw_excess_mean": float(np.mean(tnw_excess_arr)),
                    "ew30_annualized_mean": float(np.mean(ew30_arr)),
                    "nominal_power": {
                        "rate": nom_rate,
                        "wilson_ci_95": list(nom_ci),
                    },
                    "size_adjusted_power": {
                        "rate": size_rate,
                        "wilson_ci_95": list(size_ci),
                    },
                    "dsr_power": {
                        "rate": dsr_rate,
                        "wilson_ci_95": list(dsr_ci),
                    },
                    "per_seed_sharpe_net": per_seed_sn,
                }

    # 4. Primary answers: rho* for each block & construction
    primary_answers: Dict[str, Any] = {}
    for block in blocks:
        rho_list = MAIN_RHO_GRID if block == "main" else PHI05_RHO_GRID
        for c_name in CONSTRUCTIONS:
            p_size_rates = [
                aggregated_cells[f"{block}:{c_name}:rho_{r}"]["size_adjusted_power"]["rate"]
                for r in rho_list
            ]
            p_size_lows = [
                aggregated_cells[f"{block}:{c_name}:rho_{r}"]["size_adjusted_power"]["wilson_ci_95"][0]
                for r in rho_list
            ]
            p_size_highs = [
                aggregated_cells[f"{block}:{c_name}:rho_{r}"]["size_adjusted_power"]["wilson_ci_95"][1]
                for r in rho_list
            ]

            p_nom_rates = [
                aggregated_cells[f"{block}:{c_name}:rho_{r}"]["nominal_power"]["rate"]
                for r in rho_list
            ]
            p_nom_lows = [
                aggregated_cells[f"{block}:{c_name}:rho_{r}"]["nominal_power"]["wilson_ci_95"][0]
                for r in rho_list
            ]
            p_nom_highs = [
                aggregated_cells[f"{block}:{c_name}:rho_{r}"]["nominal_power"]["wilson_ci_95"][1]
                for r in rho_list
            ]

            # Primary: size-adjusted (point, low from upper Wilson, high from lower Wilson)
            rho_star_pt = interpolate_rho_star(rho_list, p_size_rates, 0.8)
            rho_star_low = interpolate_rho_star(rho_list, p_size_highs, 0.8)
            rho_star_high = interpolate_rho_star(rho_list, p_size_lows, 0.8)

            # Contrast: nominal
            rho_nom_pt = interpolate_rho_star(rho_list, p_nom_rates, 0.8)
            rho_nom_low = interpolate_rho_star(rho_list, p_nom_highs, 0.8)
            rho_nom_high = interpolate_rho_star(rho_list, p_nom_lows, 0.8)

            ans_key = f"{block}:{c_name}"
            primary_answers[ans_key] = {
                "size_adjusted": {
                    "rho_star_point": rho_star_pt,
                    "rho_star_low": rho_star_low,
                    "rho_star_high": rho_star_high,
                },
                "nominal": {
                    "rho_star_point": rho_nom_pt,
                    "rho_star_low": rho_nom_low,
                    "rho_star_high": rho_nom_high,
                },
            }

    # 5. Evaluate Gates G1-G6
    gates_report = evaluate_gates(
        aggregated_cells=aggregated_cells,
        raw_cells=raw_cells,
        blocks=blocks,
        T=T,
        N=N,
    )

    return {
        "metadata": {
            "T": T,
            "N": N,
            "R": R,
            "R0": R0,
            "blocks": blocks,
            "constructions": CONSTRUCTIONS,
            "versions": {
                "numpy": np.__version__,
                "scipy": stats.__file__ and scipy_version(),
                "numba": numba.__version__,
            },
        },
        "gates": gates_report,
        "primary_answers": primary_answers,
        "cells": aggregated_cells,
    }


def scipy_version() -> str:
    import scipy
    return scipy.__version__


def evaluate_gates(
    aggregated_cells: Dict[str, Any],
    raw_cells: Dict[str, List[Dict[str, Any]]],
    blocks: List[str],
    T: int,
    N: int,
) -> Dict[str, Any]:
    """Evaluate frozen gates G1-G6."""
    results: Dict[str, Any] = {}

    # G1a & G1b (checked across main block cells, or available blocks)
    g1a_failures: List[str] = []
    g1b_failures: List[str] = []

    eval_block = "main" if "main" in blocks else blocks[0]
    rho_grid = MAIN_RHO_GRID if eval_block == "main" else PHI05_RHO_GRID

    for rho in rho_grid:
        # Average ic_real and ic_latent across constructions
        ic_reals = [
            aggregated_cells[f"{eval_block}:{c}:rho_{rho}"]["ic_real_mean"]
            for c in CONSTRUCTIONS
        ]
        ic_latents = [
            aggregated_cells[f"{eval_block}:{c}:rho_{rho}"]["ic_latent_mean"]
            for c in CONSTRUCTIONS
        ]
        mean_ic_real = float(np.mean(ic_reals))
        mean_ic_latent = float(np.mean(ic_latents))

        # G1a: |mean(ic_real) - rho| <= 0.0025 + 0.05*rho
        tol_g1a = 0.0025 + 0.05 * rho
        diff_g1a = abs(mean_ic_real - rho)
        if diff_g1a > tol_g1a:
            g1a_failures.append(f"rho={rho}: diff={diff_g1a:.5f} > tol={tol_g1a:.5f}")

        # G1b: |mean(ic_latent) - rho*(1 - (1 - rho^2)/(2N))| <= 0.0025
        expected_latent = rho * (1.0 - (1.0 - rho**2) / (2.0 * N))
        diff_g1b = abs(mean_ic_latent - expected_latent)
        if diff_g1b > 0.0025:
            g1b_failures.append(f"rho={rho}: diff={diff_g1b:.5f} > tol=0.0025")

    results["G1a"] = {
        "passed": len(g1a_failures) == 0,
        "failures": g1a_failures,
    }
    results["G1b"] = {
        "passed": len(g1b_failures) == 0,
        "failures": g1b_failures,
    }

    # G2 (Timing control)
    g2_res = run_g2_timing_control(seeds=[1000, 1001, 1002, 1003, 1004], T=T, N=N)
    results["G2"] = g2_res

    # G3 (Size inflation reporting at rho=0)
    g3_flags: List[str] = []
    g3_summary: Dict[str, Any] = {}
    for c in CONSTRUCTIONS:
        c_null = aggregated_cells[f"{eval_block}:{c}:rho_0.0"]
        nom_rate = c_null["nominal_power"]["rate"]
        nom_ci = c_null["nominal_power"]["wilson_ci_95"]
        dsr_rate = c_null["dsr_power"]["rate"]
        dsr_ci = c_null["dsr_power"]["wilson_ci_95"]
        inflated = nom_ci[0] > 0.10
        if inflated:
            g3_flags.append(f"{c}: lower Wilson {nom_ci[0]:.3f} > 0.10 (tiêu chí danh nghĩa bị thổi phồng kích cỡ)")
        g3_summary[c] = {
            "nominal_rate": nom_rate,
            "nominal_ci": nom_ci,
            "dsr_rate": dsr_rate,
            "dsr_ci": dsr_ci,
            "inflated": inflated,
        }
    results["G3"] = {
        "passed": True,  # Non-blocking, reporting only
        "flags": g3_flags,
        "summary": g3_summary,
    }

    # G4 (Monotonicity in main block)
    g4_passed = True
    g4_details: Dict[str, Any] = {}
    if eval_block == "main":
        for c in CONSTRUCTIONS:
            runs_rho0 = raw_cells[f"main:{c}:rho_0.0"][:100]
            runs_rho15 = raw_cells[f"main:{c}:rho_0.15"]
            diffs = np.array([r15["sharpe_net"] - r0["sharpe_net"] for r0, r15 in zip(runs_rho0, runs_rho15)])
            mean_diff = float(np.mean(diffs))
            se_diff = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
            diff_pass = mean_diff > 3.0 * se_diff

            # Spearman correlation on 8 cells
            sn_means = [
                aggregated_cells[f"main:{c}:rho_{r}"]["sharpe_net_mean"]
                for r in MAIN_RHO_GRID
            ]
            spearman_corr, _ = stats.spearmanr(MAIN_RHO_GRID, sn_means)
            spearman_pass = float(spearman_corr) >= 0.8

            c_pass = diff_pass and spearman_pass
            if not c_pass:
                g4_passed = False
            g4_details[c] = {
                "passed": c_pass,
                "mean_diff_15_vs_0": mean_diff,
                "se_diff": se_diff,
                "z_diff": float(mean_diff / se_diff) if se_diff > 1e-12 else 0.0,
                "spearman_corr": float(spearman_corr),
            }
    results["G4"] = {
        "passed": g4_passed,
        "details": g4_details,
    }

    # G5 (Invariants)
    reg_path = Path("experiments/trial_registry.jsonl")
    reg_ok = False
    reg_hash = ""
    if reg_path.exists():
        reg_bytes = reg_path.read_bytes()
        reg_hash = hashlib.sha256(reg_bytes).hexdigest()
        reg_lines = len(reg_path.read_text(encoding="utf-8").splitlines())
        reg_ok = (reg_hash.upper() == "DF5A3FE886452E33EDEE62B2A0D3A34CC0869D62F61319C700FC4275A53E041D" and reg_lines == 26)

    pq_path = Path("data/vn30_production.parquet")
    pq_ok = False
    pq_hash = ""
    if pq_path.exists():
        pq_hash = hashlib.sha256(pq_path.read_bytes()).hexdigest()
        pq_ok = (pq_hash.upper() == "8A3BD271D5FAB5750D427712C0BA5715FDC9A9F638F1195661CC5E18996EB0CB")

    results["G5"] = {
        "passed": reg_ok and pq_ok,
        "registry_hash": reg_hash,
        "registry_intact": reg_ok,
        "parquet_hash": pq_hash,
        "parquet_intact": pq_ok,
    }

    # G6 (Lack of drift under null)
    # At rho = 0, |mean across seeds of (annualized arithmetic return of R_ew30 - 0.04)| <= 0.02
    ew30_means = [
        aggregated_cells[f"{eval_block}:{c}:rho_0.0"]["ew30_annualized_mean"]
        for c in CONSTRUCTIONS
    ]
    mean_ew30 = float(np.mean(ew30_means))
    drift_diff = abs(mean_ew30 - 0.04)
    results["G6"] = {
        "passed": drift_diff <= 0.02,
        "annualized_ew30_mean": mean_ew30,
        "diff_vs_rf": drift_diff,
        "tolerance": 0.02,
    }

    return results


def format_markdown_table(data: Dict[str, Any]) -> str:
    """Format power test results into a GitHub-Flavored Markdown table."""
    cells = data["cells"]
    lines = []
    lines.append("| Block | Construction | rho | Seeds | Sharpe Net (mean+/-std) | Sharpe Gross | Daily TO | Real IC | Latent IC | Size-Adj Power (Wilson 95%) | Nominal Power | DSR Power | t_NW Excess |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for key, cell in cells.items():
        blk = cell["block"]
        c_name = cell["construction"]
        rho = cell["rho"]
        n_runs = cell["n_runs"]
        sn_m, sn_s = cell["sharpe_net_mean"], cell["sharpe_net_std"]
        sg_m = cell["sharpe_gross_mean"]
        to_m = cell["daily_turnover_mean"]
        ic_r = cell["ic_real_mean"]
        ic_l = cell["ic_latent_mean"]
        sa_p = cell["size_adjusted_power"]["rate"]
        sa_ci = cell["size_adjusted_power"]["wilson_ci_95"]
        nom_p = cell["nominal_power"]["rate"]
        nom_ci = cell["nominal_power"]["wilson_ci_95"]
        dsr_p = cell["dsr_power"]["rate"]
        dsr_ci = cell["dsr_power"]["wilson_ci_95"]
        tnw_ex = cell["t_nw_excess_mean"]

        sa_str = f"{sa_p:.2f} [{sa_ci[0]:.2f}, {sa_ci[1]:.2f}]"
        nom_str = f"{nom_p:.2f} [{nom_ci[0]:.2f}, {nom_ci[1]:.2f}]"
        dsr_str = f"{dsr_p:.2f} [{dsr_ci[0]:.2f}, {dsr_ci[1]:.2f}]"

        lines.append(
            f"| {blk} | {c_name} | {rho:.3f} | {n_runs} | {sn_m:+.2f} +/- {sn_s:.2f} | {sg_m:+.2f} | {to_m*100:.2f}% | {ic_r:.4f} | {ic_l:.4f} | {sa_str} | {nom_str} | {dsr_str} | {tnw_ex:+.2f} |"
        )

    return "\n".join(lines)


def save_results_json(data: Dict[str, Any], out_path: Path) -> None:
    """Save grid results to JSON with overwrite protection and no absolute paths."""
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"Output file already exists: {out_path.name}. Overwrite is forbidden.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_str = json.dumps(data, indent=2) + "\n"
    out_path.write_text(json_str, encoding="utf-8", newline="\n")


def format_gates_text(gates: Dict[str, Any]) -> str:
    """Format gate evaluation results into human-readable text."""
    lines = ["=== PLANTED-ALPHA POWER TEST GATE VERIFICATION REPORT ==="]
    for gate_name in ["G1a", "G1b", "G2", "G3", "G4", "G5", "G6"]:
        info = gates.get(gate_name, {})
        status = "PASSED" if info.get("passed", False) else "FAILED"
        lines.append(f"\n[{gate_name}] Status: {status}")
        for k, v in info.items():
            if k != "passed":
                lines.append(f"  - {k}: {v}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Planted-alpha power test grid runner.")
    parser.add_argument("--blocks", type=str, default="main,phi05", help="Comma-separated blocks (e.g. main,phi05).")
    parser.add_argument("--R", type=int, default=100, help="Number of seeds for rho > 0.")
    parser.add_argument("--R0", type=int, default=400, help="Number of seeds for rho == 0.")
    parser.add_argument("--T", type=int, default=947, help="Number of trading sessions.")
    parser.add_argument("--N", type=int, default=30, help="Number of symbols.")
    parser.add_argument("--out", type=str, default="experiments/power_test_r18.json", help="Output JSON path.")
    parser.add_argument("--gates-out", type=str, default="audit_tmp/r18/gates.txt", help="Output path for raw gates report.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: min(CPU, 8)).")
    parser.add_argument("--print-table", action="store_true", help="Print Markdown table to stdout.")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke grid (R=2, R0=4, T=200).")

    args = parser.parse_args()

    if args.quick:
        blocks = ["main"]
        r_val = 2
        r0_val = 4
        t_val = 200
    else:
        blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]
        r_val = args.R
        r0_val = args.R0
        t_val = args.T

    data = run_grid(blocks=blocks, R=r_val, R0=r0_val, T=t_val, N=args.N, max_workers=args.workers)

    if args.out:
        out_p = Path(args.out)
        save_results_json(data, out_p)
        print(f"Results saved to: {out_p.name}")

    if args.gates_out:
        gates_p = Path(args.gates_out)
        gates_p.parent.mkdir(parents=True, exist_ok=True)
        gates_p.write_text(format_gates_text(data["gates"]), encoding="utf-8")
        print(f"Gates report saved to: {gates_p.name}")

    if args.print_table:
        print("\n" + format_markdown_table(data) + "\n")


if __name__ == "__main__":
    main()
