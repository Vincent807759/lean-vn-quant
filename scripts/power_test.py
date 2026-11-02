"""Planted-alpha power test market generator and evaluation pipeline.

Simulates synthetic market environments with planted cross-sectional alpha (IC = rho)
under realistic HOSE microstructure constraints (fees, taxes, T+2.5, tick sizes, limits).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import lfilter

from engine import alpha_ops as ops, data_pipeline as dp, vn_simulator as sim

CONSTRUCTIONS: Dict[str, Tuple[int, int, int, float]] = {
    "C1": (2, 4, 20, 0.015),
    "C2": (10, 18, 10, 0.05),
    "C3": (2, 4, 10, 0.05),
}


def round_tick_np(prices: np.ndarray) -> np.ndarray:
    """Vectorized HOSE tick rounding matching data_pipeline.round_tick_hose.

    - Price < 10,000 VND: tick size = 10 VND
    - 10,000 <= Price < 50,000 VND: tick size = 50 VND
    - Price >= 50,000 VND: tick size = 100 VND
    """
    prices = np.asarray(prices, dtype=np.float64)
    tick = np.where(prices < 10000.0, 10.0, np.where(prices < 50000.0, 50.0, 100.0))
    return np.round(prices / tick) * tick


def hose_limits_np(ref_prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized HOSE ceiling and floor limits matching compute_hose_limits.

    Ceiling is +7% rounded down to tick size.
    Floor is -7% rounded up to tick size.
    """
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
    """Draw latent market processes using 4 independent RNG streams from SeedSequence.

    Streams:
    1. rng_signal: s[0] and innovations z for AR(1) signal
    2. rng_eps: idiosyncratic return shocks eps
    3. rng_market: market parameter sigma_i, adv_i, and market return m[t]
    4. rng_gap: overnight gap shocks xi

    Returns:
        s: Signal matrix of shape (T, N)
        u: Normalized idiosyncratic return of shape (T, N)
        m: Market return vector of shape (T,)
        gap: Overnight log-return matrix of shape (T, N)
        r_oc: Intraday log-return matrix of shape (T, N)
        sigma: Volatility vector of shape (N,)
        adv: Average daily turnover (VND) vector of shape (N,)
    """
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
    """Mean cross-sectional Pearson correlation across rows (days).

    If cross-sectional std < 1e-12, daily IC is 0.0.
    """
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
    """Generate synthetic market prices with planted alpha under HOSE constraints.

    Recursion:
    - t = 0: ref[0] = open[0] = close[0] = round_tick(BASE_PRICES); ceiling[0], floor[0] from ref[0]
    - t >= 1: ref[t] = close[t-1]; ceiling[t], floor[t] = limits(ref[t])
      open[t] = clip(round_tick(ref[t] * exp(gap[t])), floor[t], ceiling[t])
      close[t] = clip(round_tick(open[t] * exp(r_oc[t])), floor[t], ceiling[t])
    """
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
    """Newey-West standard error for annualized Sharpe ratio (lags=5).

    Matches implementation in scripts/compare_backtest_b6_b11.py.
    """
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

    return {
        "sharpe_net": sharpe_net,
        "sharpe_gross": sharpe_gross,
        "t_nw": t_nw,
        "dsr_p_value": dsr_p_value,
        "daily_turnover": daily_turnover,
        "ic_real": ic_real,
        "ic_latent": ic_latent,
        "t_nw_excess": t_nw_excess,
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
    """Gate 2 timing control check (simulate_naive only, direct weights without dampener).

    (i) Clairvoyant signal s_pf[t-1, i] = close[t, i]/open[t, i] - 1: mean sharpe_gross > 5
    (ii) Signal lagged 1 additional day: |mean sharpe_gross| < 1.0 and each |sharpe_gross| < 2.5
    (iii) Negated clairvoyant signal: mean sharpe_gross < -5
    """
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
