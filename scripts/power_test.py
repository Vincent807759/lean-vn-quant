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

from engine import data_pipeline as dp


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
