"""Unit tests for planted-alpha market generator (HUONG_DI_R18 commit a)."""

from __future__ import annotations

import numpy as np
import pytest

from engine import data_pipeline as dp
from scripts.power_test import (
    cross_sectional_ic,
    draw_latents,
    generate_planted_market,
    hose_limits_np,
    round_tick_np,
)


def test_vectorized_tick_and_limits_equivalence():
    """Verify round_tick_np and hose_limits_np match scalar counterparts on >= 10,000 prices."""
    rng = np.random.default_rng(20261102)
    test_prices = rng.uniform(5000.0, 150000.0, size=12000)

    # 1. Tick rounding equivalence
    np_rounded = round_tick_np(test_prices)
    scalar_rounded = np.array([dp.round_tick_hose(float(p)) for p in test_prices])
    assert np.array_equal(np_rounded, scalar_rounded)

    # 2. Ceiling / Floor equivalence
    ceil_np, flr_np = hose_limits_np(test_prices)
    scalar_limits = [dp.compute_hose_limits(float(p)) for p in test_prices]
    ceil_scalar = np.array([item[0] for item in scalar_limits])
    flr_scalar = np.array([item[1] for item in scalar_limits])
    assert np.array_equal(ceil_np, ceil_scalar)
    assert np.array_equal(flr_np, flr_scalar)


def test_planted_market_shapes_and_invariants():
    """Verify shapes, types, finiteness, bounds, tick conformity, and day-0 equality."""
    T, N = 947, 30
    res = generate_planted_market(seed=1001, rho=0.05, T=T, N=N, phi=0.9)

    open_p = res["open"]
    close_p = res["close"]
    ceil_p = res["ceiling"]
    flr_p = res["floor"]
    adv20_shares = res["adv20_shares"]
    s = res["s"]

    # Shapes and types
    assert open_p.shape == (T, N)
    assert close_p.shape == (T, N)
    assert ceil_p.shape == (T, N)
    assert flr_p.shape == (T, N)
    assert adv20_shares.shape == (T, N)
    assert s.shape == (T, N)

    # Finiteness
    assert np.all(np.isfinite(open_p))
    assert np.all(np.isfinite(close_p))
    assert np.all(np.isfinite(ceil_p))
    assert np.all(np.isfinite(flr_p))
    assert np.all(np.isfinite(adv20_shares))
    assert np.all(np.isfinite(s))

    # Day 0 invariant: open[0] == close[0]
    np.testing.assert_array_equal(open_p[0], close_p[0])

    # Bound invariants: floor <= open <= ceiling and floor <= close <= ceiling
    assert np.all(flr_p <= open_p + 1e-9)
    assert np.all(open_p <= ceil_p + 1e-9)
    assert np.all(flr_p <= close_p + 1e-9)
    assert np.all(close_p <= ceil_p + 1e-9)

    # Tick conformity
    np.testing.assert_array_equal(open_p, round_tick_np(open_p))
    np.testing.assert_array_equal(close_p, round_tick_np(close_p))


def test_determinism_and_common_random_numbers():
    """Verify same seed produces identical results, CRN property, and global RNG unchanged."""
    global_rng_state_before = np.random.get_state()

    # Same seed -> bitwise identical
    m1 = generate_planted_market(seed=1234, rho=0.03, T=100, N=10)
    m2 = generate_planted_market(seed=1234, rho=0.03, T=100, N=10)
    np.testing.assert_array_equal(m1["open"], m2["open"])
    np.testing.assert_array_equal(m1["close"], m2["close"])
    np.testing.assert_array_equal(m1["s"], m2["s"])

    # Changing rho with same seed: signal and gap must be identical (CRN)
    lat_rho0 = draw_latents(seed=5555, rho=0.0, T=200, N=20)
    lat_rho1 = draw_latents(seed=5555, rho=0.1, T=200, N=20)
    np.testing.assert_array_equal(lat_rho0[0], lat_rho1[0])  # s
    np.testing.assert_array_equal(lat_rho0[3], lat_rho1[3])  # gap
    np.testing.assert_array_equal(lat_rho0[2], lat_rho1[2])  # m
    np.testing.assert_array_equal(lat_rho0[5], lat_rho1[5])  # sigma
    np.testing.assert_array_equal(lat_rho0[6], lat_rho1[6])  # adv

    # Global RNG state must not have changed
    global_rng_state_after = np.random.get_state()
    assert global_rng_state_before[0] == global_rng_state_after[0]
    np.testing.assert_array_equal(global_rng_state_before[1], global_rng_state_after[1])


def test_jensen_expectation():
    """Verify E[exp(gap + r_oc) - 1] = 0.04/252 within 4 standard errors for rho=0 and rho=0.1."""
    mu_f = 0.04 / 252.0
    for rho in [0.0, 0.1]:
        _s, _u, _m, gap, r_oc, _sigma, _adv = draw_latents(seed=42, rho=rho, T=200000, N=30)
        tot_ret = np.exp(gap + r_oc) - 1.0
        mean_ret = float(np.mean(tot_ret))
        se = float(np.std(tot_ret, ddof=1) / np.sqrt(tot_ret.size))
        z_score = abs(mean_ret - mu_f) / se
        assert z_score < 4.0, f"rho={rho}: z_score={z_score:.2f} >= 4.0"


def test_latent_ic_expectation():
    """Verify |ic_latent - rho*(1 - (1 - rho^2)/(2N))| < 0.01 at T=20,000."""
    T, N = 20000, 30
    for rho in [0.0, 0.05, 0.10]:
        s, u, _m, _gap, _r_oc, _sigma, _adv = draw_latents(seed=777, rho=rho, T=T, N=N)
        ic_latent = cross_sectional_ic(s[:-1], u[1:])
        expected_ic = rho * (1.0 - (1.0 - rho**2) / (2.0 * N))
        diff = abs(ic_latent - expected_ic)
        assert diff < 0.01, f"rho={rho}: diff={diff:.5f} >= 0.01"


def test_signal_gap_independence():
    """Verify correlation between s[t-1] and gap[t] is < 0.02."""
    s, _u, _m, gap, _r_oc, _sigma, _adv = draw_latents(seed=888, rho=0.05, T=20000, N=30)
    corr = float(np.corrcoef(s[:-1].ravel(), gap[1:].ravel())[0, 1])
    assert abs(corr) < 0.02, f"corr={corr:.5f} >= 0.02"


def test_adv20_shares_magnitude():
    """Verify 1e5 <= adv20_shares.mean() <= 5e7."""
    res = generate_planted_market(seed=999, rho=0.05, T=947, N=30)
    adv_mean = float(res["adv20_shares"].mean())
    assert 1e5 <= adv_mean <= 5e7, f"adv20_shares.mean={adv_mean:.1f} out of range"
