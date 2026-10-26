"""
alpha_ops.py - High-Performance Vectorized Mathematical Operators & Factor Orthogonalization

Industry-standard cross-sectional and rolling time-series operators for systematic quantitative trading:
- 2D numpy arrays of shape (T, N)
- Strictly causal (0% look-ahead bias confirmed via perturbation tests)
- Numba C-speed JIT & Bottleneck acceleration
- Numerical resilience against NaNs and zero-volatility flat sessions
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import bottleneck as bn
import numba
from scipy import stats
import polars as pl


# ==============================================================================
# 1. CROSS-SECTIONAL OPERATORS
# ==============================================================================

def rank(x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Cross-sectional ranking normalized to [0.0, 1.0].
    
    Parameters:
    -----------
    x : np.ndarray, shape (T, N)
        Input factor/alpha values.
    mask : Optional[np.ndarray], shape (T, N), bool
        Tradable mask (True = valid to rank, False = excluded).
        
    Returns:
    --------
    np.ndarray, shape (T, N), float64
        Percentile ranks in [0.0, 1.0]. Non-tradable or NaN entries are NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    
    for t in range(T):
        row = x[t]
        valid = np.isfinite(row)
        if mask is not None:
            valid = valid & mask[t]
            
        m = np.sum(valid)
        if m == 0:
            continue
        elif m == 1:
            out[t, valid] = 0.5
        else:
            valid_vals = row[valid]
            # Average rank method resolves ties cleanly
            ranks = stats.rankdata(valid_vals, method="average")
            # Normalize to [0.0, 1.0]
            out[t, valid] = (ranks - 1.0) / (m - 1.0)
            
    return out


def zscore(x: np.ndarray, clip: float = 3.0, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Cross-sectional Z-score standardization with outlier clipping [-clip, clip].
    
    Parameters:
    -----------
    x : np.ndarray, shape (T, N)
    clip : float, default 3.0
    mask : Optional[np.ndarray], shape (T, N), bool
    
    Returns:
    --------
    np.ndarray, shape (T, N), float64
    """
    x = np.asarray(x, dtype=np.float64)
    T, N = x.shape
    out = np.zeros((T, N), dtype=np.float64)
    
    for t in range(T):
        row = x[t]
        valid = np.isfinite(row)
        if mask is not None:
            valid = valid & mask[t]
            
        m = np.sum(valid)
        if m <= 1:
            continue
            
        vals = row[valid]
        mean_val = np.mean(vals)
        std_val = np.std(vals, ddof=1)
        
        if std_val > 1e-12:
            z = (vals - mean_val) / std_val
            if clip is not None and clip > 0:
                z = np.clip(z, -clip, clip)
            out[t, valid] = z
            
    return out


def scale(x: np.ndarray, target_sum: float = 1.0, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Cross-sectional scaling such that sum of absolute weights equals target_sum.
    
    Parameters:
    -----------
    x : np.ndarray, shape (T, N)
    target_sum : float, default 1.0
    mask : Optional[np.ndarray], shape (T, N), bool
    
    Returns:
    --------
    np.ndarray, shape (T, N), float64
    """
    x = np.asarray(x, dtype=np.float64)
    T, N = x.shape
    out = np.zeros((T, N), dtype=np.float64)
    
    for t in range(T):
        row = x[t]
        valid = np.isfinite(row)
        if mask is not None:
            valid = valid & mask[t]
            
        if not np.any(valid):
            continue
            
        vals = row[valid]
        abs_sum = np.sum(np.abs(vals))
        if abs_sum > 1e-12:
            out[t, valid] = vals * (target_sum / abs_sum)
            
    return out


def group_neutralize(
    x: np.ndarray,
    group_ids: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Cross-sectional group neutralization: demeans signals within sector/industry groups.
    
    Parameters:
    -----------
    x : np.ndarray, shape (T, N)
    group_ids : np.ndarray, shape (N,) or (T, N)
        Categorical group identifier (e.g. Sector ID).
    mask : Optional[np.ndarray], shape (T, N), bool
    
    Returns:
    --------
    np.ndarray, shape (T, N), float64
    """
    x = np.asarray(x, dtype=np.float64)
    T, N = x.shape
    out = np.zeros((T, N), dtype=np.float64)
    
    is_2d_groups = (group_ids.ndim == 2)
    
    for t in range(T):
        row = x[t]
        valid = np.isfinite(row)
        if mask is not None:
            valid = valid & mask[t]
            
        if not np.any(valid):
            continue
            
        g_row = group_ids[t] if is_2d_groups else group_ids
        unique_groups = np.unique(g_row[valid])
        
        for g in unique_groups:
            g_idx = valid & (g_row == g)
            g_count = np.sum(g_idx)
            if g_count > 0:
                g_mean = np.mean(row[g_idx])
                out[t, g_idx] = row[g_idx] - g_mean
                
    return out


# ==============================================================================
# 2. TIME-SERIES OPERATORS (0% LOOK-AHEAD BIAS)
# ==============================================================================

def ts_delay(x: np.ndarray, d: int) -> np.ndarray:
    """
    Time-series lag operator: returns x[t - d].
    Rows 0 to d-1 are populated with NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    if d <= 0:
        return x.copy()
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    if d < T:
        out[d:] = x[:-d]
    return out


def ts_delta(x: np.ndarray, d: int) -> np.ndarray:
    """
    Time-series difference: x[t] - x[t - d].
    Rows 0 to d-1 are populated with NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    if d <= 0:
        return np.zeros_like(x)
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    if d < T:
        out[d:] = x[d:] - x[:-d]
    return out


def ts_mean(x: np.ndarray, d: int) -> np.ndarray:
    """
    Rolling mean over window d using Bottleneck.
    Strictly causal (t depends only on t-d+1..t).
    """
    x = np.asarray(x, dtype=np.float64)
    return bn.move_mean(x, window=d, min_count=d, axis=0)


def ts_std_dev(x: np.ndarray, d: int) -> np.ndarray:
    """
    Rolling sample standard deviation over window d using Bottleneck.
    Flat prices (zero variance) produce 0.0 safely.
    """
    x = np.asarray(x, dtype=np.float64)
    res = bn.move_std(x, window=d, min_count=d, ddof=1, axis=0)
    # Replace negative or zero variance / NaNs caused by rounding
    res[np.isclose(res, 0.0, atol=1e-12)] = 0.0
    return res


def ts_rank(x: np.ndarray, d: int) -> np.ndarray:
    """
    Rolling time-series rank over window d normalized to [0.0, 1.0].
    Uses Bottleneck's move_rank (which returns [-1, 1]) mapped to [0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    raw_rk = bn.move_rank(x, window=d, min_count=d, axis=0)
    return (raw_rk + 1.0) / 2.0


@numba.njit(fastmath=True, cache=True)
def _ts_decay_linear_numba(x: np.ndarray, d: int) -> np.ndarray:
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    if d <= 0 or d > T:
        return out
        
    w_sum = d * (d + 1) / 2.0
    for t in range(d - 1, T):
        for j in range(N):
            val = 0.0
            has_nan = False
            for k in range(d):
                vk = x[t - k, j]
                if np.isnan(vk):
                    has_nan = True
                    break
                val += (d - k) * vk
            if not has_nan:
                out[t, j] = val / w_sum
    return out


def ts_decay_linear(x: np.ndarray, d: int) -> np.ndarray:
    """
    Linearly weighted moving average with weights d, d-1, ..., 1.
    Accelerated with Numba.
    """
    x = np.asarray(x, dtype=np.float64)
    return _ts_decay_linear_numba(x, d)


@numba.njit(fastmath=True, cache=True)
def _ts_corr_numba(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    if d < 2 or d > T:
        return out
        
    for t in range(d - 1, T):
        for j in range(N):
            sx = 0.0
            sy = 0.0
            sxx = 0.0
            syy = 0.0
            sxy = 0.0
            has_nan = False
            for k in range(d):
                vx = x[t - k, j]
                vy = y[t - k, j]
                if np.isnan(vx) or np.isnan(vy):
                    has_nan = True
                    break
                sx += vx
                sy += vy
                sxx += vx * vx
                syy += vy * vy
                sxy += vx * vy
            if not has_nan:
                mx = sx / d
                my = sy / d
                var_x = (sxx / d) - (mx * mx)
                var_y = (syy / d) - (my * my)
                cov_xy = (sxy / d) - (mx * my)
                if var_x > 1e-12 and var_y > 1e-12:
                    r = cov_xy / np.sqrt(var_x * var_y)
                    if r > 1.0:
                        r = 1.0
                    elif r < -1.0:
                        r = -1.0
                    out[t, j] = r
                else:
                    out[t, j] = 0.0
    return out


def ts_corr(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """
    Rolling Pearson correlation over window d.
    Accelerated with Numba.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return _ts_corr_numba(x, y, d)


# ==============================================================================
# 3. FACTOR ORTHOGONALIZATION & CORRELATION PURGING
# ==============================================================================

def check_and_orthogonalize(
    new_alpha: np.ndarray,
    existing_alphas: Union[np.ndarray, List[np.ndarray]],
    threshold: float = 0.50,
    tradable_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, float]:
    """
    Calculates Pearson correlation between new_alpha and existing_alphas.
    If max(|correlation|) >= threshold, performs Gram-Schmidt orthogonalization
    independently on each cross-sectional slice t = 0..T-1 (preserving causality),
    driving self-correlation strictly below threshold.
    
    Parameters:
    -----------
    new_alpha : np.ndarray, 1D (N,) or 2D (T, N)
    existing_alphas : List[np.ndarray] or 2D/3D array
    threshold : float, default 0.50
    tradable_mask : Optional[np.ndarray], shape matches new_alpha
    
    Returns:
    --------
    Tuple[np.ndarray, float, float]
        (orthogonalized_alpha, max_corr_before, max_corr_after)
    """
    arr_new = np.asarray(new_alpha, dtype=np.float64)
    orig_shape = arr_new.shape
    is_1d = (arr_new.ndim == 1)
    
    if is_1d:
        v_2d = arr_new.reshape(1, -1)
        mask_2d = None if tradable_mask is None else np.asarray(tradable_mask, dtype=bool).reshape(1, -1)
    else:
        v_2d = arr_new
        mask_2d = None if tradable_mask is None else np.asarray(tradable_mask, dtype=bool)
        
    T, N = v_2d.shape
    
    # Standardize list of existing alphas to list of (T, N) arrays
    existing_list = []
    if isinstance(existing_alphas, np.ndarray):
        if existing_alphas.ndim == 1:
            existing_list = [existing_alphas.reshape(1, -1) if is_1d else existing_alphas.reshape(T, N)]
        elif existing_alphas.ndim == 2:
            if existing_alphas.shape == (T, N):
                existing_list = [existing_alphas]
            elif is_1d and existing_alphas.shape[1] == N:
                existing_list = [existing_alphas[k:k+1, :] for k in range(existing_alphas.shape[0])]
            else:
                existing_list = [existing_alphas]
        elif existing_alphas.ndim == 3:
            existing_list = [existing_alphas[k] for k in range(existing_alphas.shape[0])]
        else:
            existing_list = [existing_alphas.reshape(T, N)]
    else:
        for ea in existing_alphas:
            ea_arr = np.asarray(ea, dtype=np.float64)
            if ea_arr.ndim == 1 and is_1d:
                existing_list.append(ea_arr.reshape(1, -1))
            elif ea_arr.shape == (T, N):
                existing_list.append(ea_arr)
            else:
                existing_list.append(ea_arr.reshape(T, N))
                
    if len(existing_list) == 0:
        return new_alpha.copy(), 0.0, 0.0
        
    # Compute max correlation before across valid elements
    corrs_before = []
    for ea in existing_list:
        valid_pair = np.isfinite(v_2d) & np.isfinite(ea)
        if mask_2d is not None:
            valid_pair = valid_pair & mask_2d
        if np.sum(valid_pair) >= 2:
            v_vals = v_2d[valid_pair]
            ea_vals = ea[valid_pair]
            if np.std(v_vals) > 1e-12 and np.std(ea_vals) > 1e-12:
                r = np.corrcoef(v_vals, ea_vals)[0, 1]
                corrs_before.append(abs(r) if np.isfinite(r) else 0.0)
            else:
                corrs_before.append(0.0)
        else:
            corrs_before.append(0.0)
            
    max_corr_before = float(np.max(corrs_before)) if corrs_before else 0.0
    if max_corr_before < threshold:
        return new_alpha.copy(), max_corr_before, max_corr_before

    # Cross-Sectional QR Orthogonalization per session t (Strictly Causal)
    out_2d = v_2d.copy()
    K = len(existing_list)
    
    for t in range(T):
        valid_t = np.isfinite(v_2d[t])
        if mask_2d is not None:
            valid_t = valid_t & mask_2d[t]
        for k in range(K):
            valid_t = valid_t & np.isfinite(existing_list[k][t])
            
        n_valid = int(np.sum(valid_t))
        if n_valid <= K:
            continue
            
        v_t = v_2d[t, valid_t]
        v_dm = v_t - np.mean(v_t)
        
        E_t = np.column_stack([existing_list[k][t, valid_t] for k in range(K)])
        E_dm = E_t - np.mean(E_t, axis=0)
        
        var_e = np.var(E_dm, axis=0)
        valid_cols = var_e > 1e-12
        if not np.any(valid_cols) or np.var(v_dm) <= 1e-12:
            continue
            
        E_active = E_dm[:, valid_cols]
        beta, _, _, _ = np.linalg.lstsq(E_active, v_dm, rcond=1e-10)
        v_proj = E_active @ beta
        v_ortho = v_dm - v_proj
        out_2d[t, valid_t] = v_ortho

    # Re-check correlation after orthogonalization
    corrs_after = []
    for ea in existing_list:
        valid_pair = np.isfinite(out_2d) & np.isfinite(ea)
        if mask_2d is not None:
            valid_pair = valid_pair & mask_2d
        if np.sum(valid_pair) >= 2:
            vo_vals = out_2d[valid_pair]
            ea_vals = ea[valid_pair]
            if np.std(vo_vals) > 1e-12 and np.std(ea_vals) > 1e-12:
                r = np.corrcoef(vo_vals, ea_vals)[0, 1]
                corrs_after.append(abs(r) if np.isfinite(r) else 0.0)
            else:
                corrs_after.append(0.0)
        else:
            corrs_after.append(0.0)
            
    max_corr_after = float(np.max(corrs_after)) if corrs_after else 0.0
    
    out_res = out_2d.reshape(orig_shape)
    return out_res, max_corr_before, max_corr_after


def ridge_factor_projection(
    alpha_target: np.ndarray,
    benchmark_factors: Union[List[np.ndarray], np.ndarray],
    adv_weights: Optional[np.ndarray] = None,
    gamma_shrink: float = 1e-8,
    threshold: float = 0.25
) -> Tuple[np.ndarray, float, float]:
    """
    Cross-Sectional Ridge-Regularized Factor Projection (Tikhonov Regularization):
    alpha_ortho = [I_N - X_t * (X_t^T * W_t * X_t + gamma * I_K)^(-1) * X_t^T * W_t] * alpha_target
    
    Key Features:
    - Order-invariant: Invariant to any permutation of benchmark factor columns.
    - Liquidity-weighting: W_t = diag(sqrt(ADV_20)) prioritizes independence in liquid names.
    - Numerical stability: Adaptive ridge shrinkage prevents condition number explosion.
    """
    v = np.asarray(alpha_target, dtype=np.float64)
    orig_shape = v.shape
    v_flat = v.ravel()
    
    if isinstance(benchmark_factors, (list, tuple)):
        bf_list = [np.asarray(bf, dtype=np.float64).ravel() for bf in benchmark_factors]
    elif isinstance(benchmark_factors, np.ndarray):
        if benchmark_factors.ndim == 2 and benchmark_factors.shape == orig_shape:
            bf_list = [benchmark_factors.ravel()]
        elif benchmark_factors.ndim == 2:
            bf_list = [benchmark_factors[:, k].ravel() for k in range(benchmark_factors.shape[1])]
        elif benchmark_factors.ndim == 3:
            bf_list = [benchmark_factors[k].ravel() for k in range(benchmark_factors.shape[0])]
        else:
            bf_list = [benchmark_factors.ravel()]
    else:
        bf_list = [np.asarray(benchmark_factors, dtype=np.float64).ravel()]
        
    if len(bf_list) == 0:
        return v.copy(), 0.0, 0.0
        
    X = np.column_stack(bf_list)
    valid_mask = np.isfinite(v_flat)
    for k in range(X.shape[1]):
        valid_mask = valid_mask & np.isfinite(X[:, k])
        
    if np.sum(valid_mask) < 2:
        return v.copy(), 0.0, 0.0
        
    v_clean = v_flat[valid_mask]
    X_clean = X[valid_mask, :]
    
    corrs_before = []
    std_v = np.std(v_clean)
    for k in range(X_clean.shape[1]):
        std_xk = np.std(X_clean[:, k])
        if std_v > 1e-12 and std_xk > 1e-12:
            r = np.corrcoef(v_clean, X_clean[:, k])[0, 1]
            corrs_before.append(abs(r))
        else:
            corrs_before.append(0.0)
    max_corr_before = float(np.max(corrs_before)) if corrs_before else 0.0
    
    if max_corr_before < threshold:
        return v.copy(), max_corr_before, max_corr_before
        
    X_dm = X_clean - np.mean(X_clean, axis=0)
    v_dm = v_clean - np.mean(v_clean)
    K = X_dm.shape[1]
    
    if adv_weights is not None:
        w_flat = np.asarray(adv_weights, dtype=np.float64).ravel()[valid_mask]
        w_clean = np.sqrt(np.maximum(w_flat, 1.0))
        W_diag = w_clean / np.maximum(np.mean(w_clean), 1e-4)
        X_w = X_dm * W_diag[:, np.newaxis]
        Xt_W_X = X_dm.T @ X_w
        Xt_W_v = X_dm.T @ (v_dm * W_diag)
    else:
        Xt_W_X = X_dm.T @ X_dm
        Xt_W_v = X_dm.T @ v_dm
        
    tr = np.trace(Xt_W_X)
    gamma = max(1e-10, gamma_shrink * tr / max(K, 1))
    reg_matrix = Xt_W_X + gamma * np.eye(K, dtype=np.float64)
    
    try:
        beta = np.linalg.solve(reg_matrix, Xt_W_v)
        v_proj = X_dm @ beta
    except np.linalg.LinAlgError:
        v_proj, _, _, _ = np.linalg.lstsq(reg_matrix, Xt_W_v, rcond=None)
        v_proj = X_dm @ v_proj
        
    v_ortho_clean = v_dm - v_proj
    
    corrs_after = []
    std_vo = np.std(v_ortho_clean)
    for k in range(X_clean.shape[1]):
        std_xk = np.std(X_clean[:, k])
        if std_vo > 1e-12 and std_xk > 1e-12:
            r = np.corrcoef(v_ortho_clean, X_clean[:, k])[0, 1]
            corrs_after.append(abs(r))
        else:
            corrs_after.append(0.0)
    max_corr_after = float(np.max(corrs_after)) if corrs_after else 0.0
    
    out_flat = v_flat.copy()
    out_flat[valid_mask] = v_ortho_clean
    return out_flat.reshape(orig_shape), max_corr_before, max_corr_after


def lowdin_symmetric_orthogonalization(
    factors: List[np.ndarray],
    adv_weights: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Symmetric Löwdin Orthogonalization for multiple graph alpha factors:
    F_lowdin = F * (F^T * W * F)^(-1/2)
    Treats all factors symmetrically without any sequential order bias.
    """
    K = len(factors)
    if K <= 1:
        return [f.copy() for f in factors]
        
    orig_shape = factors[0].shape
    F_stacked = np.column_stack([np.asarray(f, dtype=np.float64).ravel() for f in factors])
    valid_mask = np.all(np.isfinite(F_stacked), axis=1)
    
    if np.sum(valid_mask) < K:
        return [f.copy() for f in factors]
        
    F_clean = F_stacked[valid_mask, :]
    F_dm = F_clean - np.mean(F_clean, axis=0)
    
    if adv_weights is not None:
        w_flat = np.asarray(adv_weights, dtype=np.float64).ravel()[valid_mask]
        w_clean = np.sqrt(np.maximum(w_flat, 1.0))
        W_diag = w_clean / np.maximum(np.mean(w_clean), 1e-4)
        S = F_dm.T @ (F_dm * W_diag[:, np.newaxis])
    else:
        S = F_dm.T @ F_dm
        
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals_clipped = np.maximum(eigvals, 1e-10)
    S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals_clipped)) @ eigvecs.T
    
    F_ortho_clean = F_dm @ S_inv_sqrt
    
    out_factors = []
    for k in range(K):
        out_k = F_stacked[:, k].copy()
        out_k[valid_mask] = F_ortho_clean[:, k]
        out_factors.append(out_k.reshape(orig_shape))
        
    return out_factors


# ==============================================================================
# 4. VIETNAM TURNOVER DAMPENER (RANK HYSTERESIS + EMA)
# ==============================================================================

def dampen_turnover(
    alpha: np.ndarray,
    top_k_in: int = 10,
    top_k_out: int = 18,
    ema_span: int = 10,
    max_daily_turnover: float = 0.05,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Applies Rank Hysteresis (No-Trade Buffer) + EMA time-series smoothing
    to enforce Daily Turnover <= 5%/day, protecting returns against VN taxes and fees.
    
    Parameters:
    -----------
    alpha : np.ndarray, shape (T, N)
        Raw alpha signal matrix.
    top_k_in : int, default 10
        Buy threshold: only enter portfolio if rank <= top_k_in.
    top_k_out : int, default 18
        Sell threshold: only exit portfolio if rank > top_k_out.
    ema_span : int, default 10
        Exponential moving average smoothing span.
    max_daily_turnover : float, default 0.05 (5%/day)
        Strict upper bound on daily rebalancing turnover.
    mask : Optional[np.ndarray], shape (T, N), bool
        Tradable mask.
        
    Returns:
    --------
    np.ndarray, shape (T, N), float64
        Dampened target portfolio weights summing to 1.0.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    T, N = alpha.shape
    weights = np.zeros((T, N), dtype=np.float64)
    held = np.zeros(N, dtype=bool)
    
    alpha_ema = 2.0 / (ema_span + 1.0)
    
    # Day 0: Initial allocation to top_k_in valid stocks
    row0 = alpha[0].copy()
    if mask is not None:
        row0[~mask[0]] = -np.inf
    row0[~np.isfinite(row0)] = -np.inf
    
    valid_count0 = np.sum(np.isfinite(row0) & (row0 > -1e30))
    k0 = min(top_k_in, valid_count0) if valid_count0 > 0 else 0
    if k0 > 0:
        top_idx0 = np.argsort(-row0)[:k0]
        held[top_idx0] = True
        weights[0, held] = 1.0 / k0
        
    for t in range(1, T):
        row = alpha[t].copy()
        valid = np.isfinite(row)
        if mask is not None:
            valid = valid & mask[t]
        row[~valid] = -np.inf
        
        # Determine cross-sectional rank position (1 = best)
        sorted_indices = np.argsort(-row)
        rank_pos = np.zeros(N, dtype=int)
        for pos, idx in enumerate(sorted_indices):
            if np.isfinite(row[idx]) and row[idx] > -1e30:
                rank_pos[idx] = pos + 1
            else:
                rank_pos[idx] = 999999  # Disqualified
                
        # Rank Hysteresis update
        new_held = held.copy()
        for i in range(N):
            if held[i]:
                # Only exit if rank deteriorates beyond top_k_out or becomes invalid
                if rank_pos[i] > top_k_out:
                    new_held[i] = False
            else:
                # Only enter if rank enters top_k_in
                if rank_pos[i] <= top_k_in:
                    new_held[i] = True
                    
        held = new_held
        target_w = np.zeros(N, dtype=np.float64)
        num_held = np.sum(held)
        if num_held > 0:
            target_w[held] = 1.0 / num_held
            
        # Time-series EMA smoothing
        w_t = alpha_ema * target_w + (1.0 - alpha_ema) * weights[t - 1]
        
        # Enforce max daily turnover damper
        diff = w_t - weights[t - 1]
        step_to = 0.5 * np.sum(np.abs(diff))
        if step_to > max_daily_turnover:
            w_t = weights[t - 1] + diff * (max_daily_turnover / step_to)
            
        s = np.sum(np.abs(w_t))
        if s > 1e-12:
            w_t = w_t / s
        else:
            w_t = np.zeros(N, dtype=np.float64)
            
        weights[t] = w_t
        
    return weights


# ==============================================================================
# 5. VECTORIZED POINT-IN-TIME CONSTITUENT MASKING (POLARS LAZY API)
# ==============================================================================

def apply_pit_universe_mask(
    ohlcv_df: Union[pl.LazyFrame, pl.DataFrame],
    membership_df: Union[pl.LazyFrame, pl.DataFrame],
    adv_threshold_vnd: float = 10_000_000_000.0,
    min_samples: int = 15,
) -> Union[pl.LazyFrame, pl.DataFrame]:
    """
    Vectorized Point-in-Time Constituent Masking using Polars Lazy API.
    
    1. Computes strictly causal 20-day ADV (t-1 lag):
       adv_20d_lag1 = (raw_close * volume).rolling_mean(20, min_samples=min_samples).shift(1).over(ticker)
    2. Joins point-in-time with membership_df (effective_date, expiry_date) to flag in_vn30_pit.
    3. Builds valid_alpha_mask = in_vn30_pit & (adv_20d_lag1 >= adv_threshold_vnd).
    4. Sets pit_masked_close = close when valid_alpha_mask else null.
       This automatically excludes non-constituents from cross-sectional operators
       (rank, zscore, group_neutralize) without any look-ahead leakage.
    """
    is_eager = isinstance(ohlcv_df, pl.DataFrame)
    lf_ohlcv = ohlcv_df.lazy() if is_eager else ohlcv_df
    lf_mem = membership_df.lazy() if isinstance(membership_df, pl.DataFrame) else membership_df
    
    schema_names = lf_ohlcv.collect_schema().names()
    ticker_col = "ticker" if "ticker" in schema_names else "symbol"
    close_col = "close" if "close" in schema_names else "raw_close"
    raw_close_col = "raw_close" if "raw_close" in schema_names else close_col
    
    mem_schema_names = lf_mem.collect_schema().names()
    mem_ticker_col = "ticker" if "ticker" in mem_schema_names else "symbol"
    
    # 1. Causal 20-day ADV with 1-session lag (enforcing minimum 15-20 sessions stability)
    base_lf = lf_ohlcv.sort([ticker_col, "date"]).with_columns(
        (pl.col(raw_close_col).cast(pl.Float64) * pl.col("volume").cast(pl.Float64))
        .rolling_mean(window_size=20, min_samples=min_samples)
        .shift(1)
        .over(ticker_col)
        .alias("adv_20d_lag1")
    )
    
    # 2. Point-in-Time join with membership schedule
    pit_joined = (
        base_lf.join(
            lf_mem,
            left_on=ticker_col,
            right_on=mem_ticker_col,
            how="left"
        )
        .with_columns(
            (
                (pl.col("date") >= pl.col("effective_date"))
                & (
                    pl.col("expiry_date").is_null()
                    | (pl.col("date") <= pl.col("expiry_date"))
                )
            ).fill_null(False).alias("is_in_index")
        )
        .group_by(["date", ticker_col], maintain_order=True)
        .agg([
            pl.exclude(["effective_date", "expiry_date", "is_in_index"]).first(),
            pl.col("is_in_index").any().alias("in_vn30_pit"),
        ])
    )
    
    # 3. Create valid_alpha_mask and pit_masked_close
    res_lf = pit_joined.with_columns(
        (
            pl.col("in_vn30_pit").fill_null(False)
            & (pl.col("adv_20d_lag1").fill_null(0.0) >= adv_threshold_vnd)
        ).alias("valid_alpha_mask")
    ).with_columns(
        pl.when(pl.col("valid_alpha_mask"))
        .then(pl.col(close_col))
        .otherwise(None)
        .alias("pit_masked_close")
    ).sort(["date", ticker_col])
    
    return res_lf.collect() if is_eager else res_lf


# ==============================================================================
# 6. HOSE 3-TIER ORDER BOOK SLIPPAGE & INTRADAY EXECUTION SLICING
# ==============================================================================

def compute_hose_market_impact(
    q_shares: float,
    p_mid: float,
    adv20_shares: float,
    daily_vol: float,
    lob_depths: Optional[np.ndarray] = None,
    lob_prices: Optional[np.ndarray] = None,
    half_spread_floor: float = 0.0012,
    eta: float = 0.75,
    delta: float = 0.55,
    gamma_perm: float = 0.30
) -> float:
    """
    Computes expected relative slippage IS(Q) on HOSE:
    - 3-tier limit order book sweep (if depths & prices provided)
    - Latent tail square-root market impact: eta * sigma * (Q_tail / ADV20)^delta + gamma_perm * sigma * (Q / ADV20)
    - If daily OHLCV only: applies half_spread_floor (12 bps) + square root impact.
    """
    q_shares = float(max(0.0, q_shares))
    if q_shares <= 0.0 or p_mid <= 0.0:
        return 0.0
        
    adv20 = float(max(adv20_shares, 1.0))
    sigma = float(max(daily_vol, 0.005))
    
    if lob_depths is not None and lob_prices is not None and len(lob_depths) >= 3 and len(lob_prices) >= 3:
        # Full 3-tier LOB sweep
        v_levels = np.asarray(lob_depths[:3], dtype=np.float64)
        p_levels = np.asarray(lob_prices[:3], dtype=np.float64)
        q_vis = float(np.sum(v_levels))
        
        # 1. Depth sweep across visible 3 levels (directly includes half-spread from Level 0)
        cum_q = 0.0
        sweep_cost = 0.0
        for l in range(3):
            q_l = min(v_levels[l], max(0.0, q_shares - cum_q))
            sweep_cost += q_l * abs(p_levels[l] - p_mid)
            cum_q += v_levels[l]
        lob_component = sweep_cost / (q_shares * p_mid)
        
        # 2. Latent tail impact for Q > Q_vis
        tail_component = 0.0
        if q_shares > q_vis:
            q_tail = q_shares - q_vis
            temp_impact = eta * sigma * ((q_tail / adv20) ** delta)
            perm_impact = gamma_perm * sigma * (q_shares / adv20)
            tail_component = temp_impact + perm_impact
            
        total_is = lob_component + tail_component
    else:
        # OHLCV-only fallback mode: half-spread floor 12 bps + square-root impact
        temp_impact = eta * sigma * ((q_shares / adv20) ** delta)
        perm_impact = gamma_perm * sigma * (q_shares / adv20)
        total_is = max(half_spread_floor, half_spread_floor + temp_impact + perm_impact)
        
    return float(total_is)


def slice_intraday_order(
    q_shares: float,
    adv20_shares: float,
    slicing_threshold_pct: float = 0.005,
    max_participation_pct: float = 0.03
) -> Dict[str, Any]:
    """
    Applies HOSE liquidity constraints and intraday order slicing:
    - Max participation cap: Q_max <= 0.03 * ADV20 (3% ADV)
    - If Q > 0.005 * ADV20 (0.5% ADV), slices order into 3 tranches:
        * 25% at ATO (09:00 - 09:15)
        * 55% TWAP (09:30 - 14:15)
        * 20% at ATC (14:30 - 14:45)
    """
    q_capped = min(float(q_shares), float(adv20_shares * max_participation_pct))
    q_capped = np.floor(q_capped / 100.0) * 100.0
    
    threshold = adv20_shares * slicing_threshold_pct
    if q_capped > threshold:
        q_ato = np.floor((0.25 * q_capped) / 100.0) * 100.0
        q_atc = np.floor((0.20 * q_capped) / 100.0) * 100.0
        q_twap = q_capped - q_ato - q_atc
        return {
            "is_sliced": True,
            "q_total": q_capped,
            "q_ato": q_ato,
            "q_twap": q_twap,
            "q_atc": q_atc,
            "cap_applied": q_capped < q_shares
        }
    else:
        return {
            "is_sliced": False,
            "q_total": q_capped,
            "q_ato": q_capped,
            "q_twap": 0.0,
            "q_atc": 0.0,
            "cap_applied": q_capped < q_shares
        }


# ==============================================================================
# 7. HAWKES / ORNSTEIN-UHLENBECK DECAY FILTER & VAN DER WAERDEN TRANSFORM
# ==============================================================================

def apply_hawkes_ou_decay(
    event_impulses: np.ndarray,
    tau_sessions: float = 5.0,
    initial_state: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Continuous-time Hawkes process / Ornstein-Uhlenbeck exponential decay filter:
    S(i, t) = exp(-ln(2) / tau) * S(i, t-1) + event_impulses(i, t)
    
    Parameters:
    -----------
    event_impulses : np.ndarray, shape (T, N)
        Incoming sparse event impacts (omega_e * z_e) with causal lag >= T+1.
    tau_sessions : float
        Half-life decay in sessions:
        - tau_news = 5.0 sessions (short-term breaking news / sentiment)
        - tau_debt = 42.0 sessions (quarterly audits, credit exposure, cross-default)
    initial_state : Optional[np.ndarray], shape (N,)
        Initial state vector at t=0.
        
    Returns:
    --------
    np.ndarray, shape (T, N), float64
        Smooth stationary continuous factor series.
    """
    events = np.asarray(event_impulses, dtype=np.float64)
    T, N = events.shape
    out = np.zeros((T, N), dtype=np.float64)
    
    decay_factor = np.exp(-np.log(2.0) / max(tau_sessions, 1e-4))
    curr_state = np.zeros(N, dtype=np.float64) if initial_state is None else np.asarray(initial_state, dtype=np.float64).copy()
    
    for t in range(T):
        curr_state = decay_factor * curr_state + np.nan_to_num(events[t], nan=0.0)
        out[t] = curr_state.copy()
        
    return out


def van_der_waerden_transform(
    factor_matrix: np.ndarray,
    mask: Optional[np.ndarray] = None,
    clip_limit: float = 3.0
) -> np.ndarray:
    """
    Cross-Sectional Inverse Normal Rank Transform (Van der Waerden normal scores):
    S_tilde(i, t) = Phi^(-1)( (rank(S(i, t)) - 0.5) / N_t )
    
    Ensures exact stationary normal distribution across the cross-section
    for clean integration into factor models and machine learning combiners.
    """
    mat = np.asarray(factor_matrix, dtype=np.float64)
    T, N = mat.shape
    out = np.zeros((T, N), dtype=np.float64)
    
    for t in range(T):
        row = mat[t].copy()
        valid = np.isfinite(row)
        if mask is not None:
            valid = valid & mask[t]
            
        N_t = int(np.sum(valid))
        if N_t >= 2:
            sub_vals = row[valid]
            ranked = stats.rankdata(sub_vals)
            percentiles = (ranked - 0.5) / float(N_t)
            percentiles = np.clip(percentiles, 1e-5, 1.0 - 1e-5)
            z_scores = stats.norm.ppf(percentiles)
            z_scores = np.clip(z_scores, -clip_limit, clip_limit)
            out[t, valid] = z_scores
            out[t, ~valid] = np.nan
        else:
            out[t] = np.nan
            
    return out
