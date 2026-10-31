"""
Unit tests for optional rank_ic_ties in purged_walk_forward_combiner (R6-4).
Demonstrates:
1. Under synthetic data with tied factor values:
   - rank_ic_ties="ordinal" (np.argsort argsort) assigns arbitrary ranks based on column position,
     causing daily ICs and composite weights to vary under column permutations.
   - rank_ic_ties="average" (scipy.stats.rankdata method="average") treats ties symmetrically,
     making daily ICs and unpermuted composite alpha strictly invariant under column permutations.
2. Default rank_ic_ties="ordinal" preserves exact legacy behavior.
"""
import numpy as np
import pytest
from engine.ml_alpha_combiner import purged_walk_forward_combiner


def test_rank_ic_ties_permutation_invariance():
    """
    Construct synthetic scenario with K=2 factors, T=20 sessions, N=10 assets.
    Factor 0 has identical tied values across multiple assets (e.g. 5 assets tied at 0.5).
    Show that:
    - ordinal rank IC produces different composite alpha when asset columns are permuted.
    - average rank IC produces bit-identical composite alpha (up to floating-point precision <= 1e-12).
    """
    np.random.seed(42)
    T, N, K = 20, 10, 2
    
    # Forward return from simulated price matrix
    raw_close = 50.0 + np.cumsum(np.random.randn(T, N) * 0.5, axis=0)
    tradable_mask = np.ones((T, N), dtype=bool)
    
    # Factor 0 has identical tied values for assets 2 and 3, but different returns
    alpha0 = np.random.randn(T, N)
    alpha0[:, 2] = 0.500
    alpha0[:, 3] = 0.500
    
    alpha1 = np.random.randn(T, N)
    ortho_alphas = [alpha0, alpha1]
    
    # Permutation that swaps tied assets 2 and 3
    perm = np.arange(N)
    perm[2], perm[3] = 3, 2
    perm_inv = np.argsort(perm)
    
    ortho_alphas_perm = [a[:, perm] for a in ortho_alphas]
    raw_close_perm = raw_close[:, perm]
    tradable_mask_perm = tradable_mask[:, perm]
    
    # 1. Test "ordinal" (legacy):
    ca_ord, _ = purged_walk_forward_combiner(ortho_alphas, raw_close, tradable_mask, window=5, embargo=1, rank_ic_ties="ordinal")
    ca_ord_p, _ = purged_walk_forward_combiner(ortho_alphas_perm, raw_close_perm, tradable_mask_perm, window=5, embargo=1, rank_ic_ties="ordinal")
    unperm_ord = ca_ord_p[:, perm_inv]
    diff_ord = np.max(np.abs(unperm_ord - ca_ord))
    
    # Ordinal ranking breaks ties by column order, so permutation causes a detectable discrepancy
    assert diff_ord > 1e-6, f"Expected ordinal ranking to show column order sensitivity, got diff {diff_ord}"
    
    # 2. Test "average" (sanitized):
    ca_avg, _ = purged_walk_forward_combiner(ortho_alphas, raw_close, tradable_mask, window=5, embargo=1, rank_ic_ties="average")
    ca_avg_p, _ = purged_walk_forward_combiner(ortho_alphas_perm, raw_close_perm, tradable_mask_perm, window=5, embargo=1, rank_ic_ties="average")
    unperm_avg = ca_avg_p[:, perm_inv]
    diff_avg = np.max(np.abs(unperm_avg - ca_avg))
    
    # Average ranking treats tied values identically, guaranteeing invariance under column permutations
    assert diff_avg <= 1e-12, f"Expected average ranking to be invariant under column permutations, got diff {diff_avg}"


def test_rank_ic_ties_default_is_ordinal():
    """
    Verify that calling purged_walk_forward_combiner without rank_ic_ties defaults to 'ordinal'.
    """
    np.random.seed(123)
    T, N = 15, 10
    raw_close = 50.0 + np.cumsum(np.random.randn(T, N) * 0.5, axis=0)
    tradable_mask = np.ones((T, N), dtype=bool)
    ortho_alphas = [np.random.randn(T, N), np.random.randn(T, N)]
    
    ca_default, w_default = purged_walk_forward_combiner(ortho_alphas, raw_close, tradable_mask, window=5, embargo=1)
    ca_explicit, w_explicit = purged_walk_forward_combiner(ortho_alphas, raw_close, tradable_mask, window=5, embargo=1, rank_ic_ties="ordinal")
    
    assert np.array_equal(ca_default, ca_explicit)
    assert np.array_equal(w_default, w_explicit)
