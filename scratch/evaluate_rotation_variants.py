#!/usr/bin/env python3
"""
Comprehensive investigation into alternative rotation-based formulations
for Ensemble Conformal Prediction on CIFAR-10.

Evaluates:
1. Baseline: Standard Conformal Prediction (Orig-thr)
2. Current Method: Symmetric Rotation Invariance d(x) in [-179, +179]
3. Option 1: Agreement Count / Vote Count N_agree(x)
4. Option 2: Asymmetric Continuous Interval:
   - 2a: Total Span W(x) = d_left(x) + d_right(x)
   - 2b: Asymmetry Imbalance |d_right(x) - d_left(x)|
5. Option 3: Restricted Maximal Rotation Range:
   - 3a: D_max = 90 degrees
   - 3b: D_max = 45 degrees
   - 3c: D_max = 30 degrees
6. Option 4 (etc.):
   - 4a: Test-Time Augmentation (TTA) Conformal Predictor (smoothed probabilities over rotations)
   - 4b: Rotation Predictive Entropy Routing H_rot(x)
   - 4c: Rotation Confidence Margin Routing M_rot(x)
   - 4d: Per-Bin Independent Conformal Calibration (lambda_b per bin)
"""

import os
import sys
import numpy as np
from math import ceil

# Add workspace to path
sys.path.insert(0, "/home/lkd18/git_projects/phd/cp_version_ruediger/self_trial")

from precalibrator.bin_optimizer import PrecalibrationBinOptimizer, compute_rotation_invariance


def load_dataset(dataset_name="rot179"):
    log_dir = "logs/cifar10/2026"
    precal_file = f"{log_dir}/prob_cache_CIFAR10_precalibration_seed2026_{dataset_name}.npz"
    calib_file = f"{log_dir}/prob_cache_CIFAR10_calibration_seed2026_{dataset_name}.npz"
    test_file = f"{log_dir}/prob_cache_CIFAR10_testing_seed2026_{dataset_name}.npz"

    d_pre = np.load(precal_file)
    d_cal = np.load(calib_file)
    d_test = np.load(test_file)

    return {
        "P_pre": d_pre["probabilities"], # (n_views, N, K)
        "y_pre": d_pre["targets"],        # (N,)
        "P_cal": d_cal["probabilities"],
        "y_cal": d_cal["targets"],
        "P_test": d_test["probabilities"],
        "y_test": d_test["targets"],
    }


def compute_metrics(P_views, max_rotation=179):
    """
    Computes all candidate rotation metrics for a dataset tensor P_views (n_views, N, K).
    """
    n_views, N, K = P_views.shape
    p0 = P_views[0] # (N, K)
    top0 = np.argmax(p0, axis=-1) # (N,)

    # Views:
    # 0: unrotated
    # 1..max_rotation: negative angles [-max_rotation..-1]
    # max_rotation+1..2*max_rotation: positive angles [+1..+max_rotation]
    neg_views = P_views[1 : max_rotation + 1] # shape (max_rotation, N, K)
    pos_views = P_views[max_rotation + 1 : 2 * max_rotation + 1] # shape (max_rotation, N, K)

    # 1. Symmetric invariance d(x)
    sym_mask = np.ones(N, dtype=bool)
    d_sym = np.zeros(N, dtype=np.int64)
    for d in range(1, max_rotation + 1):
        idx_neg = max_rotation - d # d=1 -> max_rotation - 1 (-1 deg)
        idx_pos = d - 1            # d=1 -> 0 (+1 deg)
        m_neg = (np.argmax(neg_views[idx_neg], axis=-1) == top0)
        m_pos = (np.argmax(pos_views[idx_pos], axis=-1) == top0)
        sym_mask = sym_mask & m_neg & m_pos
        d_sym = np.where(sym_mask, d, d_sym)

    # 2. Option 1: Agreement Count across all 2*max_rotation angles
    # How many angles match top0?
    agree_neg = (np.argmax(neg_views, axis=-1) == top0[None, :]).sum(axis=0) # (N,)
    agree_pos = (np.argmax(pos_views, axis=-1) == top0[None, :]).sum(axis=0) # (N,)
    n_agree = agree_neg + agree_pos # in [0, 2*max_rotation]

    # 3. Option 2: Asymmetric Continuous Interval [d_left, d_right]
    # d_left: continuous span in negative direction
    left_mask = np.ones(N, dtype=bool)
    d_left = np.zeros(N, dtype=np.int64)
    for d in range(1, max_rotation + 1):
        idx_neg = max_rotation - d
        m_neg = (np.argmax(neg_views[idx_neg], axis=-1) == top0)
        left_mask = left_mask & m_neg
        d_left = np.where(left_mask, d, d_left)

    # d_right: continuous span in positive direction
    right_mask = np.ones(N, dtype=bool)
    d_right = np.zeros(N, dtype=np.int64)
    for d in range(1, max_rotation + 1):
        idx_pos = d - 1
        m_pos = (np.argmax(pos_views[idx_pos], axis=-1) == top0)
        right_mask = right_mask & m_pos
        d_right = np.where(right_mask, d, d_right)

    w_total = d_left + d_right # Total continuous width in [0, 2*max_rotation]
    asym_diff = np.abs(d_right - d_left)

    # 4. Option 3: Restricted maximal angles (e.g. 90, 45, 30)
    d_sym_90 = np.minimum(d_sym, 90)
    d_sym_45 = np.minimum(d_sym, 45)
    d_sym_30 = np.minimum(d_sym, 30)

    # 5. Option 4: Predictive Entropy and TTA Mean
    # TTA mean probability: average over all views (original + all rotations)
    p_tta = P_views.mean(axis=0) # (N, K)
    # Entropy of TTA distribution: H = -sum p log p
    eps = 1e-12
    tta_entropy = -np.sum(p_tta * np.log(np.clip(p_tta, eps, 1.0)), axis=-1) # (N,)
    # Discretize entropy into integer scale [0, 100] for bin optimizer
    h_int = np.clip(np.round((tta_entropy / np.log(K)) * 100), 0, 100).astype(np.int64)

    # Mean Margin under rotations: average (top1_prob - top2_prob) across views
    sorted_p = np.sort(P_views, axis=-1) # (n_views, N, K)
    margins = sorted_p[..., -1] - sorted_p[..., -2] # (n_views, N)
    mean_margin = margins.mean(axis=0) # (N,)
    margin_int = np.clip(np.round(mean_margin * 100), 0, 100).astype(np.int64)

    return {
        "d_sym": d_sym,
        "n_agree": n_agree,
        "d_left": d_left,
        "d_right": d_right,
        "w_total": w_total,
        "asym_diff": asym_diff,
        "d_sym_90": d_sym_90,
        "d_sym_45": d_sym_45,
        "d_sym_30": d_sym_30,
        "p_tta": p_tta,
        "h_int": h_int,
        "margin_int": margin_int,
        "p0": p0,
    }


def run_pipeline(
    metric_name,
    metric_pre,
    metric_cal,
    metric_test,
    p_pre,
    y_pre,
    p_cal,
    y_cal,
    p_test,
    y_test,
    alpha=0.1,
    n_bins=3,
    scoring_function="thr",
    min_bin_coverage=None,
    independent_calib=False,
):
    """
    Runs the full precalibration -> calibration -> test pipeline for a given metric.
    """
    N_pre = len(y_pre)
    N_cal = len(y_cal)
    N_test = len(y_test)
    K = p_pre.shape[1]

    # Precalibration MILP
    optimizer = PrecalibrationBinOptimizer(
        n_bins=n_bins,
        alpha=alpha,
        min_samples_per_bin=20,
        scoring_function=scoring_function,
        min_bin_coverage=min_bin_coverage,
        min_degree_span=2,
        solver="milp",
    )

    opt_res = optimizer.optimize(
        degrees=metric_pre,
        p_ref=p_pre,
        y_true=y_pre,
        max_degree=int(np.max([metric_pre.max(), metric_cal.max(), metric_test.max()])),
        scoring_function=scoring_function,
    )

    bins = opt_res["bins"]
    bin_taus_pre = {b["bin_index"]: b["threshold"] for b in bins}

    def get_bin(val):
        for b in bins:
            if b["min_degree"] <= val <= b["max_degree"]:
                return b["bin_index"]
        if val < bins[0]["min_degree"]:
            return bins[0]["bin_index"]
        return bins[-1]["bin_index"]

    cal_bins = np.array([get_bin(v) for v in metric_cal])
    test_bins = np.array([get_bin(v) for v in metric_test])

    # Compute nonconformity scores
    if scoring_function == "thr":
        s_cal_all = 1.0 - p_cal
        s_test_all = 1.0 - p_test
    elif scoring_function == "aps":
        def get_aps_all(P):
            order = np.argsort(-P, axis=1, kind="stable")
            Ps = np.take_along_axis(P, order, axis=1)
            csum = np.cumsum(Ps, axis=1)
            prev = np.zeros_like(csum)
            prev[:, 1:] = csum[:, :-1]
            all_s = np.empty_like(Ps)
            np.put_along_axis(all_s, order, prev, axis=1)
            return all_s
        s_cal_all = get_aps_all(p_cal)
        s_test_all = get_aps_all(p_test)

    s_cal_true = s_cal_all[np.arange(N_cal), y_cal]
    s_test_true = s_test_all[np.arange(N_test), y_test]

    if not independent_calib:
        # Standard RI-CP calibration via single global scaling factor lambda
        base_sample_taus = np.array([bin_taus_pre[b] for b in cal_bins], dtype=float)
        ratios = np.zeros(N_cal, dtype=float)
        for j in range(N_cal):
            ratios[j] = s_cal_true[j] / max(base_sample_taus[j], 1e-4)

        k_cal = int(ceil((N_cal + 1) * (1.0 - alpha)))
        if k_cal > N_cal:
            scaling_factor = np.inf
        else:
            scaling_factor = float(np.partition(ratios, k_cal - 1)[k_cal - 1])

        cal_taus = {}
        for b in bins:
            b_idx = b["bin_index"]
            cal_taus[b_idx] = 1.0 if np.isinf(scaling_factor) else min(1.0, float(scaling_factor * bin_taus_pre[b_idx]))
    else:
        # Option 4d: Independent per-bin calibration
        scaling_factor = None
        cal_taus = {}
        for b in bins:
            b_idx = b["bin_index"]
            mask_b = (cal_bins == b_idx)
            scores_b = s_cal_true[mask_b]
            n_b = len(scores_b)
            if n_b == 0:
                cal_taus[b_idx] = 1.0
            else:
                k_b = int(ceil((n_b + 1) * (1.0 - alpha)))
                if k_b > n_b:
                    cal_taus[b_idx] = 1.0
                else:
                    cal_taus[b_idx] = float(np.partition(scores_b, k_b - 1)[k_b - 1])

    # Evaluate on Test Set
    test_taus = np.array([cal_taus[b] for b in test_bins], dtype=float)
    covered = s_test_true <= test_taus
    overall_cov = float(covered.mean())

    # Prediction set sizes
    inclusions = s_test_all <= test_taus[:, None]
    # Check empty sets and at_least_one
    empty_mask = inclusions.sum(axis=1) == 0
    n_empty = int(empty_mask.sum())
    # Optional fallback
    inclusions_fallback = inclusions.copy()
    if n_empty > 0:
        top1 = np.argmax(p_test[empty_mask], axis=1)
        inclusions_fallback[empty_mask, top1] = True

    set_sizes = inclusions.sum(axis=1)
    set_sizes_fallback = inclusions_fallback.sum(axis=1)
    mean_sz = float(set_sizes.mean())
    mean_sz_fb = float(set_sizes_fallback.mean())

    # Per-bin statistics
    bin_stats = []
    for b in bins:
        b_idx = b["bin_index"]
        m_b = (test_bins == b_idx)
        n_b = int(m_b.sum())
        cov_b = float(covered[m_b].mean()) if n_b > 0 else 0.0
        sz_b = float(set_sizes[m_b].mean()) if n_b > 0 else 0.0
        bin_stats.append({
            "bin": b_idx,
            "range": (b["min_degree"], b["max_degree"]),
            "n_test": n_b,
            "cov": cov_b,
            "size": sz_b,
            "tau_pre": bin_taus_pre[b_idx],
            "tau_cal": cal_taus[b_idx],
        })

    # CovGap across bins: max |cov_b - (1 - alpha)|
    cov_gap = max([abs(bs["cov"] - (1.0 - alpha)) for bs in bin_stats if bs["n_test"] > 0])

    return {
        "metric_name": metric_name,
        "overall_cov": overall_cov,
        "mean_size": mean_sz,
        "mean_size_fallback": mean_sz_fb,
        "n_empty": n_empty,
        "cov_gap": cov_gap,
        "scaling_factor": scaling_factor,
        "bins": bin_stats,
    }


def evaluate_all():
    print("=" * 80)
    print("STARTING EVALUATION OF ALL ROTATION VARIANTS ON CIFAR-10")
    print("=" * 80)

    data = load_dataset("rot179")
    p0_pre = data["P_pre"][0]
    p0_cal = data["P_cal"][0]
    p0_test = data["P_test"][0]

    y_pre = data["y_pre"]
    y_cal = data["y_cal"]
    y_test = data["y_test"]

    print(f"Data splits: Precalibration={len(y_pre)}, Calibration={len(y_cal)}, Test={len(y_test)}")

    metrics_pre = compute_metrics(data["P_pre"])
    metrics_cal = compute_metrics(data["P_cal"])
    metrics_test = compute_metrics(data["P_test"])

    # 1. Standard Conformal Prediction Baseline (Single Global Threshold)
    for score_fn in ["thr", "aps"]:
        print(f"\n--- BASELINE: Standard CP (Orig-{score_fn}) ---")
        if score_fn == "thr":
            s_cal_true = (1.0 - p0_cal)[np.arange(len(y_cal)), y_cal]
            s_test_true = (1.0 - p0_test)[np.arange(len(y_test)), y_test]
            s_test_all = 1.0 - p0_test
        else:
            order = np.argsort(-p0_cal, axis=1, kind="stable")
            Ps = np.take_along_axis(p0_cal, order, axis=1)
            csum = np.cumsum(Ps, axis=1)
            prev = np.zeros_like(csum)
            prev[:, 1:] = csum[:, :-1]
            all_s = np.empty_like(Ps)
            np.put_along_axis(all_s, order, prev, axis=1)
            s_cal_true = all_s[np.arange(len(y_cal)), y_cal]

            order_t = np.argsort(-p0_test, axis=1, kind="stable")
            Ps_t = np.take_along_axis(p0_test, order_t, axis=1)
            csum_t = np.cumsum(Ps_t, axis=1)
            prev_t = np.zeros_like(csum_t)
            prev_t[:, 1:] = csum_t[:, :-1]
            s_test_all = np.empty_like(Ps_t)
            np.put_along_axis(s_test_all, order_t, prev_t, axis=1)
            s_test_true = s_test_all[np.arange(len(y_test)), y_test]

        for a in [0.1, 0.01]:
            k = int(ceil((len(y_cal) + 1) * (1.0 - a)))
            qhat = float(np.partition(s_cal_true, k - 1)[k - 1])
            cov = float((s_test_true <= qhat).mean())
            sets = s_test_all <= qhat
            empty = (sets.sum(axis=1) == 0).sum()
            sz = float(sets.sum(axis=1).mean())
            print(f"  Orig-{score_fn} (alpha={a}): qhat={qhat:.6f}, Test Cov={cov:.4f}, Mean Size={sz:.4f}, Empty={empty}")

    # Now evaluate all variants at alpha=0.1, score=thr
    variants = [
        ("Current Symmetric d(x) in [0..179]", "d_sym", None),
        ("Option 1: Agreement Count N_agree(x)", "n_agree", None),
        ("Option 2a: Total Asymmetric Span W(x) = d_L + d_R", "w_total", None),
        ("Option 2b: Asymmetry Imbalance |d_R - d_L|", "asym_diff", None),
        ("Option 3a: Restricted Rotation D_max = 90 deg", "d_sym_90", None),
        ("Option 3b: Restricted Rotation D_max = 45 deg", "d_sym_45", None),
        ("Option 3c: Restricted Rotation D_max = 30 deg", "d_sym_30", None),
        ("Option 4b: Rotation Predictive Entropy H_rot(x)", "h_int", None),
        ("Option 4c: Rotation Confidence Margin M_rot(x)", "margin_int", None),
        ("Option 4d: Per-Bin Independent Calibration (lambda_b)", "d_sym", True),
    ]

    print("\n" + "=" * 100)
    print("COMPARATIVE EVALUATION OF ROTATION VARIANTS (alpha=0.1, B=3, scoring=thr, min_bin_cov=0.78)")
    print("=" * 100)
    results_alpha01 = []
    for label, m_key, indep in variants:
        res = run_pipeline(
            metric_name=label,
            metric_pre=metrics_pre[m_key],
            metric_cal=metrics_cal[m_key],
            metric_test=metrics_test[m_key],
            p_pre=p0_pre,
            y_pre=y_pre,
            p_cal=p0_cal,
            y_cal=y_cal,
            p_test=p0_test,
            y_test=y_test,
            alpha=0.1,
            n_bins=3,
            scoring_function="thr",
            min_bin_coverage=0.78,
            independent_calib=bool(indep),
        )
        results_alpha01.append(res)
        print(f"\nVariant: {label}")
        print(f"  Test Cov: {res['overall_cov']:.4f} (target 0.9000), Mean Size: {res['mean_size_fallback']:.4f}, Empty: {res['n_empty']}, CovGap: {res['cov_gap']:.4f}")
        for b in res["bins"]:
            print(f"    Bin {b['bin']} [{b['range'][0]}..{b['range'][1]}]: N={b['n_test']}, Cov={b['cov']:.4f}, Size={b['size']:.4f}, tau_pre={b['tau_pre']:.4f}, tau_cal={b['tau_cal']:.4f}")

    # Now evaluate Option 4a: TTA Mean Conformal Predictor
    print("\n" + "=" * 100)
    print("EVALUATION OF OPTION 4a: TEST-TIME AUGMENTATION (TTA) CONFORMAL PREDICTOR")
    print("=" * 100)
    p_tta_cal = metrics_cal["p_tta"]
    p_tta_test = metrics_test["p_tta"]
    s_tta_cal = (1.0 - p_tta_cal)[np.arange(len(y_cal)), y_cal]
    s_tta_test = (1.0 - p_tta_test)[np.arange(len(y_test)), y_test]
    k_01 = int(ceil((len(y_cal) + 1) * 0.9))
    qhat_tta_01 = float(np.partition(s_tta_cal, k_01 - 1)[k_01 - 1])
    cov_tta_01 = float((s_tta_test <= qhat_tta_01).mean())
    sets_tta_01 = (1.0 - p_tta_test) <= qhat_tta_01
    sz_tta_01 = float(sets_tta_01.sum(axis=1).mean())
    empty_tta_01 = (sets_tta_01.sum(axis=1) == 0).sum()
    print(f"TTA-Mean-thr (alpha=0.1): qhat={qhat_tta_01:.6f}, Test Cov={cov_tta_01:.4f}, Mean Size={sz_tta_01:.4f}, Empty={empty_tta_01}")

    k_001 = int(ceil((len(y_cal) + 1) * 0.99))
    qhat_tta_001 = float(np.partition(s_tta_cal, k_001 - 1)[k_001 - 1])
    cov_tta_001 = float((s_tta_test <= qhat_tta_001).mean())
    sets_tta_001 = (1.0 - p_tta_test) <= qhat_tta_001
    sz_tta_001 = float(sets_tta_001.sum(axis=1).mean())
    empty_tta_001 = (sets_tta_001.sum(axis=1) == 0).sum()
    print(f"TTA-Mean-thr (alpha=0.01): qhat={qhat_tta_001:.6f}, Test Cov={cov_tta_001:.4f}, Mean Size={sz_tta_001:.4f}, Empty={empty_tta_001}")

    # Repeat for alpha=0.01
    print("\n" + "=" * 100)
    print("COMPARATIVE EVALUATION OF ROTATION VARIANTS (alpha=0.01, B=3, scoring=thr, min_bin_cov=0.95)")
    print("=" * 100)
    results_alpha001 = []
    for label, m_key, indep in variants:
        res = run_pipeline(
            metric_name=label,
            metric_pre=metrics_pre[m_key],
            metric_cal=metrics_cal[m_key],
            metric_test=metrics_test[m_key],
            p_pre=p0_pre,
            y_pre=y_pre,
            p_cal=p0_cal,
            y_cal=y_cal,
            p_test=p0_test,
            y_test=y_test,
            alpha=0.01,
            n_bins=3,
            scoring_function="thr",
            min_bin_coverage=0.95,
            independent_calib=bool(indep),
        )
        results_alpha001.append(res)
        print(f"\nVariant: {label}")
        print(f"  Test Cov: {res['overall_cov']:.4f} (target 0.9900), Mean Size: {res['mean_size_fallback']:.4f}, Empty: {res['n_empty']}, CovGap: {res['cov_gap']:.4f}")
        for b in res["bins"]:
            print(f"    Bin {b['bin']} [{b['range'][0]}..{b['range'][1]}]: N={b['n_test']}, Cov={b['cov']:.4f}, Size={b['size']:.4f}, tau_pre={b['tau_pre']:.4f}, tau_cal={b['tau_cal']:.4f}")

    # Repeat for APS scoring function (alpha=0.1)
    print("\n" + "=" * 100)
    print("COMPARATIVE EVALUATION OF ROTATION VARIANTS (alpha=0.1, B=3, scoring=aps, min_bin_cov=0.78)")
    print("=" * 100)
    for label, m_key, indep in variants:
        res = run_pipeline(
            metric_name=label,
            metric_pre=metrics_pre[m_key],
            metric_cal=metrics_cal[m_key],
            metric_test=metrics_test[m_key],
            p_pre=p0_pre,
            y_pre=y_pre,
            p_cal=p0_cal,
            y_cal=y_cal,
            p_test=p0_test,
            y_test=y_test,
            alpha=0.1,
            n_bins=3,
            scoring_function="aps",
            min_bin_coverage=0.78,
            independent_calib=bool(indep),
        )
        print(f"\nVariant: {label} (APS)")
        print(f"  Test Cov: {res['overall_cov']:.4f} (target 0.9000), Mean Size: {res['mean_size_fallback']:.4f}, Empty: {res['n_empty']}, CovGap: {res['cov_gap']:.4f}")
        for b in res["bins"]:
            print(f"    Bin {b['bin']} [{b['range'][0]}..{b['range'][1]}]: N={b['n_test']}, Cov={b['cov']:.4f}, Size={b['size']:.4f}, tau_pre={b['tau_pre']:.4f}, tau_cal={b['tau_cal']:.4f}")


if __name__ == "__main__":
    evaluate_all()
