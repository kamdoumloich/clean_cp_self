#!/usr/bin/env python3
"""
Diagnostic: does a rotation-routing feature r(x) carry information about errors
BEYOND the unrotated softmax confidence?

If it does not, no binning of r(x) can beat a single global threshold (up to the
APS tail-shape effect). If it does, the size of the gain bounds what binning can win.

Inputs: the probability caches written by the pipeline
  logs/<dataset>/<seed>/prob_cache_<BENCH>_<split>_seed<seed>_rot<max_rotation>.npz
(probabilities: (1 + 2*max_rotation, N, K), targets: (N,)). Labelled splits are pooled,
so only exchangeable labelled data (precalibration + calibration) should be passed.

Targets (binary, 1 = bad event):
  top1   : top-1 error of the unrotated view
  thr    : miscoverage of the global split-CP set with THR score 1 - p(y|x) at level alpha
  aps    : same with the (non-randomised) APS score used in PrecalibrationBinOptimizer

Models (cross-validated, out-of-fold):
  conf        : spline(rank of logit max-softmax)
  route       : spline(rank of r)
  conf+route  : additive splines (no interaction)
  gbm-conf / gbm-conf+route : HistGradientBoosting (allows interaction)

Reported: AUROC / log-loss per model, and for (conf+route) - (conf): the paired-bootstrap CI
of the difference and a permutation p-value (r permuted within confidence deciles).
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from precalibrator.bin_optimizer import (  # noqa: E402
    compute_agreement_count,
    compute_asymmetric_span,
    compute_rotation_invariance,
)

EPS = 1e-6


def load_pooled(cache_dir, bench, seed, max_rotation, splits):
    P, Y = [], []
    for s in splits:
        f = os.path.join(cache_dir, f"prob_cache_{bench}_{s}_seed{seed}_rot{max_rotation}.npz")
        with np.load(f) as z:
            P.append(z["probabilities"])
            Y.append(z["targets"])
        print(f"  loaded {f}  views/N/K = {P[-1].shape}")
    return np.concatenate(P, axis=1), np.concatenate(Y)


def conformal_quantile(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.sort(scores)[min(k, n) - 1]


def aps_true_scores(p, y):
    order = np.argsort(-p, axis=1, kind="stable")
    ps = np.take_along_axis(p, order, axis=1)
    prev = np.cumsum(ps, axis=1) - ps
    out = np.empty_like(ps)
    np.put_along_axis(out, order, prev, axis=1)
    return out[np.arange(len(y)), y]


def rank01(v):
    return (rankdata(v, method="average") - 0.5) / len(v)


def make_models(seed):
    def spl():
        return SplineTransformer(n_knots=8, degree=3, knots="uniform", extrapolation="constant")

    def lr():
        return LogisticRegression(C=1.0, max_iter=5000)

    def gbm():
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0,
                                              min_samples_leaf=40, random_state=seed)

    return {
        "conf": (lambda: make_pipeline(spl(), lr()), ["conf"]),
        "route": (lambda: make_pipeline(spl(), lr()), ["route"]),
        "conf+route": (lambda: make_pipeline(spl(), lr()), ["conf", "route"]),
        "gbm-conf": (gbm, ["conf"]),
        "gbm-conf+route": (gbm, ["conf", "route"]),
    }


def oof_predict(factory, X, y, n_folds, n_repeats, seed):
    """Average out-of-fold probability over repeats; also per-repeat metrics."""
    n = len(y)
    acc = np.zeros(n)
    per_rep = []
    for r in range(n_repeats):
        pred = np.zeros(n)
        for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed + r).split(X, y):
            m = factory().fit(X[tr], y[tr])
            pred[te] = m.predict_proba(X[te])[:, 1]
        pred = np.clip(pred, EPS, 1 - EPS)
        per_rep.append((roc_auc_score(y, pred), log_loss(y, pred)))
        acc += pred / n_repeats
    return acc, np.array(per_rep)


def paired_bootstrap(y, p_a, p_b, n_boot, rng):
    """CI of metric(p_b) - metric(p_a): AUROC (higher better) and log-loss reduction (a - b)."""
    n = len(y)
    d_auc, d_ll = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        d_auc.append(roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i]))
        d_ll.append(log_loss(y[i], p_a[i], labels=[0, 1]) - log_loss(y[i], p_b[i], labels=[0, 1]))
    q = lambda v: np.percentile(v, [2.5, 97.5])
    return q(d_auc), q(d_ll)


def permutation_pvalue(y, conf, route, n_perm, n_folds, seed, strata):
    """H0: route adds nothing given conf. Permute route within confidence strata; statistic = log-loss reduction."""
    spl = lambda: SplineTransformer(n_knots=8, degree=3, knots="uniform", extrapolation="constant")
    fac = lambda: make_pipeline(spl(), LogisticRegression(C=1.0, max_iter=5000))
    folds = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(conf[:, None], y))

    def cv_ll(X):
        pred = np.zeros(len(y))
        for tr, te in folds:
            pred[te] = fac().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        return log_loss(y, np.clip(pred, EPS, 1 - EPS))

    base = cv_ll(conf[:, None])
    obs = base - cv_ll(np.column_stack([conf, route]))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        rp = route.copy()
        for s in np.unique(strata):
            idx = np.where(strata == s)[0]
            rp[idx] = route[rng.permutation(idx)]
        null[b] = base - cv_ll(np.column_stack([conf, rp]))
    return obs, (1 + np.sum(null >= obs)) / (1 + n_perm), null


def describe(route_name, r, p1, err, n_bins=8):
    rho_c = spearmanr(r, p1).correlation
    rho_e = spearmanr(r, err).correlation
    print(f"\n  [{route_name}] Spearman(r, max-softmax) = {rho_c:+.3f}   Spearman(r, top-1 error) = {rho_e:+.3f}")
    edges = np.unique(np.quantile(r, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, len(edges) - 2)
    print("  top-1 error by quantile bin of r  (non-monotone => see the sequence):")
    for k in range(len(edges) - 1):
        m = b == k
        if m.sum():
            print(f"    r in [{r[m].min():>5.0f}, {r[m].max():>5.0f}]  n={m.sum():>4}  err={err[m].mean():.3f}  mean max-softmax={p1[m].mean():.3f}")
    # error vs r inside confidence terciles
    ct = np.digitize(p1, np.quantile(p1, [1 / 3, 2 / 3]))
    rt = np.digitize(r, np.unique(np.quantile(r, [1 / 3, 2 / 3])))
    print("  top-1 error, rows = confidence tercile (low->high), cols = r tercile (low->high):")
    for c in range(3):
        row = []
        for t in range(rt.max() + 1):
            m = (ct == c) & (rt == t)
            row.append(f"{err[m].mean():.3f} (n={m.sum()})" if m.sum() >= 20 else "   --   ")
        print("    " + "   ".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--bench", default="CIFAR10")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max_rotation", type=int, default=179)
    ap.add_argument("--splits", default="precalibration,calibration")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--routes", default="symmetric,agreement,asymmetric_span")
    args = ap.parse_args()

    print("Loading caches")
    P, y = load_pooled(args.cache_dir, args.bench, args.seed, args.max_rotation, args.splits.split(","))
    p0 = P[0].astype(np.float64)
    n = len(y)
    p1 = p0.max(1)
    err = (p0.argmax(1) != y).astype(int)
    print(f"  pooled N={n}  top-1 accuracy={1 - err.mean():.4f}  alpha={args.alpha}")

    routes = {}
    for r in args.routes.split(","):
        fn = {"symmetric": compute_rotation_invariance, "agreement": compute_agreement_count,
              "asymmetric_span": compute_asymmetric_span}[r]
        routes[r] = fn(P, args.max_rotation).astype(float)
    del P

    thr_scores = 1.0 - p0[np.arange(n), y]
    aps_scores = aps_true_scores(p0, y)
    q_thr, q_aps = conformal_quantile(thr_scores, args.alpha), conformal_quantile(aps_scores, args.alpha)
    targets = {"top1": err, "thr": (thr_scores > q_thr).astype(int), "aps": (aps_scores > q_aps).astype(int)}
    print(f"  global THR q={q_thr:.4f} (miscoverage rate {targets['thr'].mean():.3f}); "
          f"global APS q={q_aps:.4f} (miscoverage rate {targets['aps'].mean():.3f})")

    conf = rank01(np.log(np.clip(p1, EPS, 1 - EPS) / np.clip(1 - p1, EPS, 1)))
    strata = np.minimum((conf * 10).astype(int), 9)

    print("\n=== DESCRIPTIVE ===")
    for name, r in routes.items():
        describe(name, r, p1, err)

    rng = np.random.default_rng(args.seed)
    print("\n=== CROSS-VALIDATED INFORMATION TEST ===")
    print(f"  {args.n_folds}-fold CV x {args.n_repeats} repeats; bootstrap={args.n_boot}; permutations={args.n_perm}")
    for rname, r in routes.items():
        feats = {"conf": conf, "route": rank01(r)}
        print(f"\n--- routing feature: {rname}  (distinct values: {len(np.unique(r))}) ---")
        for tname, t in targets.items():
            print(f"\n  target = {tname}   (positive rate {t.mean():.3f}, n_pos={t.sum()})")
            print(f"  {'model':<16} {'AUROC':>8} {'(sd)':>7} {'logloss':>9} {'(sd)':>7}")
            base_ll = log_loss(t, np.full(n, t.mean()))
            print(f"  {'prior only':<16} {0.5:>8.4f} {'':>7} {base_ll:>9.4f}")
            oof = {}
            for mname, (factory, cols) in make_models(args.seed).items():
                X = np.column_stack([feats[c] for c in cols])
                oof[mname], rep = oof_predict(factory, X, t, args.n_folds, args.n_repeats, args.seed)
                print(f"  {mname:<16} {rep[:, 0].mean():>8.4f} {rep[:, 0].std():>7.4f} {rep[:, 1].mean():>9.4f} {rep[:, 1].std():>7.4f}")
            for a, b in [("conf", "conf+route"), ("gbm-conf", "gbm-conf+route")]:
                (lo_a, hi_a), (lo_l, hi_l) = paired_bootstrap(t, oof[a], oof[b], args.n_boot, rng)
                print(f"  Δ({b} - {a}):  AUROC 95% CI [{lo_a:+.4f}, {hi_a:+.4f}]   logloss reduction 95% CI [{lo_l:+.5f}, {hi_l:+.5f}]")
            obs, pval, null = permutation_pvalue(t, conf, feats["route"], args.n_perm, args.n_folds, args.seed, strata)
            print(f"  permutation test (route shuffled within confidence deciles, additive spline model): "
                  f"logloss reduction obs={obs:+.5f}  null mean={null.mean():+.5f}  p={pval:.3f}")


if __name__ == "__main__":
    main()
