import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix


def compute_rotation_invariance(P_views: np.ndarray, max_rotation: int) -> np.ndarray:
    """
    Computes the rotation-invariance degree d(x) for each sample.
    d(x) is the largest integer d in [0, max_rotation] such that rotating
    the input by any value r in [-d, d] preserves the top-1 predicted class
    of the unrotated input (view 0).

    Parameters
    ----------
    P_views : np.ndarray
        Array of shape (n_views, n_samples, K) or (n_views, K).
        View 0 is the original unrotated image.
        Views 1..max_rotation correspond to negative angles [-max_rotation, ..., -1].
        Views max_rotation+1..2*max_rotation correspond to positive angles [1, ..., max_rotation].
    max_rotation : int
        Maximum tested rotation angle in degrees (e.g. 179).

    Returns
    -------
    invariance_degrees : np.ndarray of shape (n_samples,), dtype int
    """
    P = np.asarray(P_views, dtype=np.float64)
    if P.ndim == 2:
        P = P[:, None, :]

    n_views, n_samples, K = P.shape
    expected_views = 1 + 2 * max_rotation
    if n_views != expected_views:
        raise ValueError(f"Expected {expected_views} views (1 original + 2*{max_rotation} rotations), got {n_views}.")

    p0 = P[0] # (N, K)
    top0 = np.argmax(p0, axis=-1) # (N,)

    # Views 1..max_rotation are [-max_rotation, ..., -1]
    # Views max_rotation+1..2*max_rotation are [+1, ..., +max_rotation]
    neg_views = P[1 : max_rotation + 1] # shape (max_rotation, N, K)
    pos_views = P[max_rotation + 1 : 2 * max_rotation + 1] # shape (max_rotation, N, K)

    symmetric_mask = np.ones(n_samples, dtype=bool)
    invariance = np.zeros(n_samples, dtype=np.int64)

    # Step d from 1 to max_rotation
    # d = 1 corresponds to:
    # neg_views[max_rotation - 1] (which is -1 deg)
    # pos_views[0] (which is +1 deg)
    for d in range(1, max_rotation + 1):
        idx_neg = max_rotation - d # for d=1 -> max_rotation - 1 (-1 deg)
        idx_pos = d - 1            # for d=1 -> 0 (+1 deg)

        match_neg = (np.argmax(neg_views[idx_neg], axis=-1) == top0)
        match_pos = (np.argmax(pos_views[idx_pos], axis=-1) == top0)

        symmetric_mask = symmetric_mask & match_neg & match_pos
        invariance = np.where(symmetric_mask, d, invariance)

    return invariance


def compute_agreement_count(P_views: np.ndarray, max_rotation: int) -> np.ndarray:
    """
    Computes Option 1: Global Agreement Count N_agree(x).
    Counts how many rotated views yield the same top-1 predicted class as
    the unrotated reference view (view 0).
    Range: [0, 2 * max_rotation].
    """
    P = np.asarray(P_views, dtype=np.float64)
    if P.ndim == 2:
        P = P[:, None, :]

    n_views, n_samples, K = P.shape
    expected_views = 1 + 2 * max_rotation
    if n_views != expected_views:
        raise ValueError(f"Expected {expected_views} views (1 original + 2*{max_rotation} rotations), got {n_views}.")

    p0 = P[0] # (N, K)
    top0 = np.argmax(p0, axis=-1) # (N,)

    neg_views = P[1 : max_rotation + 1] # (max_rotation, N, K)
    pos_views = P[max_rotation + 1 : 2 * max_rotation + 1] # (max_rotation, N, K)

    agree_neg = (np.argmax(neg_views, axis=-1) == top0[None, :]).sum(axis=0)
    agree_pos = (np.argmax(pos_views, axis=-1) == top0[None, :]).sum(axis=0)

    return (agree_neg + agree_pos).astype(np.int64)


def compute_asymmetric_span(P_views: np.ndarray, max_rotation: int) -> np.ndarray:
    """
    Computes Option 2a: Total Asymmetric Span W(x) = d_L(x) + d_R(x).
    d_L(x) is the largest integer in [0, max_rotation] such that rotating by
    any angle in [-d_L, -1] preserves the unrotated top-1 prediction.
    d_R(x) is the largest integer in [0, max_rotation] such that rotating by
    any angle in [+1, +d_R] preserves the unrotated top-1 prediction.
    Range: [0, 2 * max_rotation].
    """
    P = np.asarray(P_views, dtype=np.float64)
    if P.ndim == 2:
        P = P[:, None, :]

    n_views, n_samples, K = P.shape
    expected_views = 1 + 2 * max_rotation
    if n_views != expected_views:
        raise ValueError(f"Expected {expected_views} views (1 original + 2*{max_rotation} rotations), got {n_views}.")

    p0 = P[0] # (N, K)
    top0 = np.argmax(p0, axis=-1) # (N,)

    neg_views = P[1 : max_rotation + 1] # (max_rotation, N, K)
    pos_views = P[max_rotation + 1 : 2 * max_rotation + 1] # (max_rotation, N, K)

    left_mask = np.ones(n_samples, dtype=bool)
    d_left = np.zeros(n_samples, dtype=np.int64)
    for d in range(1, max_rotation + 1):
        idx_neg = max_rotation - d
        m_neg = (np.argmax(neg_views[idx_neg], axis=-1) == top0)
        left_mask = left_mask & m_neg
        d_left = np.where(left_mask, d, d_left)

    right_mask = np.ones(n_samples, dtype=bool)
    d_right = np.zeros(n_samples, dtype=np.int64)
    for d in range(1, max_rotation + 1):
        idx_pos = d - 1
        m_pos = (np.argmax(pos_views[idx_pos], axis=-1) == top0)
        right_mask = right_mask & m_pos
        d_right = np.where(right_mask, d, d_right)

    w_total = d_left + d_right
    return w_total.astype(np.int64)


def compute_routing_metric(
    P_views: np.ndarray,
    max_rotation: int = 179,
    metric: str = "symmetric",
) -> np.ndarray:
    """
    Unified dispatcher for input invariance / stability routing metrics:
      - "symmetric" (default): Symmetric continuous rotation invariance d(x) in [0, max_rotation]
      - "agreement" (Option 1): Global agreement count N_agree(x) in [0, 2*max_rotation]
      - "asymmetric_span" (Option 2a): Total asymmetric continuous span W(x) = d_L + d_R in [0, 2*max_rotation]
    """
    P = np.asarray(P_views, dtype=np.float64)
    expected_views = 1 + 2 * max_rotation
    if P.shape[0] > expected_views and P.shape[0] >= 2 * max_rotation + 1:
        # Extract view 0 (ORIGINAL reference) and the last 2*max_rotation views (rotation probes)
        P = np.concatenate([P[0:1], P[-2 * max_rotation :]], axis=0)

    m = str(metric).lower().strip()
    if m in ["symmetric", "sym", "d_sym"]:
        return compute_rotation_invariance(P, max_rotation=max_rotation)
    elif m in ["agreement", "n_agree", "option1", "agree"]:
        return compute_agreement_count(P, max_rotation=max_rotation)
    elif m in ["asymmetric_span", "asym_span", "span", "w_total", "option2a"]:
        return compute_asymmetric_span(P, max_rotation=max_rotation)
    else:
        raise ValueError(
            f"Unknown routing metric: '{metric}'. Choose from 'symmetric', 'agreement', 'asymmetric_span'."
        )


class PrecalibrationBinOptimizer:
    """
    Optimizes bin split points and per-bin base calibration thresholds tau_b
    on the precalibration dataset to minimize total conformal set size
    subject to the overall precalibration coverage constraint:
        sum_i 1{ y_i in C(x_i) } >= ceil((1 - alpha) * N_pre).
    """

    def __init__(
        self,
        n_bins: int = 3,
        alpha: float = 0.1,
        min_samples_per_bin: int = 20,
        n_threshold_quantiles: int = 60,
        max_candidate_splits: int = 25,
        scoring_function: str = "thr",
        min_bin_coverage: Optional[float] = None,
        coverage_slack: Optional[float] = None,
        min_degree_span: int = 2,
        score_in_log_space: bool = False,
        solver: str = "milp",
    ):
        self.n_bins = int(n_bins)
        self.alpha = float(alpha)
        self.min_samples_per_bin = int(min_samples_per_bin)
        self.n_threshold_quantiles = int(n_threshold_quantiles)
        self.max_candidate_splits = int(max_candidate_splits)
        self.scoring_function = str(scoring_function).lower()
        self.min_bin_coverage = float(min_bin_coverage) if min_bin_coverage is not None else None
        self.coverage_slack = float(coverage_slack) if coverage_slack is not None else None
        self.min_degree_span = int(min_degree_span)
        self.score_in_log_space = bool(score_in_log_space)
        self.solver = solver

    def optimize(
        self,
        degrees: np.ndarray,
        p_ref: np.ndarray,
        y_true: np.ndarray,
        max_degree: Optional[int] = None,
        scoring_function: Optional[str] = None,
        score_in_log_space: Optional[bool] = None,
        metric_unit: str = "deg",
    ) -> Dict[str, Any]:
        """
        Runs the joint optimization over bin boundaries and thresholds.

        Parameters
        ----------
        degrees : (N,) array of integer invariance values
        p_ref : (N, K) array of class probabilities
        y_true : (N,) array of ground truth class indices
        max_degree : optional upper limit of invariance range
        scoring_function : optional override of scoring function ("thr", "aps")

        Returns
        -------
        Dictionary containing optimized bins, thresholds, and coverage report.
        """
        N = len(degrees)
        if N == 0:
            raise ValueError("Precalibration set is empty.")

        degrees = np.asarray(degrees, dtype=np.int64)
        p_ref = np.asarray(p_ref, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.int64)

        K_target = int(np.ceil((1.0 - self.alpha) * N))
        D_max = int(degrees.max()) if max_degree is None else int(max_degree)

        # Nonconformity scores based on scoring_function and log-space setting
        scoring_fn = self.scoring_function if scoring_function is None else str(scoring_function).lower()
        use_log_space = self.score_in_log_space if score_in_log_space is None else bool(score_in_log_space)
        if scoring_fn == "thr":
            all_scores = 1.0 - p_ref
        elif scoring_fn == "aps":
            order = np.argsort(-p_ref, axis=1, kind="stable")
            Ps = np.take_along_axis(p_ref, order, axis=1)
            csum = np.cumsum(Ps, axis=1)
            prev = np.zeros_like(csum)
            prev[:, 1:] = csum[:, :-1]
            all_scores = np.empty_like(Ps)
            np.put_along_axis(all_scores, order, prev, axis=1)
        else:
            raise ValueError(f"Unsupported scoring function: {scoring_fn}")

        if use_log_space:
            all_scores = -np.log(np.clip(1.0 - all_scores, 1e-24, 1.0))

        true_scores = all_scores[np.arange(N), y_true]

        # Determine candidate boundary split points
        unique_degs = np.unique(degrees)
        if len(unique_degs) <= self.max_candidate_splits:
            candidate_splits = unique_degs
        else:
            # Include quantiles of observed degrees
            quantiles = np.linspace(5, 95, self.max_candidate_splits)
            candidate_splits = np.percentile(degrees, quantiles).astype(int)
            candidate_splits = np.unique(candidate_splits)

        all_boundaries = np.unique(np.concatenate(([0], candidate_splits, [D_max])))
        M = len(all_boundaries) - 1

        # Adjust min_samples_per_bin if dataset is small
        effective_min_samples = min(self.min_samples_per_bin, max(5, N // (2 * self.n_bins)))

        # Build candidate intervals (L, R)
        interval_dict = {}
        for i in range(len(all_boundaries)):
            for j in range(i + 1, len(all_boundaries)):
                L = 0 if i == 0 else int(all_boundaries[i] + 1)
                R = int(all_boundaries[j])
                if L <= R and (R - L + 1) >= self.min_degree_span:
                    sample_indices = np.where((degrees >= L) & (degrees <= R))[0]
                    if len(sample_indices) >= effective_min_samples:
                        s_bin = all_scores[sample_indices]
                        y_bin = y_true[sample_indices]
                        t_bin = true_scores[sample_indices]

                        # Generate candidate thresholds up to 1.0 quantile to enable high-coverage targets (e.g. alpha=0.01)
                        n_q = min(self.n_threshold_quantiles, len(sample_indices))
                        q_grid = np.linspace(0.0, 1.0, n_q)
                        cand_taus = np.unique(np.quantile(t_bin, q_grid))
                        if use_log_space:
                            min_tau_floor = 1e-4
                            cand_taus = np.unique(cand_taus[cand_taus >= min_tau_floor])
                            if len(cand_taus) == 0:
                                cand_taus = np.array([min_tau_floor, float(np.max(t_bin))])
                        else:
                            # Floor candidate thresholds to prevent ratio division collapse in calibration,
                            # and clip to 1.0 to guard against floating-point precision overflow
                            min_tau_floor = 0.05 if scoring_fn == "thr" else 0.01
                            cand_taus = np.clip(cand_taus, min_tau_floor, 1.0)
                            cand_taus = np.unique(cand_taus[cand_taus >= min_tau_floor])
                            if len(cand_taus) == 0:
                                cand_taus = np.array([min_tau_floor, 1.0])

                        cands = []
                        last_cov = -1
                        if self.coverage_slack is not None:
                            max_errors = max(1, int(np.floor((self.alpha + self.coverage_slack) * len(sample_indices))))
                            min_cov_samples = max(0, len(sample_indices) - max_errors)
                        elif self.min_bin_coverage is not None:
                            min_cov_samples = int(np.ceil(self.min_bin_coverage * len(sample_indices)))
                        else:
                            min_cov_samples = 0

                        for tau in cand_taus:
                            sets = (s_bin <= tau)
                            empty_mask = (sets.sum(axis=1) == 0)
                            if empty_mask.any():
                                top1 = np.argmax(p_ref[sample_indices[empty_mask]], axis=1)
                                sets[empty_mask, top1] = True

                            cov = int(sets[np.arange(len(sample_indices)), y_bin].sum())
                            sz = int(sets.sum())
                            if cov >= min_cov_samples and cov != last_cov:
                                cands.append((cov, sz, float(tau)))
                                last_cov = cov

                        if cands:
                            interval_dict[(i, j)] = (L, R, len(sample_indices), sample_indices, cands)

        # Fallback if no valid partition exists with given min_samples
        if len(interval_dict) < self.n_bins:
            # Relax to full range split into n_bins quantiles
            bin_edges = np.quantile(degrees, np.linspace(0, 1, self.n_bins + 1)).astype(int)
            bin_edges[0] = 0
            bin_edges[-1] = D_max
            return self._solve_fixed_bins(
                degrees, all_scores, y_true, true_scores, bin_edges, K_target,
                p_ref=p_ref, scoring_fn=scoring_fn, score_in_log_space=use_log_space
            )

        # Solve via MILP
        return self._solve_milp(
            interval_dict=interval_dict,
            M=M,
            N=N,
            K_target=K_target,
            degrees=degrees,
            all_scores=all_scores,
            y_true=y_true,
            true_scores=true_scores,
            D_max=D_max,
            p_ref=p_ref,
            scoring_fn=scoring_fn,
            score_in_log_space=use_log_space,
        )

    def _solve_milp(
        self,
        interval_dict: Dict,
        M: int,
        N: int,
        K_target: int,
        degrees: np.ndarray,
        all_scores: np.ndarray,
        y_true: np.ndarray,
        true_scores: np.ndarray,
        D_max: int,
        p_ref: Optional[np.ndarray] = None,
        scoring_fn: str = "thr",
        score_in_log_space: bool = False,
    ) -> Dict[str, Any]:
        intervals = list(interval_dict.keys())
        n_intervals = len(intervals)

        x_records = [] # (var_idx, it_idx, cov, sz, tau)
        cur_idx = n_intervals
        for it_idx, it in enumerate(intervals):
            _, _, _, _, cands = interval_dict[it]
            for cov, sz, tau in cands:
                x_records.append((cur_idx, it_idx, cov, sz, tau))
                cur_idx += 1

        n_vars = cur_idx
        c_obj = np.zeros(n_vars)
        for v_idx, it_idx, cov, sz, tau in x_records:
            c_obj[v_idx] = sz

        rows, cols, vals = [], [], []
        row_idx = 0
        lb, ub = [], []

        # 1. Exactly n_bins intervals chosen
        for it_idx in range(n_intervals):
            rows.append(row_idx)
            cols.append(it_idx)
            vals.append(1.0)
        lb.append(self.n_bins)
        ub.append(self.n_bins)
        row_idx += 1

        # 2. Leave node 0
        for it_idx, (i, j) in enumerate(intervals):
            if i == 0:
                rows.append(row_idx)
                cols.append(it_idx)
                vals.append(1.0)
        lb.append(1.0)
        ub.append(1.0)
        row_idx += 1

        # 3. Enter node M
        for it_idx, (i, j) in enumerate(intervals):
            if j == M:
                rows.append(row_idx)
                cols.append(it_idx)
                vals.append(1.0)
        lb.append(1.0)
        ub.append(1.0)
        row_idx += 1

        # 4. Flow conservation at intermediate nodes 1..M-1
        for v in range(1, M):
            for it_idx, (i, j) in enumerate(intervals):
                if j == v:
                    rows.append(row_idx)
                    cols.append(it_idx)
                    vals.append(1.0)
                elif i == v:
                    rows.append(row_idx)
                    cols.append(it_idx)
                    vals.append(-1.0)
            lb.append(0.0)
            ub.append(0.0)
            row_idx += 1

        # 5. sum_k x_{it, k} - u_{it} = 0
        it_to_x = {}
        for v_idx, it_idx, cov, sz, tau in x_records:
            it_to_x.setdefault(it_idx, []).append(v_idx)

        for it_idx in range(n_intervals):
            rows.append(row_idx)
            cols.append(it_idx)
            vals.append(-1.0)
            for v_idx in it_to_x.get(it_idx, []):
                rows.append(row_idx)
                cols.append(v_idx)
                vals.append(1.0)
            lb.append(0.0)
            ub.append(0.0)
            row_idx += 1

        # 6. Global coverage >= K_target
        for v_idx, it_idx, cov, sz, tau in x_records:
            rows.append(row_idx)
            cols.append(v_idx)
            vals.append(float(cov))
        lb.append(float(K_target))
        ub.append(np.inf)
        row_idx += 1

        A = coo_matrix((vals, (rows, cols)), shape=(row_idx, n_vars)).tocsc()
        constraints = LinearConstraint(A, lb=lb, ub=ub)
        integrality = np.ones(n_vars, dtype=int)
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))

        res = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints)

        if not res.success:
            # Fallback to quantile split if MILP finds no solution under strict constraints
            bin_edges = np.quantile(degrees, np.linspace(0, 1, self.n_bins + 1)).astype(int)
            bin_edges[0] = 0
            bin_edges[-1] = D_max
            return self._solve_fixed_bins(
                degrees=degrees,
                all_scores=all_scores,
                y_true=y_true,
                true_scores=true_scores,
                bin_edges=bin_edges,
                K_target=K_target,
                p_ref=p_ref,
                scoring_fn=scoring_fn,
            )

        # Parse solution
        selected_bins = []
        for v_idx, it_idx, cov, sz, tau in x_records:
            if res.x[v_idx] > 0.5:
                L, R, n_samples, sample_indices, _ = interval_dict[intervals[it_idx]]
                selected_bins.append({
                    "min_degree": int(L),
                    "max_degree": int(R),
                    "n_samples": int(n_samples),
                    "threshold": float(tau),
                    "covered_count": int(cov),
                    "coverage": float(cov / max(1, n_samples)),
                    "set_size_sum": float(sz),
                    "average_set_size": float(sz / max(1, n_samples)),
                    "sample_indices": sample_indices,
                })

        # Sort bins by min_degree
        selected_bins.sort(key=lambda b: b["min_degree"])
        for b_idx, b in enumerate(selected_bins):
            b["bin_index"] = b_idx

        # Ensure full contiguous coverage of [0, D_max]
        if selected_bins:
            selected_bins[0]["min_degree"] = 0
            selected_bins[-1]["max_degree"] = D_max

        total_covered = sum(b["covered_count"] for b in selected_bins)
        total_set_size = sum(b["set_size_sum"] for b in selected_bins)

        result = {
            "bins": selected_bins,
            "n_bins": len(selected_bins),
            "total_samples": N,
            "total_covered": total_covered,
            "total_set_size": total_set_size,
            "precalibration_coverage": total_covered / N,
            "precalibration_average_set_size": total_set_size / N,
            "target_coverage": 1.0 - self.alpha,
            "required_covered": K_target,
            "scoring_function": scoring_fn,
            "score_in_log_space": score_in_log_space,
        }
        return result

    def _solve_fixed_bins(
        self,
        degrees: np.ndarray,
        all_scores: np.ndarray,
        y_true: np.ndarray,
        true_scores: np.ndarray,
        bin_edges: np.ndarray,
        K_target: int,
        p_ref: Optional[np.ndarray] = None,
        scoring_fn: str = "thr",
        score_in_log_space: bool = False,
    ) -> Dict[str, Any]:
        """
        Fallback when arbitrary edge graph has no feasible integer solution:
        solves optimal threshold allocation across the fixed quantile intervals.
        """
        N = len(degrees)
        edges = np.unique(np.asarray(bin_edges, dtype=int))
        if len(edges) < 2:
            edges = np.array([0, int(degrees.max()) if len(degrees) else 179], dtype=int)

        raw_bins = []
        for i in range(len(edges) - 1):
            L = 0 if i == 0 else int(edges[i] + 1)
            R = int(edges[i + 1])
            if L <= R:
                idx = np.where((degrees >= L) & (degrees <= R))[0]
                raw_bins.append((L, R, idx))

        bins_data = []
        for L, R, idx in raw_bins:
            if len(idx) == 0:
                if bins_data:
                    prev_L, prev_R, prev_idx = bins_data[-1]
                    bins_data[-1] = (prev_L, max(prev_R, R), prev_idx)
            else:
                if not bins_data and L > 0:
                    L = 0
                bins_data.append((L, R, idx))

        if not bins_data:
            bins_data.append((0, int(edges[-1]), np.arange(len(degrees), dtype=int)))
        else:
            last_L, last_R, last_idx = bins_data[-1]
            if last_R < edges[-1]:
                bins_data[-1] = (last_L, int(edges[-1]), last_idx)

        B = len(bins_data)

        # Solve optimal thresholds for these B bins using scipy.optimize.milp
        var_records = []
        cur_v = 0
        for b_idx, (L, R, idx) in enumerate(bins_data):
            s_bin = all_scores[idx]
            y_bin = y_true[idx]
            t_bin = true_scores[idx]
            cand_taus = np.unique(np.quantile(t_bin, np.linspace(0, 1.0, 50)))
            if score_in_log_space:
                min_tau_floor = 1e-4
                cand_taus = np.unique(cand_taus[cand_taus >= min_tau_floor])
                if len(cand_taus) == 0:
                    cand_taus = np.array([min_tau_floor, float(np.max(t_bin))])
            else:
                min_tau_floor = 0.05 if scoring_fn == "thr" else 0.01
                cand_taus = np.clip(cand_taus, min_tau_floor, 1.0)
                cand_taus = np.unique(cand_taus[cand_taus >= min_tau_floor])
                if len(cand_taus) == 0:
                    cand_taus = np.array([min_tau_floor, 1.0])

            if self.coverage_slack is not None:
                max_errors = max(1, int(np.floor((self.alpha + self.coverage_slack) * len(idx))))
                min_cov_samples = max(0, len(idx) - max_errors)
            elif self.min_bin_coverage is not None:
                min_cov_samples = int(np.ceil(self.min_bin_coverage * len(idx)))
            else:
                min_cov_samples = 0

            bin_cands = []
            for tau in cand_taus:
                sets = (s_bin <= tau)
                if p_ref is not None:
                    empty_mask = (sets.sum(axis=1) == 0)
                    if empty_mask.any():
                        top1 = np.argmax(p_ref[idx[empty_mask]], axis=1)
                        sets[empty_mask, top1] = True

                cov = int(sets[np.arange(len(idx)), y_bin].sum())
                sz = int(sets.sum())
                if cov >= min_cov_samples:
                    bin_cands.append((cov, sz, float(tau)))

            # Fallback if no threshold met min_cov_samples: use highest candidate to prevent solver crash
            if not bin_cands:
                tau = float(cand_taus[-1])
                sets = (s_bin <= tau)
                if p_ref is not None:
                    empty_mask = (sets.sum(axis=1) == 0)
                    if empty_mask.any():
                        top1 = np.argmax(p_ref[idx[empty_mask]], axis=1)
                        sets[empty_mask, top1] = True
                cov = int(sets[np.arange(len(idx)), y_bin].sum())
                sz = int(sets.sum())
                bin_cands.append((cov, sz, tau))

            for cov, sz, tau in bin_cands:
                var_records.append((cur_v, b_idx, cov, sz, tau))
                cur_v += 1

        n_vars = cur_v
        c_obj = np.array([r[3] for r in var_records], dtype=float)

        rows, cols, vals = [], [], []
        row_idx = 0
        lb, ub = [], []

        for b_idx in range(B):
            for v_idx, b, cov, sz, tau in var_records:
                if b == b_idx:
                    rows.append(row_idx)
                    cols.append(v_idx)
                    vals.append(1.0)
            lb.append(1.0)
            ub.append(1.0)
            row_idx += 1

        for v_idx, b, cov, sz, tau in var_records:
            rows.append(row_idx)
            cols.append(v_idx)
            vals.append(float(cov))
        lb.append(float(K_target))
        ub.append(np.inf)
        row_idx += 1

        A = coo_matrix((vals, (rows, cols)), shape=(row_idx, n_vars)).tocsc()
        constraints = LinearConstraint(A, lb=lb, ub=ub)
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        integrality = np.ones(n_vars, dtype=int)

        res = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints)

        selected_bins = []
        for b_idx in range(B):
            cand_vars = [r for r in var_records if r[1] == b_idx]
            chosen = max(cand_vars, key=lambda r: res.x[r[0]] if res.success else r[2])
            L, R, idx = bins_data[b_idx]
            n_samples = len(idx)
            cov = chosen[2]
            sz = chosen[3]
            tau = chosen[4]
            selected_bins.append({
                "bin_index": b_idx,
                "min_degree": L,
                "max_degree": R,
                "n_samples": n_samples,
                "threshold": tau,
                "covered_count": cov,
                "coverage": cov / max(1, n_samples),
                "set_size_sum": sz,
                "average_set_size": sz / max(1, n_samples),
            })

        total_covered = sum(b["covered_count"] for b in selected_bins)
        total_set_size = sum(b["set_size_sum"] for b in selected_bins)

        return {
            "bins": selected_bins,
            "n_bins": len(selected_bins),
            "total_samples": N,
            "total_covered": total_covered,
            "total_set_size": total_set_size,
            "precalibration_coverage": total_covered / N,
            "precalibration_average_set_size": total_set_size / N,
            "target_coverage": 1.0 - self.alpha,
            "required_covered": K_target,
            "scoring_function": scoring_fn,
            "score_in_log_space": score_in_log_space,
        }

    @staticmethod
    def format_precalibration_report(result: Dict[str, Any]) -> str:
        """
        Formats a comprehensive, clean precalibration report table.
        """
        unit = result.get("metric_unit", "deg")
        score_name = result.get("scoring_function", "thr").upper()
        log_info = " (log-space: -log(p))" if result.get("score_in_log_space", False) else ""
        lines = []
        lines.append("=" * 84)
        lines.append("                      PRECALIBRATION BIN OPTIMIZATION REPORT")
        lines.append("=" * 84)
        lines.append(f"Scoring Function: {score_name}{log_info}")
        lines.append(f"Target Global Coverage: {result['target_coverage']:.4f} (Required Covered: {result['required_covered']}/{result['total_samples']})")
        lines.append(f"Achieved Precalibration Coverage: {result['precalibration_coverage']:.4f} ({result['total_covered']}/{result['total_samples']})")
        lines.append(f"Achieved Precalibration Mean Set Size: {result['precalibration_average_set_size']:.4f}")
        lines.append("-" * 84)
        header = f"{'Bin':<5} | {f'Range ({unit})':<15} | {'Samples':<8} | {'Threshold':<11} | {'Covered':<12} | {'Coverage':<10} | {'Avg Size':<9}"
        lines.append(header)
        lines.append("-" * 84)

        for b in result["bins"]:
            rng_str = f"[{b['min_degree']:>3}..{b['max_degree']:<3}]"
            cov_str = f"{b['covered_count']}/{b['n_samples']}"
            line = (
                f"{b['bin_index']:<5} | "
                f"{rng_str:<15} | "
                f"{b['n_samples']:<8} | "
                f"{b['threshold']:<11.6f} | "
                f"{cov_str:<12} | "
                f"{b['coverage']:<10.4f} | "
                f"{b['average_set_size']:<9.3f}"
            )
            lines.append(line)

        lines.append("=" * 84)
        return "\n".join(lines)


def compute_test_bin_comparison(
    ensemble_predictor,
    baseline_predictor,
    test_data: Tuple[np.ndarray, np.ndarray],
    max_rotation: int,
) -> Dict[str, Any]:
    """
    Evaluates both the ensemble and baseline predictors on the test set,
    aggregating results both overall and according to the ensemble's bins.
    """
    probabilities, targets = test_data
    N_test = len(targets)
    bins = ensemble_predictor.optimized_bins

    bin_stats = {
        b["bin_index"]: {
            "min_degree": b["min_degree"],
            "max_degree": b["max_degree"],
            "n_samples": 0,
            "ens_covered": 0,
            "ens_size_sum": 0,
            "base_covered": 0,
            "base_size_sum": 0,
        }
        for b in bins
    }

    total_ens_cov = 0
    total_ens_size = 0
    total_base_cov = 0
    total_base_size = 0

    for i in range(N_test):
        these_probs = probabilities[:, i, :]
        y = int(targets[i])

        ens_classes, _ = ensemble_predictor.predict(these_probs)
        base_classes, _ = baseline_predictor.predict(these_probs)

        b_idx = ensemble_predictor.last_cell
        s = bin_stats[b_idx]
        s["n_samples"] += 1

        is_ens_cov = int(y in ens_classes)
        s["ens_covered"] += is_ens_cov
        s["ens_size_sum"] += len(ens_classes)
        total_ens_cov += is_ens_cov
        total_ens_size += len(ens_classes)

        is_base_cov = int(y in base_classes)
        s["base_covered"] += is_base_cov
        s["base_size_sum"] += len(base_classes)
        total_base_cov += is_base_cov
        total_base_size += len(base_classes)

    rows = []
    for b_idx in sorted(bin_stats.keys()):
        s = bin_stats[b_idx]
        n_b = s["n_samples"]
        if n_b > 0:
            ens_cov = s["ens_covered"] / n_b
            ens_avg_sz = s["ens_size_sum"] / n_b
            base_cov = s["base_covered"] / n_b
            base_avg_sz = s["base_size_sum"] / n_b
            sz_diff = ens_avg_sz - base_avg_sz
            sz_red_pct = 100.0 * (base_avg_sz - ens_avg_sz) / max(1e-6, base_avg_sz)
        else:
            ens_cov = base_cov = ens_avg_sz = base_avg_sz = sz_diff = sz_red_pct = 0.0

        rows.append({
            "bin_index": b_idx,
            "min_degree": s["min_degree"],
            "max_degree": s["max_degree"],
            "n_samples": n_b,
            "ens_coverage": ens_cov,
            "ens_avg_size": ens_avg_sz,
            "base_coverage": base_cov,
            "base_avg_size": base_avg_sz,
            "size_diff": sz_diff,
            "size_reduction_pct": sz_red_pct,
        })

    unit = getattr(ensemble_predictor, "metric_unit", "deg")
    ens_name = getattr(ensemble_predictor, "getShortName", lambda: "Ensemble")()

    return {
        "rows": rows,
        "total_samples": N_test,
        "ensemble_total_coverage": total_ens_cov / N_test,
        "ensemble_total_avg_size": total_ens_size / N_test,
        "baseline_total_coverage": total_base_cov / N_test,
        "baseline_total_avg_size": total_base_size / N_test,
        "overall_size_reduction_pct": 100.0 * (total_base_size - total_ens_size) / max(1e-6, total_base_size),
        "metric_unit": unit,
        "ensemble_name": ens_name,
    }


def format_test_bin_comparison(res: Dict[str, Any], alpha: float) -> str:
    """
    Formats the comparative test performance table across bins.
    """
    unit = res.get("metric_unit", "deg")
    ens_name = res.get("ensemble_name", "Ensemble")
    lines = []
    lines.append("=" * 105)
    lines.append(f"      TEST SET COMPARISON: {ens_name.upper()} vs STANDARD BASELINE CONFORMAL PREDICTOR")
    lines.append("=" * 105)
    lines.append(f"Target Coverage (1 - alpha): {1.0 - alpha:.4f}")
    lines.append(f"Ensemble Total: Coverage={res['ensemble_total_coverage']:.4f}, Mean Set Size={res['ensemble_total_avg_size']:.4f}")
    lines.append(f"Baseline Total: Coverage={res['baseline_total_coverage']:.4f}, Mean Set Size={res['baseline_total_avg_size']:.4f}")
    lines.append(f"Overall Set Size Reduction: {res['overall_size_reduction_pct']:+.2f}%")
    lines.append("-" * 105)
    header = (
        f"{'Bin':<5} | {f'Range ({unit})':<15} | {'Samples':<8} | "
        f"{'Ens Cov':<9} | {'Base Cov':<9} | "
        f"{'Ens Size':<9} | {'Base Size':<9} | {'Size Gain':<10}"
    )
    lines.append(header)
    lines.append("-" * 105)

    for r in res["rows"]:
        rng_str = f"[{r['min_degree']:>3}..{r['max_degree']:<3}]"
        line = (
            f"{r['bin_index']:<5} | "
            f"{rng_str:<15} | "
            f"{r['n_samples']:<8} | "
            f"{r['ens_coverage']:<9.4f} | "
            f"{r['base_coverage']:<9.4f} | "
            f"{r['ens_avg_size']:<9.3f} | "
            f"{r['base_avg_size']:<9.3f} | "
            f"{r['size_reduction_pct']:>+8.2f}%"
        )
        lines.append(line)

    lines.append("=" * 105)
    return "\n".join(lines)
