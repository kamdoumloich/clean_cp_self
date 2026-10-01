import json
import os
from math import ceil
from typing import Dict, List, Mapping, Optional, Any

import numpy as np

from approaches.tta_baselines import _MixturePredictorBase, EPS
from precalibrator.bin_optimizer import (
    PrecalibrationBinOptimizer,
    compute_rotation_invariance,
    compute_agreement_count,
    compute_asymmetric_span,
    compute_routing_metric,
)


class RotationInvariantPredictor(_MixturePredictorBase):
    """
    Ensemble Conformal Predictor with Rotation-Invariance Bins.

    Concept:
    - Measures how robust the top-1 prediction is under symmetric rotation:
      d(x) is the maximum degree d such that rotating x by any angle in [-d, d]
      preserves the prediction of the unrotated input.
    - Continuous ranges of d ("bins") partition the invariance domain.
    - Precalibration: Solves an optimization problem on the precalibration split
      to jointly select bin boundaries [L_b, R_b] and base thresholds tau_b that
      minimize the total conformal set size subject to overall (1 - alpha) coverage.
    - Calibration: On the follow-up calibration set, determines a single multiplicative
      factor lambda such that thresholds min(1.0, lambda * tau_b) achieve the
      exact finite-sample (1 - alpha) marginal coverage guarantee.
    - Prediction: An input x is routed to its bin based on d(x), and uses that bin's
      calibrated threshold.
    """

    def __init__(
        self,
        precalibration_data,
        alpha: float,
        path_logs: str,
        n_augs: int,
        n_bins: int = 3,
        n_classes: int = 10,
        seed: int = 2026,
        inputs_type: str = "probs",
        n_leading_views: int = 1,
        leading_views_mode: str = "individual",
        reference_view: int = 0,
        leading_pool: str = "arithmetic",
        leading_others_as_aug: bool = False,
        scoring_function: str = "thr",
        routing_metric: str = "symmetric",
        max_rotation: int = 179,
        min_samples_per_bin: int = 20,
        min_bin_coverage: Optional[float] = None,
        coverage_slack: Optional[float] = None,
        min_degree_span: int = 2,
        score_in_log_space: bool = False,
        verbose: bool = True,
        symmetric_approach: bool = True,
    ):
        super().__init__(
            precalibration_data=precalibration_data,
            alpha=alpha,
            path_logs=path_logs,
            n_classes=n_classes,
            n_augs=n_augs,
            seed=seed,
            verbose=verbose,
            inputs_type=inputs_type,
            scoring_function=scoring_function,
            n_leading_views=n_leading_views,
            leading_views_mode=leading_views_mode,
            reference_view=reference_view,
            leading_pool=leading_pool,
            leading_others_as_aug=leading_others_as_aug,
            aps_randomized=False,
            score_in_log_space=score_in_log_space,
        )

        self.n_bins = int(n_bins)
        self.routing_metric = str(routing_metric).lower().strip()
        self.max_rotation = int(max_rotation)
        self.min_samples_per_bin = int(min_samples_per_bin)
        self.min_bin_coverage = float(min_bin_coverage) if min_bin_coverage is not None else None
        self.coverage_slack = float(coverage_slack) if coverage_slack is not None else None
        self.min_degree_span = int(min_degree_span)
        self.score_in_log_space = bool(score_in_log_space)
        self.symmetric_approach = symmetric_approach

        if self.max_rotation < 1:
            raise ValueError("max_rotation must be >= 1.")

        self.optimized_bins: List[Dict[str, Any]] = []
        self.bin_thresholds_precal: Dict[int, float] = {}
        self.precal_report_text: str = ""
        self.precal_results: Optional[Dict[str, Any]] = None
        self.scaling_factor: Optional[float] = None
        self.qhat_by_bin: Dict[int, float] = {}
        self.is_calibrated: bool = False
        self.qhat = None

        # Last prediction state for diagnostics
        self.last_cell: Optional[int] = None
        self.last_qhat: Optional[float] = None
        self.last_scores: Optional[np.ndarray] = None
        self.last_degree: Optional[int] = None

    def _compute_routing_metric(self, P_views: np.ndarray) -> np.ndarray:
        P = np.asarray(P_views, dtype=np.float64)
        expected_views = 1 + 2 * self.max_rotation
        if P.shape[0] != expected_views:
            # If extra leading views exist (e.g. multiple leading views before rotation probes),
            # extract the unrotated reference view and the rotation probe views
            if P.shape[0] >= self.n_leading_views + 2 * self.max_rotation:
                p_ref = P[self.reference_view : self.reference_view + 1]
                p_rot = P[self.n_leading_views : self.n_leading_views + 2 * self.max_rotation]
                P = np.concatenate([p_ref, p_rot], axis=0)
            elif P.shape[0] >= 2 * self.max_rotation + 1:
                P = np.concatenate([P[0:1], P[-2 * self.max_rotation :]], axis=0)
        return compute_routing_metric(
            P,
            max_rotation=self.max_rotation,
            metric=self.routing_metric,
        )

    def _get_bin_for_degree(self, d: int) -> int:
        """Finds the bin index b that contains invariance degree d."""
        for b in self.optimized_bins:
            if b["min_degree"] <= d <= b["max_degree"]:
                return b["bin_index"]
        if d < self.optimized_bins[0]["min_degree"]:
            return self.optimized_bins[0]["bin_index"]
        return self.optimized_bins[-1]["bin_index"]

    def precalibrate(self):
        """
        Runs the precalibration bin and threshold optimization on precalibration data.
        Prints the precalibration coverage of each bin, thresholds, and bin sizes.
        """
        os.makedirs(self.path_logs, exist_ok=True)
        self.theta = self._leading_theta()

        # Parse precalibration views
        P_views = self._to_probs(self.p_views_pre) # (n_views, N_pre, K)
        p_ref = P_views[self.reference_view]        # (N_pre, K)
        y_precal = self.targets_pre                 # (N_pre,)

        N_pre = len(y_precal)
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"Starting Precalibration for {self.getShortName()} (N_pre={N_pre}, alpha={self.alpha}, B={self.n_bins}, metric={self.routing_metric})")
            print("=" * 80)

        # 1. Compute routing metric for each precalibration sample
        degrees = self._compute_routing_metric(P_views)
        max_metric_val = 2 * self.max_rotation if self.routing_metric not in ["symmetric", "sym", "d_sym"] else self.max_rotation
        max_degree = int(max(degrees.max(), max_metric_val))

        # 2. Run PrecalibrationBinOptimizer
        optimizer = PrecalibrationBinOptimizer(
            n_bins=self.n_bins,
            alpha=self.alpha,
            min_samples_per_bin=self.min_samples_per_bin,
            scoring_function=self.scoring_function,
            min_bin_coverage=self.min_bin_coverage,
            coverage_slack=self.coverage_slack,
            min_degree_span=self.min_degree_span,
            score_in_log_space=self.score_in_log_space,
            solver="milp",
        )

        res = optimizer.optimize(
            degrees=degrees,
            p_ref=p_ref,
            y_true=y_precal,
            max_degree=max_degree,
            scoring_function=self.scoring_function,
            score_in_log_space=self.score_in_log_space,
        )

        self.precal_results = res
        self.optimized_bins = res["bins"]
        self.bin_thresholds_precal = {b["bin_index"]: b["threshold"] for b in self.optimized_bins}

        # 3. Print precalibration coverage of each bin
        self.precal_report_text = optimizer.format_precalibration_report(res)
        print("\n" + self.precal_report_text + "\n", flush=True)

        # Save precalibration report to disk
        metric_tag = f"_{self.routing_metric}" if self.routing_metric not in ["symmetric", "sym", "d_sym"] else ""
        score_tag = f"_{self.scoring_function}" if self.scoring_function != "thr" else ""
        log_tag = "_logspace" if self.score_in_log_space else ""
        report_file = os.path.join(self.path_logs, f"precalibration_report{metric_tag}_alpha{self.alpha}_bins{self.n_bins}{score_tag}{log_tag}.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(self.precal_report_text + "\n")

        self.is_calibrated = False
        self.scaling_factor = None
        self.qhat_by_bin = {}

    def calibrate(self, calibration_data, alpha: Optional[float] = None):
        """
        Calibrates the multiplicative factor lambda on the follow-up calibration set.
        """
        if not self.optimized_bins:
            raise RuntimeError("Call precalibrate() before calibrate().")

        a = self.alpha if alpha is None else float(alpha)
        if not (0.0 < a < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {a}.")

        V, y = self._parse(calibration_data)
        V = self._to_probs(V)
        y = np.asarray(y, dtype=np.int64)

        N_cal = len(y)
        if N_cal == 0:
            raise ValueError("Calibration set is empty.")

        p_ref = V[self.reference_view] # (N_cal, K)
        # Nonconformity scores s(x, y) based on scoring_function
        true_scores = self._true_scores(p_ref, y, score=self.scoring_function)

        # Compute routing metric for each calibration sample
        degrees_cal = self._compute_routing_metric(V)

        # Map each calibration sample to its bin
        sample_bins = np.array([self._get_bin_for_degree(int(d)) for d in degrees_cal], dtype=int)

        # Required coverage count: ceil((N_cal + 1) * (1 - alpha))
        k_cal = int(ceil((N_cal + 1) * (1.0 - a)))

        # Precalibration thresholds for each sample: tau_{b(j)}
        base_sample_thresholds = np.array([self.bin_thresholds_precal[b] for b in sample_bins], dtype=float)

        # Compute required factor lambda for each sample: r_j = s_j / tau_{b(j)}
        # To avoid division by zero or extreme instability, guard against tau <= 1e-4
        min_tau_guard = 1e-4
        ratios = np.zeros(N_cal, dtype=float)
        for j in range(N_cal):
            tau_b = base_sample_thresholds[j]
            s_j = true_scores[j]
            ratios[j] = s_j / max(tau_b, min_tau_guard)

        # Find conformal quantile factor lambda
        if k_cal > N_cal:
            scaling_factor = np.inf
        else:
            scaling_factor = float(np.partition(ratios, k_cal - 1)[k_cal - 1])

        self.scaling_factor = scaling_factor

        # Scaled thresholds per bin:
        # In probability space: tau_b^{final} = min(1.0, lambda * tau_b)
        # In log space: tau_b^{final} = lambda * tau_b (no 1.0 ceiling)
        self.qhat_by_bin = {}
        for b in self.optimized_bins:
            b_idx = b["bin_index"]
            base_tau = self.bin_thresholds_precal[b_idx]
            if np.isinf(scaling_factor):
                cal_tau = np.inf if self.score_in_log_space else 1.0
            else:
                if self.score_in_log_space:
                    cal_tau = float(scaling_factor * base_tau)
                else:
                    cal_tau = min(1.0, float(scaling_factor * base_tau))
            self.qhat_by_bin[b_idx] = cal_tau

        # Verify calibration coverage
        cal_sample_taus = np.array([self.qhat_by_bin[b] for b in sample_bins], dtype=float)
        covered_mask = true_scores <= cal_sample_taus
        n_covered_cal = int(covered_mask.sum())
        cal_coverage = n_covered_cal / N_cal

        self.is_calibrated = True
        self.qhat = self.qhat_by_bin

        if self.verbose:
            print("\n" + "=" * 80)
            print(f"CALIBRATION STAGE ({self.getShortName()}):")
            print(f"  - Calibration samples: {N_cal}")
            print(f"  - Required covered: {k_cal}/{N_cal} (target >= {1.0 - a:.4f})")
            print(f"  - Multiplicative scaling factor lambda: {scaling_factor:.6f}")
            print(f"  - Achieved calibration coverage: {n_covered_cal}/{N_cal} ({cal_coverage:.4f})")
            print("  - Calibrated thresholds by bin:")
            for b in self.optimized_bins:
                b_idx = b["bin_index"]
                base_t = self.bin_thresholds_precal[b_idx]
                final_t = self.qhat_by_bin[b_idx]
                n_b_cal = int((sample_bins == b_idx).sum())
                cov_b_cal = int((covered_mask & (sample_bins == b_idx)).sum())
                pct_b = cov_b_cal / max(1, n_b_cal)
                print(f"      Bin {b_idx} [{b['min_degree']:>3}..{b['max_degree']:<3} deg]: "
                      f"base_tau={base_t:.6f} -> final_tau={final_t:.6f}, "
                      f"cal_samples={n_b_cal}, cov={cov_b_cal}/{n_b_cal} ({pct_b:.4f})")
            print("=" * 80 + "\n", flush=True)

        return self.qhat_by_bin

    def predict(self, predictedLogitVector: np.ndarray, calibrationValue=None):
        """
        Generates conformal prediction set for an input sample.

        Parameters
        ----------
        predictedLogitVector : np.ndarray of shape (n_views, K)
        calibrationValue : unused, kept for API compatibility

        Returns
        -------
        (classes, nextUp) : tuple of (list of included class indices, None)
        """
        if not self.is_calibrated:
            raise RuntimeError("Call calibrate() before predict().")

        V = np.asarray(predictedLogitVector, dtype=np.float64)
        if V.ndim != 2:
            raise ValueError(f"Expected input shape (n_views, K), got {V.shape}.")

        # Convert to probabilities if necessary
        P = self._to_probs(V[:, None, :]) # (n_views, 1, K)
        p0 = P[self.reference_view, 0]    # (K,)

        # Compute routing metric for this single sample
        deg = int(self._compute_routing_metric(P)[0])
        bin_idx = self._get_bin_for_degree(deg)
        tau = self.qhat_by_bin[bin_idx]

        # Nonconformity score for each class using self._all_scores
        scores = self._all_scores(p0, score=self.scoring_function)

        # Generate prediction set with at_least_one fallback to eliminate empty sets
        classes, nextUp = self._set_from_scores_predict_only(
            scores, tau, at_least_one=False, score_in_log_space=self.score_in_log_space
        )

        # Update last state for diagnostics / compact.py
        self.last_cell = bin_idx
        self.last_qhat = tau
        self.last_scores = scores
        self.last_degree = deg

        return classes, nextUp

    def texInfo(self) -> str:
        degree_unit = "$^\\circ$" if self.routing_metric in ["symmetric", "sym", "d_sym"] else ""
        b_str = ", ".join(
            f"[{b['min_degree']}..{b['max_degree']}{degree_unit}]"
            for b in self.optimized_bins
        )
        slack_str = f" with coverage slack $\\epsilon = {self.coverage_slack:.4f}$" if self.coverage_slack is not None else ""
        log_desc = " in log-space" if self.score_in_log_space else ""
        if self.routing_metric in ["agreement", "n_agree", "option1", "agree"]:
            metric_desc = f"Global Agreement Count $N_{{\\text{{agree}}}}$"
        elif self.routing_metric in ["asymmetric_span", "asym_span", "span", "w_total", "option2a"]:
            metric_desc = f"Total Asymmetric Span $W = d_L + d_R$"
        else:
            metric_desc = f"Continuous Symmetric Rotation Invariance $d(x)$"

        return (
            f"Ensemble Conformal Predictor ({metric_desc}, scoring: {self.scoring_function.upper()}{log_desc}) "
            f"with {self.n_bins} optimized bins: {b_str}. "
            f"Base thresholds pre-calibrated via joint MILP optimization on precalibration split{slack_str}, "
            f"scaled by a single factor $\\lambda = {self.scaling_factor if self.scaling_factor is not None else 1.0:.4f}$ "
            f"on calibration split for finite-sample $(1-\\alpha)$ marginal coverage."
        )

    def getShortName(self) -> str:
        score_tag = f"-{self.scoring_function.upper()}" if self.scoring_function != "thr" else ""
        log_tag = "-Log" if self.score_in_log_space else ""
        if self.routing_metric in ["agreement", "n_agree", "option1", "agree"]:
            prefix = "Agree-CP"
        elif self.routing_metric in ["asymmetric_span", "asym_span", "span", "w_total", "option2a"]:
            prefix = "AsymSpan-CP"
        else:
            prefix = "RI-CP"
        return f"{prefix}{score_tag}{log_tag}-B{self.n_bins}"


class AgreementCountPredictor(RotationInvariantPredictor):
    """
    Option 1: Global Agreement Count N_{agree} Conformal Predictor.

    Routes samples based on the number of rotated views whose top-1 prediction
    matches the unrotated view's top-1 prediction:
      N_agree(x) in [0, 2 * max_rotation]
    Uses APS scoring function by default.
    """

    def __init__(
        self,
        *args,
        scoring_function: str = "aps",
        routing_metric: str = "agreement",
        **kwargs,
    ):
        super().__init__(
            *args,
            scoring_function=scoring_function,
            routing_metric=routing_metric,
            **kwargs,
        )


class AsymmetricSpanPredictor(RotationInvariantPredictor):
    """
    Option 2a: Total Asymmetric Span W = d_L + d_R Conformal Predictor.

    Routes samples based on the total continuous rotation invariance span across
    both negative and positive rotation directions:
      W(x) = d_left(x) + d_right(x) in [0, 2 * max_rotation]
    where d_left and d_right are continuous invariance spans without prediction flip.
    Uses APS scoring function by default.
    """

    def __init__(
        self,
        *args,
        scoring_function: str = "aps",
        routing_metric: str = "asymmetric_span",
        **kwargs,
    ):
        super().__init__(
            *args,
            scoring_function=scoring_function,
            routing_metric=routing_metric,
            **kwargs,
        )
