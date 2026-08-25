import json
import os
from dataclasses import dataclass
from math import ceil
from typing import Dict, Mapping, Optional

import numpy as np

from approaches.tta_baselines import _MixturePredictorBase, EPS


@dataclass
class InvariantRangeConformalPredictor:
    """Independent conformal calibration state for one range."""

    cp_id: int
    invariant_rotation_range: dict
    alpha: float
    qhat: float = float("inf")
    n_calibration: int = 0

    def calibrate(self, true_scores):
        import utils.function_utils
        
        S = np.asarray(true_scores, dtype=np.float64).ravel()
        self.n_calibration = int(S.size)
        self.qhat = utils.function_utils.UtilsConformalPrediction.compute_calibration_value(S, self.alpha)
        # self.qhat = self.conformal_quantile(S, self.alpha)
        return self.qhat
    


class RotationInvariantPredictor(_MixturePredictorBase):
    """
    One distinct split-conformal predictor per range.

    Precalibration:
        - compute the invariant interval ranges
        - filter them to select stable ones
        - freeze the range partition
        - create one InvariantRangeConformalPredictor for each retained invariant range

    Calibration:
        - route every calibration sample to its frozen range
        - compute THR scores s(x,y) = 1 - p_y (through the base score primitive)
        - version 1: calibrate a separate qhat_c from the true-label scores in **each range**

    Prediction:
        - route x to range c
        - use that range's qhat_c only
        - C_c(x) = {y : s_THR(x,y) <= qhat_c}

    """

    def __init__(
        self,
        precalibration_data,
        alpha,
        path_logs,
        n_classes: int = 10,
        seed: int = 2026,
        inputs_type: str = "probs",
        # leading-view block
        n_leading_views: int = 1,
        leading_views_mode: str = "individual",
        reference_view: int = 0,
        leading_pool: str = "arithmetic",
        leading_others_as_aug: bool = False,
        score='thr',
        # range construction
        score_chunk: int = 8000,
        max_rotation: int = 10,
        # diagnostics
        # feature_diagnostics: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            precalibration_data=precalibration_data,
            alpha=alpha,
            path_logs=path_logs,
            n_classes=n_classes,
            seed=seed,
            verbose=verbose,
            inputs_type=inputs_type,
            score=score,
            n_leading_views=n_leading_views,
            leading_views_mode=leading_views_mode,
            reference_view=reference_view,
            leading_pool=leading_pool,
            leading_others_as_aug=leading_others_as_aug,
            aps_randomized=False,
        )

        self.score_chunk = None if score_chunk is None else int(score_chunk)
        self.feature_diagnostics = bool(verbose)
        self.max_rotation = int(max_rotation)
        if self.max_rotation < 1:
            raise ValueError("max_rotation must be >= 1.")

        rotation_angles = np.concatenate(
            [
                np.arange(-self.max_rotation, 0, dtype=np.int32),
                np.arange(1, self.max_rotation + 1, dtype=np.int32),
            ]
        )
        self.aug_angles = np.asarray(rotation_angles, dtype=np.int32)

        if len(self.aug_angles) != len(self.aug_indices):
            raise ValueError(
                f"Received {len(self.aug_angles)} rotation angles but "
                f"{len(self.aug_indices)} augmentation views."
            )
        if len(np.unique(self.aug_angles)) != len(self.aug_angles):
            raise ValueError("Every rotation angle must occur exactly once.")

        for r in np.unique(np.abs(self.aug_angles)):
            values = set(self.aug_angles[np.abs(self.aug_angles) == r].tolist())
            if values != {-int(r), int(r)}:
                raise ValueError(f"Expected both {-int(r)} and {int(r)} degrees.")

        self.amount_samples = self.n
        self.allTargets = self.targets_pre
        if np.any(self.allTargets < 0) or np.any(self.allTargets >= self.K):
            raise ValueError("Precalibration targets contain invalid class indices.")

        # This is the minimum n_c for which the standard finite-sample
        # (1-alpha) split-conformal order statistic is finite.
        self.min_sample_per_interval = max(1, int(ceil(1.0 / self.alpha)) - 1)
        # self.min_sample_per_interval = 100.0

        # Frozen by precalibrate().
        self.selected_rotation_ranges = {}
        self.mapping_rotation_range_to_cp_id = {}
        self.cp_id_to_single_cp_object: Dict[int, InvariantRangeConformalPredictor] = {}
        self.qhat_by_cp_id = {}

        self.edges = None
        self.feature_names = None
        self.precal_ranges = None
        self.precal_range_counts: Dict[int, int] = {}
        self.n_invariant_rotation_range = None

        # Created by precalibrate(), fitted by calibrate().
        self.qhat_by_range: Dict[int, float] = {}
        self.calibration_range_counts: Dict[int, int] = {}
        self.calibration_report = None
        self.global_qhat_diagnostic = None
        self.is_calibrated = False

        # Backward-facing attribute. It becomes a dict after calibrate().
        self.qhat = None

        self.already_printed = False
        self.last_range = None
        self.last_qhat = None
        self.last_scores = None

        if self.verbose:
            print(
                "Initialization of RotationInvariantPredictor done "
                f"(views={self.n_views}, leading={self.n_leading_views}/"
                f"{self.leading_views_mode}, augmentations={self.A}, "
                f"alpha={self.alpha}, min_sample_per_interval={self.min_sample_per_interval}).",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Base hooks
    # ------------------------------------------------------------------
    def _predictive_distribution(self, V):
        return self._pool_leading(V)

    def _check_precalibration_ready(self):
        if not self.selected_rotation_ranges:
            raise RuntimeError("Call precalibrate() before calibrate() or predict().")

    def _get_intervals_above_threshold(self, candidate_ranges_and_counts, threshold:int):
        intervals = {}

        for cluster in candidate_ranges_and_counts["clusters"]:
            if cluster["count"] >= threshold:
                intervals[cluster["joint_interval"]] = cluster["count"]

        for interval, count in candidate_ranges_and_counts["untouched"].items():
            if count >= threshold:
                intervals[interval] = count

        return intervals
    
    def _features_from_prob_views(self, P, p_ref):
        
        unique_ranges_and_counts = self._stable_angular_ranges_from_prob_views(P, p_ref)

        return unique_ranges_and_counts["sample_ranges"]

    
    def _stable_angular_ranges_from_prob_views(self, P, p_ref):
        """
        For each sample, find the largest contiguous angular interval [L, U]
        containing 0 such that every prediction inside the interval has the
        same top-1 class as the original view at 0 degrees.

        Example:
            angles/top1:
                -4  -3  -2  -1   0   1   2   3   4
                B   B   A   A   A   A   A   C   C

            -> stable interval = [-2, 2]

        The interval does NOT have to be symmetric.

        Worst case:
            -1 != top0 and +1 != top0
            -> interval = [0, 0]

        Returns
        -------
        {
            "sample_ranges": np.ndarray,     # shape (N, 2), [L, U] per sample
            "range_counts": dict,            # {(L, U): number_of_samples}
        }
        """
        P = np.asarray(P, dtype=np.float64)

        if P.ndim == 2:
            P = P[:, None, :]

        if P.ndim != 3:
            raise ValueError(f"Expected (M,N,K) or (M,K), got {P.shape}.")

        if P.shape[0] != self.n_views:
            raise ValueError( f"Expected {self.n_views} views, got {P.shape[0]}.")

        p0 = np.asarray(p_ref, dtype=np.float64)

        if p0.ndim == 1:
            p0 = p0[None, :]

        if p0.ndim != 2:
            raise ValueError(f"Expected p_ref (N,K) or (K,), got {p0.shape}.")

        Q = P[self.aug_indices]
        angles = np.asarray(self.aug_angles, dtype=np.int32)

        if Q.shape[0] != len(angles):
            raise ValueError("aug_angles is not aligned with aug_indices.")

        # ---------------------------------------------------------
        # Top-1 prediction for original and rotated views
        # ---------------------------------------------------------
        top0 = np.argmax(p0, axis=1)       # (N,)
        top_aug = np.argmax(Q, axis=2)     # (A, N)

        # ---------------------------------------------------------
        # Insert original image at angle 0 and sort all rotations
        # ---------------------------------------------------------
        all_angles = np.concatenate([angles, np.array([0], dtype=np.int32)])

        all_argmax = np.concatenate([top_aug, top0[None, :]], axis=0)

        sorted_list_angles = np.argsort(all_angles)

        ordered_angles = all_angles[sorted_list_angles]
        ordered_all_argmax = all_argmax[sorted_list_angles]       # (A+1, N)

        position_rotation_zero = np.flatnonzero(ordered_angles == 0)

        if len(position_rotation_zero) != 1:
            raise ValueError("Exactly one reference angle 0 is required.")

        zero_idx = int(position_rotation_zero[0])
        N = p0.shape[0]

        same_as_original = (ordered_all_argmax == top0[None, :])

        # LEFT boundary
        left_matches = same_as_original[:zero_idx][::-1]

        if left_matches.shape[0] == 0:
            left_steps = np.zeros(N, dtype=np.int32)
        else:
            left_mismatch = ~left_matches
            has_left_mismatch = left_mismatch.any(axis=0)
            first_left_mismatch = np.argmax(left_mismatch, axis=0)
            left_steps = np.where(has_left_mismatch, first_left_mismatch, left_matches.shape[0])

        left_indices = zero_idx - left_steps
        lower_bounds = ordered_angles[left_indices]

        # RIGHT boundary
        right_matches = same_as_original[zero_idx + 1:]

        if right_matches.shape[0] == 0:
            right_steps = np.zeros(N, dtype=np.int32)
        else:
            right_mismatch = ~right_matches
            has_right_mismatch = right_mismatch.any(axis=0)
            first_right_mismatch = np.argmax(right_mismatch, axis=0)
            right_steps = np.where(has_right_mismatch, first_right_mismatch, right_matches.shape[0])

        right_indices = zero_idx + right_steps
        upper_bounds = ordered_angles[right_indices]

        # [L, U] per sample sample
        # Very important at this step
        sample_ranges = np.column_stack([lower_bounds,upper_bounds]).astype(np.int32)

        unique_ranges, counts = np.unique(sample_ranges, axis=0, return_counts=True)

        range_counts = {
            (int(L), int(U)): int(count) for (L, U), count in zip(unique_ranges, counts)
        }

        return {
            "sample_ranges": sample_ranges,  # shape (N, 2)
            "range_counts": range_counts,
        }

    # ------------------------------------------------------------------
    # THR score + frozen routing
    # ------------------------------------------------------------------

    def _thr_scores_and_ranges_from_views(self, V):
        import utils.function_utils

        V = np.asarray(V, dtype=np.float64)

        if V.shape[0] != self.n_views:
            raise ValueError(f"Expected {self.n_views} views, got {V.shape[0]}.")

        if V.shape[-1] != self.K:
            raise ValueError(f"Expected K={self.K}, got {V.shape[-1]}.")

        P_views = self._to_probs(V)
        p_ref = np.atleast_2d(self._pool_leading(V))

        # Shape: (N, 2)
        stability_intervals = self._features_from_prob_views(P_views, p_ref)

        default_cp_id = self.mapping_rotation_range_to_cp_id.get((0,0))
        assert not default_cp_id is None, "Problem with the default CP (0,0)"

        # One CP/range ID per sample
        sample_cp_ids = np.asarray([
            self.mapping_rotation_range_to_cp_id.get((int(L), int(U)), default_cp_id)
            for L, U in stability_intervals
        ], dtype=np.int32)

        return sample_cp_ids


    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def precalibrate(self):
        """Fit and freeze the range partition; do not calibrate any CP here."""
        import utils.function_utils

        os.makedirs(self.path_logs, exist_ok=True)
        self.theta = self._leading_theta()

        P_views = self._to_probs(self.p_views_pre)
        p_ref = self._pool_leading(self.p_views_pre)

        result = self._stable_angular_ranges_from_prob_views(P_views, p_ref)

        sample_ranges = result["sample_ranges"]
        unique_ranges_and_counts = result["range_counts"]

        tmp_str = utils.function_utils.UtilsGeneral.dict_items_to_string(unique_ranges_and_counts)
        utils.function_utils.UtilsGeneral.write_string_to_file(tmp_str, path=f"{self.path_logs}invariant_interval_{self.seed}.log")

        self.selected_rotation_ranges = set(unique_ranges_and_counts.keys())
        # Default case
        self.selected_rotation_ranges.add((0, 0))
        # if not (0,0) in self.selected_rotation_ranges:
        #     self.selected_rotation_ranges = self.selected_rotation_ranges | (0,0)

        cp_ids = range(len(self.selected_rotation_ranges))
        self.mapping_rotation_range_to_cp_id = {
            key: cp_id for key, cp_id in zip(self.selected_rotation_ranges, cp_ids)
            }

        self.cp_id_to_single_cp_object = {
            cp_id: InvariantRangeConformalPredictor(cp_id=cp_id, alpha=self.alpha, invariant_rotation_range=key) 
            for key, cp_id in zip(self.selected_rotation_ranges, cp_ids)
            }

        self.n_invariant_rotation_range = len(self.selected_rotation_ranges)

        self.is_calibrated = False
        self.qhat = None

        if self.verbose:
            observed = len(unique_ranges_and_counts)
            print(
                f"[QCTHR] precalibration done: {observed} observed ranges, "
                f"{self.n_invariant_rotation_range} selected ranges, \n"
                f"computed min amount of samples for alpha={self.alpha}: "
                f"{self.min_sample_per_interval}\n",
                flush=True,
            )


    def calibrate(self, calibration_data, alpha=None):
        """
        Calibrate one independent THR threshold per range.

        Returns
        -------
        dict[int, float]
            Mapping range_id -> qhat_range. The prediction method accepts this
            dictionary directly as calibrationValue for compatibility with runners
            that pass calibrate()'s return value to predict().
        """
        import utils.function_utils
        
        self._check_precalibration_ready()

        a = self.alpha if alpha is None else float(alpha)
        if not (0.0 < a < 1.0):
            raise ValueError("alpha must be in (0,1).")

        V, y = self._parse(calibration_data)
        y = np.asarray(y, dtype=np.int32)
        if len(y) == 0:
            raise ValueError("Calibration data is empty.")
        if np.any(y < 0) or np.any(y >= self.K):
            raise ValueError("Calibration targets contain invalid class indices.")

        sample_cp_ids = self._thr_scores_and_ranges_from_views(V)

        S = self._all_scores(self._pool_leading(V))

        rows = np.arange(len(y))
        true_scores = S[rows, y]

        self.calibration_range_counts = {}
        self.qhat_by_range = {}
        report = {}

        for loop_rotation_range in self.selected_rotation_ranges:
            this_cp_id = self.mapping_rotation_range_to_cp_id[loop_rotation_range]
            cp = self.cp_id_to_single_cp_object.get(this_cp_id)

            # ranges contains CP IDs
            mask = sample_cp_ids == this_cp_id
            n_c = int(np.sum(mask))

            cp.alpha = a

            if n_c > 0:
                q_c = cp.calibrate(true_scores[mask])
            else:
                q_c = float("inf")

            self.calibration_range_counts[loop_rotation_range] = n_c
            self.qhat_by_range[loop_rotation_range] = float(q_c)
            self.qhat_by_cp_id[this_cp_id] = float(q_c)

            if n_c > 0:
                member_c = S[mask] <= q_c
                y_c = y[mask]
                rows_c = np.arange(n_c)
                sizes_c = member_c.sum(axis=1)
                coverage_c = float(np.mean(member_c[rows_c, y_c]))
                avg_size_c = float(np.mean(sizes_c))
            else:
                coverage_c = None
                avg_size_c = None

            report[loop_rotation_range] = {
                "n": n_c,
                "qhat": float(q_c),
                "finite_qhat": bool(np.isfinite(q_c)),
                "coverage_in_sample": coverage_c,
                "avg_size_in_sample": avg_size_c,
            }
     
        self.global_qhat_diagnostic = utils.function_utils.UtilsConformalPrediction.compute_calibration_value(true_scores, a)

        # Overall in-sample diagnostics under range-specific thresholds.
        tau_rows = np.asarray(
            [self.qhat_by_cp_id[cp_id] for cp_id in sample_cp_ids], dtype=np.float64
        )
        member = S <= tau_rows[:, None]
        sizes = member.sum(axis=1)
        coverage = float(np.mean(member[rows, y]))

        self.calibration_report = {
            "alpha": a,
            "n": int(len(y)),
            "overall_coverage_in_sample": coverage,
            "overall_avg_size_in_sample": float(np.mean(sizes)),
            "overall_median_size_in_sample": float(np.median(sizes)),
            "global_qhat_diagnostic": float(self.global_qhat_diagnostic),
            "ranges": report,
        }

        self.qhat = dict(self.qhat_by_range)
        self.is_calibrated = True
        self._write_log()

        if self.verbose:
            finite_ranges = sum(np.isfinite(q) for q in self.qhat_by_range.values())
            observed_cal_ranges = sum(n > 0 for n in self.calibration_range_counts.values())
            print("="*20)
            print(self.calibration_range_counts)
            print("="*20)
            sparse_observed = sum(
                0 < n < self.min_sample_per_interval
                for n in self.calibration_range_counts.values()
            )
            print(
                f"[QCTHR] calibration done: n={len(y)}, "
                f"observed_cal_ranges={observed_cal_ranges}/{self.n_invariant_rotation_range}, "
                f"finite_qhat_ranges={finite_ranges}/{self.n_invariant_rotation_range}, "
                f"observed-but-too-small={sparse_observed}, "
                f"overall in-sample coverage={coverage:.4f}, "
                f"avg size={np.mean(sizes):.4f}",
                flush=True,
            )

        return dict(self.qhat_by_range)

    def predict(self, predictedLogitVector, calibrationValue=None):
        self._check_calibration_ready()

        V = np.asarray(predictedLogitVector, dtype=np.float64)
        if V.ndim != 2:
            raise ValueError(f"Expected one sample with shape (M,K), got {V.shape}.")

        ranges = self._thr_scores_and_ranges_from_views(V)

        z = self.compute_prob_dist(predictedLogitVector[0], list(predictedLogitVector[1:]))
        u = float(self._rng.uniform()) if (self.score == "aps" and self.aps_randomized) else 0.0
        scores = self._all_scores(z, score=self.score, u=u)
        
        # scores = S[0]
        range = int(ranges[0])

        default_qhat = self.qhat_by_range.get((0,0))

        assert not default_qhat is None, "Problem in prediction with default CP"

        if calibrationValue is None:
            tau = self.qhat_by_range.get(range, default_qhat)
        else:
            raise NotImplementedError(f"Invalid calibration value.")
            # arr = np.asarray(calibrationValue, dtype=np.float64).ravel()
            # if arr.size != 1:
            #     raise TypeError("calibrationValue must be None, a range->qhat mapping, or a scalar debug override.")
            # tau = float(arr[0])

        tau = float(tau)
        self.last_range = range
        self.last_qhat = tau
        self.last_scores = scores

        return self._set_from_scores(scores, tau)

    # ------------------------------------------------------------------
    # Diagnostics / metadata
    # ------------------------------------------------------------------
    @staticmethod
    def _json_float(x):
        if x is None:
            return None
        x = float(x)
        return x if np.isfinite(x) else "inf"

    def _write_log(self):
        os.makedirs(self.path_logs, exist_ok=True)
        log_path = os.path.join(
            self.path_logs,
            f"QCTHR_log_{self.seed}_alpha{self.alpha}.json",
        )

        payload = {
            "method": "RotationInvariantPredictor",
            "alpha": self.alpha,
            "seed": self.seed,
            "inputs_type": self.inputs_type,
            "n_views": self.n_views,
            "n_leading_views": self.n_leading_views,
            "leading_views_mode": self.leading_views_mode,
            "leading_pool": self.leading_pool,
            "reference_view": self.reference_view,
            "aug_indices": list(self.aug_indices),
            "aug_angles": np.asarray(self.aug_angles).tolist(),
            "n_augmentations": self.A,
            "K": self.K,
            "score": "thr",
            "feature_names": self.feature_names,
            "edges": None
            if self.edges is None
            else [np.asarray(e, dtype=float).tolist() for e in self.edges],
            "n_ranges_total": self.n_invariant_rotation_range,
            "calibration_range_counts": {
                str(c): int(n) for c, n in self.calibration_range_counts.items()
            },
            "qhat_by_range": {
                str(c): self._json_float(q)
                for c, q in self.qhat_by_range.items()
            },
            "global_qhat_diagnostic": self._json_float(
                self.global_qhat_diagnostic
            ),
            "is_calibrated": self.is_calibrated,
        }

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        if self.verbose:
            print(f"[QCTHR] log written to {log_path}", flush=True)

    def texInfo(self):
        g = self.n_invariant_rotation_range if self.n_invariant_rotation_range is not None else "?"
        return (
            "Quantile-Cell THR Conformal Prediction: rotated views define a frozen "
            f"quantile partition with {g} possible ranges. Each range owns a distinct "
            "THR split-conformal predictor and is calibrated only with calibration "
            "samples routed to that range. Prediction uses the range-specific "
            "finite-sample conformal threshold; ranges with insufficient calibration "
            "mass use the conservative full-set threshold."
        )

    def getShortName(self):
        g = self.n_invariant_rotation_range if self.n_invariant_rotation_range is not None else 0
        return f"QCTHR-G{g}"
