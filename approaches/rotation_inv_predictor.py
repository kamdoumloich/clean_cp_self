import json
import os
from dataclasses import dataclass
from math import ceil
from typing import Dict, Mapping, Optional

import numpy as np

from approaches.tta_baselines import _MixturePredictorBase, EPS

import utils.function_utils

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
        n_augs:int,
        n_classes: int = 10,
        seed: int = 2026,
        inputs_type: str = "probs",
        # leading-view block
        n_leading_views: int = 1,
        leading_views_mode: str = "individual",
        reference_view: int = 0,
        leading_pool: str = "arithmetic",
        leading_others_as_aug: bool = False,
        scoring_function='thr',
        # range construction
        score_chunk: int = 8000,
        max_rotation: int = 10,
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
        )

        self.score_chunk = None if score_chunk is None else int(score_chunk)
        self.max_rotation = int(max_rotation)
        self.symmetric_approach = symmetric_approach
        if self.max_rotation < 1:
            raise ValueError("max_rotation must be >= 1.")

        rotation_angles = np.concatenate(
            [
                np.arange(-self.max_rotation, 0, dtype=np.int64),
                np.arange(1, self.max_rotation + 1, dtype=np.int64),
            ]
        )
        self.aug_angles = np.asarray(rotation_angles, dtype=np.int64)

        if len(self.aug_angles) != len(self.aug_indices):
            raise ValueError(f"Received {len(self.aug_angles)} rotation angles but {len(self.aug_indices)} augmentation views.")
        if len(np.unique(self.aug_angles)) != len(self.aug_angles):
            raise ValueError("Every rotation angle must occur exactly once.")

        # Make sure rotations are symmetric
        for r in np.unique(np.abs(self.aug_angles)):
            values = set(self.aug_angles[np.abs(self.aug_angles) == r].tolist())
            if values != {-int(r), int(r)}:
                raise ValueError(f"Expected both {-int(r)} and {int(r)} degrees.")

        # self.allTargets = self.targets_pre
        # if np.any(self.allTargets < 0) or np.any(self.allTargets >= self.K):
        #     raise ValueError("Precalibration targets contain invalid class indices.")

        # This is the minimum n_c for which the standard finite-sample
        self.min_sample_per_interval = max(1, int(ceil(1.0 / self.alpha)) - 1)

        # Frozen by precalibrate().
        self.selected_rotation_ranges = {}

        self.precal_range_counts: Dict[int, int] = {}
        self.n_invariant_rotation_range = None

        self.stat = {}
        self.best_precal_value_per_bin = {}
        self.final_cal_value_per_bin = {}
        self.extra_calibration_value = None
        
        # TODO: This acitvate the feature at_least_one in the precalibration set
        self.at_least_one_precalibration = False
        self.write_debug_files = False

        # Created by precalibrate(), fitted by calibrate().
        self.qhat_by_range: Dict[int, float] = {}
        self.calibration_range_counts: Dict[int, int] = {}
        self.calibration_report = None
        self.is_calibrated = False

        # is a dict after calibration
        self.qhat = None
        self.exhaustive_range_to_cp_mapping_list = {}

        self.last_range = None
        self.last_qhat = None
        self.last_scores = None

        if self.verbose:
            print(
                "Initialization of RotationInvariantPredictor done "
                f"(views={self.n_views}, leading={self.n_leading_views}/"
                f"{self.leading_views_mode}, augmentations={self.A}, "
                f"alpha={self.alpha}, min sample per valid precal bin={self.min_sample_per_interval}).",
                flush=True,
            )

    # ------------------------------------------------------------------

    def _check_precalibration_ready(self):
        if not self.selected_rotation_ranges:
            raise RuntimeError("Call precalibrate() before calibrate() or predict().")

    def _symmetric_invariant_intervals_from_prob_views(self, P, p_ref):
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
        angles = np.asarray(self.aug_angles, dtype=np.int64)

        if Q.shape[0] != len(angles) or (Q.shape[0] % 2) != 0:
            raise ValueError("aug_angles is not aligned with aug_indices.")

        top0 = np.argmax(p0, axis=-1)

        symmetric_mask = np.ones(len(p0), dtype=np.int64)
        symmetric_robustness_per_sample = np.zeros(len(p0), dtype=np.int64)

        rotation_1_degree_idx = int(len(Q)/2)

        for idx in range(rotation_1_degree_idx):
            idx_going_left = rotation_1_degree_idx - 1 - idx
            idx_going_right = rotation_1_degree_idx + idx

            same_as_orig_left = (np.argmax(Q[idx_going_left], axis=-1) == top0)
            same_as_orig_right = (np.argmax(Q[idx_going_right], axis=-1) == top0)

            symmetric_mask *= (same_as_orig_left * same_as_orig_right)

            symmetric_robustness_per_sample = np.where(symmetric_mask, symmetric_robustness_per_sample + 1, symmetric_robustness_per_sample)


        sample_idx_and_robustness = set()
        sample_count_per_robustness = {}
        for sample_idx, sample_robustness in enumerate(symmetric_robustness_per_sample):
            # print(sample_idx, "  -> ", sample_robustness)
            sample_idx_and_robustness.add((int(sample_idx), float(sample_robustness)))

            if sample_count_per_robustness.get(sample_robustness, None) == None:
                sample_count_per_robustness[sample_robustness] = 1
            else:
                sample_count_per_robustness[sample_robustness] = int(sample_count_per_robustness[sample_robustness]) + 1

        return {
            "sample_idx_and_robustness" : sample_idx_and_robustness,
            "sample_count_per_robustness" : sample_count_per_robustness,
        }


    # ------------------------------------------------------------------
    # THR score + frozen routing
    # ------------------------------------------------------------------

    def _thr_scores_and_ranges_from_views(self, V):
        V = np.asarray(V, dtype=np.float64)

        if V.shape[0] != self.n_views:
            raise ValueError(f"Expected {self.n_views} views, got {V.shape[0]}.")

        if V.shape[-1] != self.K:
            raise ValueError(f"Expected K={self.K}, got {V.shape[-1]}.")

        P_all_views = self._to_probs(V)
        p_ref = self._predictive_distribution(V)
        # p_ref = np.atleast_2d(self._pool_leading(V))

        # Shape: (N, 2)
        feature_stats = self._symmetric_invariant_intervals_from_prob_views(P_all_views, p_ref)

        sample_idx_and_robustness = feature_stats["sample_idx_and_robustness"]
        
        assert len(self.exhaustive_range_to_cp_mapping_list.keys()) == (self.max_rotation + 1), f"some invariant rotation cases are missing. {self.exhaustive_range_to_cp_mapping_list.keys()}"

        per_sample_cp_ids = {
            sample_idx: self.exhaustive_range_to_cp_mapping_list.get(degree_rotation) for sample_idx, degree_rotation in sample_idx_and_robustness
        } 

        return per_sample_cp_ids

    # def _first_step_create_bins_according_to_splitting_range(self, dict_of_samples_and_idx_per_robustness, arbitrary_splitting_ranges):
    #     # In this step I concretely create bins, but only based on sample idx
    #     # I return the list of indices per bin
    #     samples_idx_per_bin = {}

    #     for bin_number, (lb, ub) in enumerate(arbitrary_splitting_ranges):
    #         samples_idx_per_bin[bin_number] = []

    #         for robustness_degree in range(lb, ub + 1):
    #             self.exhaustive_range_to_cp_mapping_list[robustness_degree] = bin_number

    #             if robustness_degree not in dict_of_samples_and_idx_per_robustness:
    #                 # print(robustness_degree, "not observed")
    #                 continue

    #             samples_idx_per_bin[bin_number].extend(dict_of_samples_and_idx_per_robustness[robustness_degree])

    #     return samples_idx_per_bin

    def _first_step_create_bins_according_to_splitting_range(self, dict_of_samples_and_idx_per_robustness, arbitrary_splitting_ranges):
        robustness_degrees_per_bin = set()

        for i, (lb, ub) in enumerate(arbitrary_splitting_ranges):
            robustness_degrees_per_bin.add((i, range(lb, ub + 1)))
            for robustness_degree in range(lb, ub + 1):
                self.exhaustive_range_to_cp_mapping_list[robustness_degree] = i

        samples_idx_per_bin = {}
        available_robustness_degrees = dict_of_samples_and_idx_per_robustness.keys()

        for bin_number, robustness_degrees in robustness_degrees_per_bin:
            samples_idx_per_bin[bin_number] = []
            for robustness_degree in robustness_degrees:
                if robustness_degree not in available_robustness_degrees:
                    continue
                samples_idx_per_bin[bin_number].extend(dict_of_samples_and_idx_per_robustness[robustness_degree])

        return samples_idx_per_bin

    @staticmethod
    def _set_from_scores_matrix_per_bin(scores, tau, at_least_one=False):
        assert len(scores.shape) == 2, f"This function takes a score matrice, incorrect shape given {scores.shape}"
        s = np.asarray(scores, dtype=np.float64)
        K = s.shape[-1]
        if tau is None:
            raise RuntimeError("No calibration value defined.")
        tau = float(tau)

        if tau >= 1.0:
            # return np.full_like(scores, np.arange(K))
            return np.full(scores.shape, True)

        conformal_sets = np.where(scores <= tau, True, False) # True, False

        if at_least_one:
            for i, this_set in enumerate(conformal_sets):
                if np.sum(this_set) == 0:
                    conformal_sets[i][np.argmin(s[i])] = True

        return conformal_sets

    def _second_step_collect_precalibration_data_per_bin(self, bin_number, bin_member_indices, p_ref, y_precal,):

        indices = np.asarray(sorted(bin_member_indices), dtype=np.int64, )

        bin_samples = np.asarray(p_ref)[indices]
        bin_true_classes = np.asarray(y_precal, dtype=np.int64,)[indices]

        all_score_bin = self._all_scores(bin_samples)

        rows = np.arange(len(indices))
        true_scores_bin = all_score_bin[rows, bin_true_classes]

        # unique already sorts the result
        # candidate_cal_values = np.unique(np.clip(all_score_bin, None, 1.0))

        # Use only true scores to focus on the coverage -- don;t forget at_least_one=True
        # candidate_cal_values = np.unique(true_scores_bin)
        candidate_cal_values = np.unique(np.concatenate(([0.0], true_scores_bin)))

        stat_current_bin = []

        for calibration_val in candidate_cal_values:
            generated_sets = self._set_from_scores_matrix_per_bin(all_score_bin, calibration_val, at_least_one=self.at_least_one_precalibration)

            covered_count = int(generated_sets[np.arange(len(indices)), bin_true_classes,].sum())
            set_size_sum = int(generated_sets.sum())

            stat_current_bin.append([covered_count, set_size_sum, float(calibration_val),])
            
        if len(stat_current_bin) == 0:
            stat_current_bin.append([0, 0.0, 0.0])

        utils.function_utils.UtilsGeneral.write_set_to_file(stat_current_bin, path=f"{self.path_logs}stats_bin_{bin_number}.txt",)


    def _third_step_parallel_calibration(self, dict_samples_idx_per_bin,):

        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix

        bin_numbers = sorted(dict_samples_idx_per_bin)
        print("Bin received in thirst step: ", bin_numbers)
        amount_bins = len(bin_numbers)
        print("Amount bin in thirst step: ", amount_bins)
        
        # assert False
        

        if amount_bins == 0:
            raise ValueError("Something is wrong with the amount of bins")

        total_samples = sum(len(dict_samples_idx_per_bin[bin_number]) for bin_number in bin_numbers)
        assert self.n == total_samples, "Some samples dissapered from the bins"

        required_covered = int(np.ceil((1.0 - float(self.alpha)) * total_samples))

        # One record corresponds to one binary MILP variable.
        # (bin_number, covered_count, set_size_sum, cand_cal_val)
        variable_records = []

        for bin_number in bin_numbers:
            stats = utils.function_utils.UtilsGeneral.read_bin_stat(path=f"{self.path_logs}stats_bin_{bin_number}.txt")

            stats = np.atleast_2d(np.asarray(stats, dtype=np.float64))

            if stats.shape[1] != 3:
                print(stats)
                raise ValueError(f"Stat_bin_{bin_number} has an incorrect amount of columns. Received shape {stats.shape}.")

            n_samples_current_bin = len(dict_samples_idx_per_bin[bin_number])

            for covered_count, sum_ss, calibration_val in stats:
                variable_records.append((bin_number, int(covered_count), float(sum_ss), float(calibration_val),))

        number_variables = len(variable_records)

        # Objective:
        objective = np.asarray([sum_ss for (_, _, sum_ss, _) in variable_records], dtype=np.float64,)

        # Construct the constraint matrix.
        #
        # Rows 0,...,B-1:
        # exactly one candidate must be selected per bin.
        #
        # Final row:
        # total covered samples >= required_covered.
        bin_to_row = {
            bin_number: row_index for row_index, bin_number in enumerate(bin_numbers)
            }

        matrix_rows = []
        matrix_columns = []
        matrix_values = []

        coverage_row = amount_bins

        for variable_index, record in enumerate(variable_records):
            bin_number, covered_count, _, _ = record

            # Coefficient for the exactly-one constraint
            matrix_rows.append(bin_to_row[bin_number])
            matrix_columns.append(variable_index)
            matrix_values.append(1.0)

            # Coefficient for the global coverage constraint
            matrix_rows.append(coverage_row)
            matrix_columns.append(variable_index)
            matrix_values.append(float(covered_count))

        constraint_matrix = coo_matrix(
            (matrix_values, (matrix_rows, matrix_columns),), 
            shape=(amount_bins + 1, number_variables),
            dtype=np.float64,
        ).tocsc()

        lower_bounds = np.concatenate([np.ones(amount_bins), np.asarray([required_covered]),])
        upper_bounds = np.concatenate([np.ones(amount_bins), np.asarray([np.inf]),])
        constraints = LinearConstraint(constraint_matrix, lb=lower_bounds, ub=upper_bounds,)

        # All variables are binary:
        # x_{b,j} ∈ {0,1}
        variable_bounds = Bounds(lb=np.zeros(number_variables), ub=np.ones(number_variables),)

        integrality = np.ones(number_variables, dtype=np.int64,)

        maximum_possible_coverage = 0

        print(f"Total samples: {total_samples}, required covered: {required_covered}")

        for bin_number in bin_numbers:
            records = [record for record in variable_records if record[0] == bin_number]

            n_samples_current_bin = len(dict_samples_idx_per_bin[bin_number])
            maximum_bin_covered = max(record[1] for record in records)

            maximum_possible_coverage += maximum_bin_covered

            print(f"Bin {bin_number}: n={n_samples_current_bin}, candidates={len(records)-1}, max_covered={maximum_bin_covered}")

        print("Maximum possible covered:", maximum_possible_coverage,)

        if maximum_possible_coverage < required_covered:
            raise RuntimeError(
                f"The generated statistics cannot satisfy coverage: "
                f"maximum={maximum_possible_coverage}, "
                f"required={required_covered}."
            )

        result = milp(
            c=objective,
            integrality=integrality,
            bounds=variable_bounds,
            constraints=constraints,
            options={
                "presolve": True,
                "mip_rel_gap": 0.0,
                "disp": False,
            },
        )

        if not result.success:
            raise RuntimeError(f"MILP optimization failed: {result.message}")

        selected_stats_per_bin = {}

        best_cal_value_per_bin = {}
        selected_covered_count = 0
        selected_set_size_sum = 0.0

        for bin_number in bin_numbers:
            candidate_indices = [
                variable_index
                for variable_index, record in enumerate(variable_records)
                if record[0] == bin_number
            ]

            # Protect against numerical values such as 0.999999.
            selected_index = max(
                candidate_indices,
                key=lambda index: result.x[index],
            )

            if result.x[selected_index] < 0.5:
                raise RuntimeError(f"No threshold selected for bin {bin_number}.")

            (
                _,
                covered_count,
                set_size_sum,
                calibration_val,
            ) = variable_records[selected_index]
            
            n_bin = len(dict_samples_idx_per_bin[bin_number])
            
            best_cal_value_per_bin[bin_number] = calibration_val
            selected_covered_count += covered_count
            selected_set_size_sum += set_size_sum

            selected_stats_per_bin[bin_number] = {
                "n_samples": n_bin,
                "covered_count": covered_count,
                "coverage": covered_count / max(n_bin, 1),
                "set_size_sum": set_size_sum,
                "average_set_size": set_size_sum / max(n_bin, 1),
                "threshold": calibration_val,
            }

        best_cov = selected_covered_count / total_samples
        best_avg_ss = selected_set_size_sum / total_samples

        self.best_cov = float(best_cov)
        self.best_avg_ss = float(best_avg_ss)
        self.best_precal_value_per_bin = best_cal_value_per_bin
        self.milp_result = result

        print("[Result Parallel Calib]")
        print("self.selected_covered_count: ", selected_covered_count)
        print("self.best_cov: ", self.best_cov)
        print("self.selected_set_size_sum: ", selected_set_size_sum)
        print("self.best_avg_ss: ", self.best_avg_ss)
        print("self.best_precal_value_per_bin: ", self.best_precal_value_per_bin)
        print("*"*20)
        print("self.milp_result: ", self.milp_result)
        print("*"*20)
        
        print("\n[MILP selected statistics per bin]")

        for bin_number, stats in selected_stats_per_bin.items():
            print(
                f"Bin {bin_number}: "
                f"n={stats['n_samples']}, "
                f"covered={stats['covered_count']}, "
                f"coverage={stats['coverage']:.6f}, "
                f"set_size_sum={stats['set_size_sum']:.0f}, "
                f"avg_set_size={stats['average_set_size']:.6f}, "
                f"threshold={stats['threshold']:.10f}"
            )

        # self.precalibration_stats_per_bin = selected_stats_per_bin


        return best_cal_value_per_bin


    def _third_step_parallel_calibration_me(self, dict_samples_idx_per_bin, ):
        from itertools import product

        bin_numbers = sorted(dict_samples_idx_per_bin)

        dict_all_stats = {}
        bin_sizes = {}

        for bin_number in bin_numbers:
            stats = utils.function_utils.UtilsGeneral.read_bin_stat(
                path=f"{self.path_logs}stats_bin_{bin_number}.txt"
            )

            stats = np.atleast_2d(
                np.asarray(stats, dtype=np.float64)
            )

            # Columns: coverage, average set size, threshold
            stats = stats[np.argsort(stats[:, 2])]

            dict_all_stats[bin_number] = stats
            bin_sizes[bin_number] = len(dict_samples_idx_per_bin[bin_number])

        total_samples = sum(bin_sizes.values())
        required_covered = int(np.ceil((1.0 - float(self.alpha)) * total_samples))

        derived_bin = bin_numbers[-1]
        searched_bins = bin_numbers[:-1]

        derived_stats = dict_all_stats[derived_bin]
        derived_size = bin_sizes[derived_bin]

        derived_covered_counts = np.rint(derived_stats[:, 0] * derived_size).astype(np.int64)

        best_avg_ss = np.inf
        best_cov = 0.0
        best_cal_value_per_bin = None

        candidate_rows = [dict_all_stats[bin_number] for bin_number in searched_bins]

        # product over B-1 bins
        for selected_rows in product(*candidate_rows):

            covered_so_far = 0
            set_size_sum_so_far = 0.0
            thresholds = {}

            for bin_number, row in zip(searched_bins, selected_rows,):
                cov, avg_ss, tau = row
                n_bin = bin_sizes[bin_number]

                covered_so_far += int(round(cov * n_bin))
                set_size_sum_so_far += avg_ss * n_bin
                thresholds[bin_number] = float(tau)

            required_from_last = required_covered - covered_so_far

            feasible_indices = np.flatnonzero(derived_covered_counts >= required_from_last)

            if feasible_indices.size == 0:
                continue

            # Among feasible thresholds, use the one with smallest set size
            local_index = np.argmin(derived_stats[feasible_indices, 1])
            selected_index = feasible_indices[local_index]

            last_cov, last_avg_ss, last_tau = derived_stats[selected_index]

            total_covered = covered_so_far + derived_covered_counts[selected_index]
            total_set_size_sum = set_size_sum_so_far + last_avg_ss * derived_size

            global_cov = total_covered / total_samples
            global_avg_ss = total_set_size_sum / total_samples

            if global_avg_ss < best_avg_ss:
                best_avg_ss = float(global_avg_ss)
                best_cov = float(global_cov)

                best_cal_value_per_bin = dict(thresholds)
                best_cal_value_per_bin[derived_bin] = float(last_tau)

        if best_cal_value_per_bin is None:
            raise RuntimeError("No threshold combination satisfies the required global coverage.")

        self.best_cov = best_cov
        self.best_avg_ss = best_avg_ss
        self.best_precal_value_per_bin = best_cal_value_per_bin

        return best_cal_value_per_bin

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def precalibrate(self):
        """Fit and freeze the range partition; do not calibrate any CP here."""

        os.makedirs(self.path_logs, exist_ok=True)
        self.theta = self._leading_theta()
        
        assert self.at_least_one_precalibration == False, f"at_least_one_precalibration={self.at_least_one_precalibration}"

        P_views = self._to_probs(self.p_views_pre)
        p_ref = self._predictive_distribution(self.p_views_pre)
        y_precal = self.targets_pre.tolist()

        feature_stats = self._symmetric_invariant_intervals_from_prob_views(P_views, p_ref)

        sets_of_sample_idx_and_robustness = feature_stats["sample_idx_and_robustness"]
        sample_count_per_robustness = feature_stats["sample_count_per_robustness"]

        if self.write_debug_files:
            with open(f"{self.path_logs}items_and_robustness.txt", "w") as f:
                f.writelines(f"{idx}, {robustness}\n" for idx, robustness in sets_of_sample_idx_and_robustness)

        self.selected_rotation_ranges = set(sample_count_per_robustness.keys())

        dict_of_samples_and_idx_and_trueclass_per_robustness = {} # Only for statistic purpose
        dict_of_samples_and_idx_per_robustness = {}

        # arbitrary_splitting_ranges = [(0, self.max_rotation-2), (self.max_rotation-1, self.max_rotation)]
        # arbitrary_splitting_ranges = [(0,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10),
        #                               (11,20), (21, 80), (81, 160), (161, 179)]
        arbitrary_splitting_ranges = []
        
        _step = 1
        for i in range(0, self.max_rotation+1, _step):
            arbitrary_splitting_ranges.append((i, min(i+_step-1, self.max_rotation)))
            
        print("Splitting ranges:")
        print(arbitrary_splitting_ranges)
        print()
        print("shape of P: ", P_views.shape)
        print()
        
        list_robutsness = []

        for sample_idx, robutness in sets_of_sample_idx_and_robustness:
            dict_of_samples_and_idx_and_trueclass_per_robustness.setdefault(robutness, set()).add((sample_idx , y_precal[sample_idx], tuple(p_ref[sample_idx].tolist())))
            list_robutsness.append((robutness))
            dict_of_samples_and_idx_per_robustness.setdefault(int(robutness), []).append(sample_idx)

        
        if self.write_debug_files:
            with open(f"{self.path_logs}dict_of_samples_and_idx_and_trueclass_per_robustness.txt", "w") as f:
                for key in range(self.max_rotation):
                    if key in dict_of_samples_and_idx_and_trueclass_per_robustness.keys():
                        set_items = dict_of_samples_and_idx_and_trueclass_per_robustness.get(key)
                        f.write(f"grade: {key}, counts: {len(set_items)}\n")
                        for item_ in set_items:
                            f.write(f"  {item_}\n")
                    else:
                        f.write(f"grade: {key}, counts: 0\n")
                    f.write(f"\n")
                
                    
        del dict_of_samples_and_idx_and_trueclass_per_robustness

        dict_samples_idx_per_bin = self._first_step_create_bins_according_to_splitting_range(dict_of_samples_and_idx_per_robustness, arbitrary_splitting_ranges)

        # dict_samples_idx_per_bin = self._first_step_create_bins_according_to_min_sample_per_bin(dict_of_samples_and_idx_per_robustness)

        # utils.function_utils.UtilsGeneral.write_dict_to_file(dict_samples_idx_per_bin, f"{self.path_logs}dict_samples_idx_per_bin.txt")
        
        if self.write_debug_files:
            with open(f"{self.path_logs}dict_samples_idx_per_bin.txt", "w") as f:
                for key in range(self.max_rotation):
                    if key in dict_samples_idx_per_bin.keys():
                        list_idx = list(dict_samples_idx_per_bin.get(key))
                        f.write(f"grade: {key}, counts: {len(list_idx)}, items: {list_idx}\n")
                    else:
                        f.write(f"grade: {key}, counts: 0, items: []\n")

        all_assigned_indices = []

        for bin_number, this_indices in dict_samples_idx_per_bin.items():
            all_assigned_indices.extend(this_indices)

        if len(all_assigned_indices) != self.n or len(np.unique(all_assigned_indices)) != self.n:
            print(f"size_all_assigned_samples: {len(all_assigned_indices)} vs amount_samples: {self.n} vs unique_sample_idx: {len(np.unique(all_assigned_indices))}")
            print(np.sort(all_assigned_indices))
            raise RuntimeError("Some precalibration samples are missing ")

        for bin_number, bin_member_idx in dict_samples_idx_per_bin.items():
            self._second_step_collect_precalibration_data_per_bin(bin_number=bin_number, bin_member_indices=bin_member_idx, p_ref=p_ref, y_precal=y_precal)
        

        print("Second step done.", flush=True)

        self.best_precal_value_per_bin = self._third_step_parallel_calibration(dict_samples_idx_per_bin)

        
        # Here I validate the coverage and precal values got from the optimization step
        # I do it on the same data.
        set_by_bin = {}
        for bin_number in self.best_precal_value_per_bin.keys():
            # P_views
            current_set = []
            indices_bin = list(dict_samples_idx_per_bin.get(bin_number))
            for idx in indices_bin:
                bin_sample = P_views[:, idx, :]
                z = self.compute_prob_dist(bin_sample[0], list(bin_sample[1:]))
                u = float(self._rng.uniform()) if (self.scoring_function == "aps" and self.aps_randomized) else 0.0
                sc = self._all_scores(z, score=self.scoring_function, u=u)
                result = self._set_from_scores_predict_only(sc, self.best_precal_value_per_bin.get(bin_number), at_least_one=self.at_least_one_precalibration)
                current_set.append((idx, sc, result))
            set_by_bin[bin_number] = current_set
            
        if self.write_debug_files:
            with open(f"{self.path_logs}precal_prediction.log", "w") as f:
                for key, value in set_by_bin.items():
                    f.write(f"grade: {key}, Precal_val: {float(self.best_precal_value_per_bin.get(key))}\n")
                    for item in value:
                        f.write(f"idx: {item[0]}, score: {item[1]}, set: {item[2][0]}\n")
                    f.write(f"\n")


        self.is_calibrated = False
        self.qhat = None

        if self.verbose:
            observed = len(self.selected_rotation_ranges)
            print(
                f"[QCTHR] precalibration done: {observed} observed ranges, "
                f"computed min amount of samples for alpha={self.alpha}: ",
                flush=True,
            )

        if self.write_debug_files:    
            with open(f'{self.path_logs}prediction_stats.log', 'w') as f:
                f.write("Ready to write in.\n")


    def calibrate(self, calibration_data, alpha=None):
        """
        Calibrate an additional global offset on top of the independently
        precalibrated THR thresholds.
        """

        self._check_precalibration_ready()


        # with open(f'{self.path_logs}Report_calibration.log', 'w') as f:
        #     f.write("\n")

        a = self.alpha if alpha is None else float(alpha)
        if not (0.0 < a < 1.0):
            raise ValueError("alpha must be in (0,1).")

        V, y = self._parse(calibration_data)
        V = self._to_probs(V)

        y = np.asarray(y, dtype=np.int64)

        if len(y) == 0:
            raise ValueError("Calibration data is empty.")

        if np.any(y < 0) or np.any(y >= self.K):
            raise ValueError("Calibration targets contain invalid class indices.")

        print("*"*20)
        print(f"\n\n[STATS IN DER CALIBRATION]")
        
        print(f"Shape data: {V.shape}")


        sample_idx_to_cp_ids = self._thr_scores_and_ranges_from_views(V)
        
        dict_of_samples_and_idx_and_trueclass_per_robustness = {}
        dict_of_samples_and_idx_per_robustness = {}
        
        for sample_idx, robutness in sample_idx_to_cp_ids.items():
            dict_of_samples_and_idx_and_trueclass_per_robustness.setdefault(robutness, set()).add((sample_idx , int(y[sample_idx]), tuple(V[0][sample_idx].tolist())))
            dict_of_samples_and_idx_per_robustness.setdefault(robutness, []).append(sample_idx)

        if self.write_debug_files:
            with open(f"{self.path_logs}calibration_dict_of_samples_and_idx_and_trueclass_per_robustness.txt", "w") as f:
                for key in range(self.max_rotation):
                    if key in dict_of_samples_and_idx_and_trueclass_per_robustness.keys():
                        set_items = dict_of_samples_and_idx_and_trueclass_per_robustness.get(key)
                        f.write(f"grade: {key}, counts: {len(set_items)}\n")
                        for item_ in set_items:
                            f.write(f"  {item_}\n")
                    else:
                        f.write(f"grade: {key}, counts: 0\n")
                    f.write(f"\n")
                
                    
        del dict_of_samples_and_idx_and_trueclass_per_robustness
        
        
        # S = self._all_scores(self._pool_leading(V))

        true_scores = self._get_calibration_true_scores(V, y)

        # np.savetxt(f"{self.path_logs}calibration_scores.txt", V[0])

        # combined_arrays = np.hstack([true_scores[:, None], y[:, None], np.argmax(V[0], axis=1)[:, None], 1.0- np.max(V[0], axis=1)[:, None], (1.0-V[0][np.arange(len(y)), y])[:, None], np.max(V[0], axis=1)[:, None],])

        # np.savetxt(f"{self.path_logs}labels_scores.txt", combined_arrays, header='true_scores            | y                      | np.argmax(V[0], axis=1)| (1.0-np.max(V[0], axis=1))')

        self.calibration_range_counts = {}
        self.qhat_by_range = {}
        report = {}

        n = len(y)

        assert len(sample_idx_to_cp_ids) == n, "Missing some samples in calibration"

        per_sample_precal_value = np.empty(n, dtype=float)
        sample_cp_cal_value = set()

        print("[Stats -- Calibration]")
        
        calibration_dict_per_sample_per_all_calib = []
        for sample_idx in range(n):
            cp_id = int(sample_idx_to_cp_ids.get(sample_idx,))            
            precal_value = float(self.best_precal_value_per_bin.get(cp_id,))
            
            assert precal_value != None and cp_id != None, "Incorrect precal or cp_id found"
            
            calibration_dict_per_sample_per_all_calib.append((int(sample_idx), int(cp_id), float(precal_value)))
            
            per_sample_precal_value[sample_idx] = precal_value
            sample_cp_cal_value.add((sample_idx, cp_id, precal_value))
            self.calibration_range_counts[cp_id] = self.calibration_range_counts.get(cp_id, 0) + 1
            
        
        if self.write_debug_files:
            with open(f"{self.path_logs}calibation_dict_per_sample_precal_val.log", "w") as f:
                last_cp = 0
                for sample_idx, cp_id, precal_value in sorted(calibration_dict_per_sample_per_all_calib, key=lambda x: x[1]):
                    f.write(f"idx:{sample_idx}, cpid: {cp_id}, precalval: {precal_value} \n")
                    if last_cp != cp_id:
                        f.write(f"\n")
                        last_cp = cp_id
                        
#----------------------------------------------------------
        assert len(true_scores.shape) == 1 and true_scores.shape[0] == n, "Incorrect shape of calibration score"
        true_scores = np.asarray(true_scores, dtype=np.float64, )

        if np.any(per_sample_precal_value < 0.0) or np.any(per_sample_precal_value > 1.0):
            raise ValueError("Precalibration thresholds must be in [0,1].")

        k = int(np.ceil((n + 1) * (1.0 - a))) 

        def _scaled_thresholds(base_values, scaling_value):
            """
            scaling_value = -1 -> every threshold becomes 0
            scaling_value =  0 -> original precalibration thresholds
            scaling_value =  1 -> every threshold becomes 1
            """
            base_values = np.asarray(base_values, dtype=np.float64,)

            if scaling_value >= 0.0:
                return base_values + scaling_value * (1.0 - base_values)

            return base_values + scaling_value * base_values

        def num_covered(scaling_value):
            sample_thresholds = _scaled_thresholds(per_sample_precal_value, scaling_value, )
            assert true_scores.shape == sample_thresholds.shape, "Incorrect shape of sample_thresholds"

            return int(np.count_nonzero(true_scores <= sample_thresholds))
        
        coverage_without_scaling = num_covered(0.0) / n
        print("Calibration coverage at scaling=0:", coverage_without_scaling)

        # ------------------------------------------------------------
        # Handle the finite-sample full-set case
        # ------------------------------------------------------------

        if k > n:
            # return all classes and for THR (1-p), lambda=1 gives q_b=1 for every bin, which is the full prediction set.
            scaling_value = 1.0
        else:
            lower = -1.0
            upper = 1.0

            if num_covered(upper) < k:
                raise RuntimeError("Scaling value 1 should have covered all cases, error somewhere.")

            bisection_tolerance = 1e-12
            maximum_iterations = 200

            for _ in range(maximum_iterations):
                if upper - lower <= bisection_tolerance:
                    break

                midpoint = lower + 0.5 * (upper - lower)
                if num_covered(midpoint) >= k:
                    # Feasible: search for a smaller scaling value.
                    upper = midpoint
                else:
                    # Infeasible: increase the scaling value.
                    lower = midpoint

            # Use the feasible endpoint, not the midpoint or lower bound.
            scaling_value = float(upper)

        # scaling_value = 0.0
        final_covered = num_covered(scaling_value)

        if final_covered < k:
            raise RuntimeError("Bisection returned an infeasible scaling value.")

        self.qhat_by_range = {}

        for cp_id, precal_value in self.best_precal_value_per_bin.items():
            precal_value = float(precal_value)

            if scaling_value >= 0.0:
                final_value = precal_value + scaling_value * (1.0 - precal_value)
            else:
                final_value = precal_value + scaling_value * precal_value

            if not 0.0 <= final_value <= 1.0:
                raise ValueError(f"Scaled threshold for bin {cp_id} is outside [0,1]: {final_value}.")

            self.qhat_by_range[cp_id] = float(final_value)
            
        for sample_idx in range(len(calibration_dict_per_sample_per_all_calib)):
            # (sample_idx, cp_id, precalVal)
            _item = calibration_dict_per_sample_per_all_calib[sample_idx]
            calibration_dict_per_sample_per_all_calib[sample_idx] += (float(self.qhat_by_range[int(_item[1])]),)
        
        
        stats_calib_by_cp = {}
        # for sample_idx, cp_id, precalVal, finalCal in calibration_dict_per_sample_per_all_calib:
        #     f.write(f"idx:{sample_idx}, cpid: {cp_id}, precalval: {precalVal}, calVal: {finalCal} \n")
        #     stats_calib_by_cp[cp_id]
        if self.write_debug_files:
            with open(f"{self.path_logs}calibration_dict_per_sample_per_all_calib.log", "w") as f:
                last_cp = 0
                for sample_idx, cp_id, precalVal, finalCal in sorted(calibration_dict_per_sample_per_all_calib, key=lambda x: x[1]):
                    f.write(f"idx:{sample_idx}, cpid: {cp_id}, precalval: {precalVal}, calVal: {finalCal} \n")
                    if last_cp != cp_id:
                        f.write(f"\n")
                        last_cp = cp_id

        achieved_calibration_coverage = final_covered / n

        self.scaling_value = float(scaling_value)

        report["alpha"] = a
        report["n_calibration"] = n
        report["required_covered_samples"] = k
        report["scaling_value"] = self.scaling_value
        report["calibration_coverage"] = achieved_calibration_coverage
        report["precalibration_thresholds"] = {
            int(cp_id): float(value) for cp_id, value in self.best_precal_value_per_bin.items()
        }
        report["final_thresholds"] = {
            int(cp_id): float(value) for cp_id, value in self.qhat_by_range.items()
        }

        self.calibration_report = report
        self.is_calibrated = True
        self.qhat = self.qhat_by_range
        
        print(report)

        if self.write_debug_files:
            with open(f"{self.path_logs}prediction_stats.log", "w",) as f:
                # f.write(json.dumps(report, indent=2))
                f.write("\n")

        return self.qhat_by_range
    
    def predict(self, predictedLogitVector, calibrationValue=None):
        self._check_calibration_ready()

        V = np.asarray(predictedLogitVector, dtype=np.float64)
        if V.ndim != 2:
            raise ValueError(f"Expected one sample with shape (M,K), got {V.shape}.")

        sample_range = self._thr_scores_and_ranges_from_views(V)

        z = self.compute_prob_dist(predictedLogitVector[0], list(predictedLogitVector[1:]))
        u = float(self._rng.uniform()) if (self.scoring_function == "aps" and self.aps_randomized) else 0.0
        scores = self._all_scores(z, score=self.scoring_function, u=u)
        
        if calibrationValue is None:
            tau = self.qhat_by_range.get(sample_range.get(0))
        else:
            raise NotImplementedError(f"Invalid calibration value.")

        tau = float(tau)
        self.last_range = sample_range
        self.last_qhat = tau
        self.last_scores = scores

        result = self._set_from_scores_predict_only(scores, tau)
        
        if self.write_debug_files:
            with open(f'{self.path_logs}prediction_stats.log', 'a') as f:
                f.write(f"{scores}, {result}, {sample_range.get(0)}, {tau}\n\n")


        return result

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
            "stat": {
                str(rotation_range): [
                    np.asarray(sample).tolist()
                    for sample in samples
                ]
                for rotation_range, samples in self.stat.items()
            },
            "n_ranges_total": self.n_invariant_rotation_range,
            "calibration_range_counts": {
                str(c): int(n) for c, n in self.calibration_range_counts.items()
            },
            "qhat_by_range": {
                str(c): self._json_float(q)
                for c, q in self.qhat_by_range.items()
            },
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
