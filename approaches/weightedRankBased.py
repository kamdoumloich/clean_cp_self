import os
import subprocess

import numpy as np

from approaches.optimizeWithSolver import OptimizeWithSolver


class WeightedRank2DPredictor:

    def __init__(self, precalibration_data, alpha, path_logs, n_classes:int = 10, seed: int = 2026,
                 min_samples_per_pattern: int = 1, verbose: bool = True,
                 precalibration_backend: str = "precalibrator"):
        if precalibration_backend not in {"precalibrator", "solver"}:
            raise ValueError(
                "precalibration_backend must be 'precalibrator' or 'solver'."
            )

        self.n_classes = n_classes
        self.seed = seed
        self.dict_pattern_and_w = {}
        self.min_samples_per_pattern = int(min_samples_per_pattern)
        self.alpha = alpha
        self.verbose = verbose
        self.path_logs = path_logs
        self.precalibration_backend = precalibration_backend
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.u_global = 0.0
        self.tau_from_optimization = None


        tmp_probDistributions, tmp_targets = zip(*precalibration_data)
        self.allProbDist = np.asarray(list(zip(*tmp_probDistributions)))
        self.allTargets = np.asarray(tmp_targets)

        self.amount_samples = self.allTargets.shape[0]

        del tmp_probDistributions, tmp_targets
        self.n_augmentations = self.allProbDist.shape[0] - 1

        assert self.allProbDist.shape[0] == 2, "At the moment the approach works only with one augmentation"

        # A pattern is a vector that contains the ordered position of the concatened probability distribution for both probDist
        # e.g.: [7 8 3 0 9 1 6 4 2 5 7 9 3 0 8 1 6 4 2 5] -> The highest probDist for Orig correspond to the 4rd class (1-based index)
        #       [0 9 3 6 8 4 1 2 7 5 0 9 3 6 8 4 1 2 7 5] -> The highest probDist for Orig correspond to the 1st class (1-based index)
        self.extracted_patterns = self.create_patterns(self.allProbDist[0], self.allProbDist[1])

        print("Initalization of WeightedRank2DPredictor done.", flush=True)


    def create_patterns(self, probDist_orig, probDist_aug):
        prob_orig = np.stack(probDist_orig, axis=0)  # (N, K)
        prob_aug = np.stack(probDist_aug, axis=0)  # (N, K)

        N, K = prob_orig.shape

        # 2) Argsort along classes (axis=1), stable to match loop
        order_orig = np.argsort(-prob_orig, axis=1, kind="stable")  # (N, K)
        order_aug = np.argsort(-prob_aug, axis=1, kind="stable")  # (N, K)

        # 3) Invert permutations to get ranks per class
        ranks_per_classes_orig = np.empty_like(order_orig)
        ranks_per_classes_aug = np.empty_like(order_aug)

        row_idx = np.arange(N)[:, None]  # shape (N, 1)
        col_ranks = np.arange(K)  # shape (K,)

        ranks_per_classes_orig[row_idx, order_orig] = col_ranks
        ranks_per_classes_aug[row_idx, order_aug] = col_ranks

        return np.concatenate([ranks_per_classes_orig, ranks_per_classes_aug], axis=1).astype(np.int64)

        # return np.hstack(tup=(ranks_per_classes_orig, ranks_per_classes_aug), dtype=int)

    # def compute_candidate_calibration_value_for_orig(self, p_orig, ):

    def _get_precalibrator_paths(self):
        base_name = f"precalibrator_{self.seed}_alpha{self.alpha}_msp{self.min_samples_per_pattern}"
        return {
            "step1_input": os.path.join(self.path_logs, f"{base_name}_step1_input.txt"),
            "step1_output": os.path.join(self.path_logs, f"{base_name}_step1_output.txt"),
            "step2_output": os.path.join(self.path_logs, f"{base_name}_step2_output.txt"),
            "step3_output": os.path.join(self.path_logs, f"{base_name}_step3_output.txt"),
        }

    def _build_precalibrator_group(self, unique_pats, inv, counts, kept_pat_ids):
        groups = []
        fallback_indices = np.flatnonzero(counts[inv] < self.min_samples_per_pattern)
        if fallback_indices.size > 0:
            groups.append(
                {
                    "group_type": "fallback",
                    "sample_indices": fallback_indices,
                }
            )

        for pat_id in kept_pat_ids:
            groups.append(
                {
                    "group_type": tuple(unique_pats[pat_id].tolist()),
                    "sample_indices": np.flatnonzero(inv == pat_id),
                }
            )

        return groups

    def _write_precalibrator_step1_input(self, groups, target_path):
        with open(target_path, "w", encoding="utf-8") as file:
            for exported_pattern_id, group in enumerate(groups):
                sample_indices = group["sample_indices"]
                file.write(
                    f"Case: {group['group_type']}\n"
                    f"##PATTERN DATA {exported_pattern_id} {len(sample_indices)}\n")

                for sample_idx in sample_indices:
                    orig_values = np.asarray(self.allProbDist[0][sample_idx], dtype=np.float64)
                    aug_values = np.asarray(self.allProbDist[1][sample_idx], dtype=np.float64)
                    target_class = int(self.allTargets[sample_idx])

                    line_values = [
                        *orig_values.tolist(),
                        *aug_values.tolist(),
                        target_class,
                    ]
                    file.write(" ".join(str(value) for value in line_values))
                    file.write("\n")

    def _run_precalibrator_pipeline(self, step1_input_path):
        paths = self._get_precalibrator_paths()
        paths["step1_input"] = step1_input_path

        subprocess.run(
            [
                "python3",
                os.path.join("precalibrator", "precalibrator_step1.py"),
                paths["step1_input"],
                paths["step1_output"],
            ],
            cwd=self.project_root,
            check=True,
        )
        subprocess.run(
            [
                "python3",
                os.path.join("precalibrator", "precalibrator_step2.py"),
                paths["step1_output"],
                str(1.0 - self.alpha),
                paths["step2_output"],
            ],
            cwd=self.project_root,
            check=True,
        )
        subprocess.run(
            [
                "python3",
                os.path.join("precalibrator", "precalibrator_step3.py"),
                paths["step1_input"],
                paths["step2_output"],
                paths["step3_output"],
            ],
            cwd=self.project_root,
            check=True,
        )
        return paths

    def _load_precalibrator_step3_results(self, step3_output_path, groups):
        with open(step3_output_path, "r", encoding="utf-8") as file:
            first_line = file.readline().strip().split()
            if len(first_line) != 2 or first_line[0] != "calibration_value":
                raise ValueError(
                    f"Unexpected header in precalibrator output: {first_line}"
                )

            calibration_value_on_precal = float(first_line[1])
            pattern_weights = []
            for line in file:
                parts = line.strip().split()
                if len(parts) != 4:
                    raise ValueError(
                        f"Unexpected line in precalibrator output: {line}"
                    )
                pattern_weights.append(float(parts[1]))

        if len(pattern_weights) != len(groups):
            raise ValueError(
                "Precalibrator output does not match the number of exported groups."
            )

        self.tau_from_optimization = float(1.0 - calibration_value_on_precal)
        self.u_global = 0.0

        for group, w_value in zip(groups, pattern_weights):
            if group["group_type"] == "fallback":
                self.u_global = float(w_value)
                continue

            pat_key = group["group_type"]
            self.dict_pattern_and_w[pat_key] = float(w_value)
            self.dict_pattern_to_w_vectors[pat_key] = float(w_value)
            self.pattern_counts[pat_key] = int(group["sample_indices"].size)

        return calibration_value_on_precal

    def _store_solver_results(self, unique_pats, kept_pat_ids, counts, pat_to_global, best_result):
        self.tau_from_optimization = float(best_result["tau"])
        u_sol = np.asarray(best_result["u"], dtype=np.float64)

        # Group 0 is the fallback group for all patterns below support threshold
        self.u_global = float(u_sol[0])

        for pat_id in kept_pat_ids:
            pat_key = tuple(unique_pats[pat_id].tolist())
            global_id = pat_to_global[pat_id]
            w_value = float(u_sol[global_id])

            self.dict_pattern_and_w[pat_key] = w_value
            self.dict_pattern_to_w_vectors[pat_key] = w_value
            self.pattern_counts[pat_key] = int(counts[pat_id])

    def _write_precalibration_log(self, unique_pats, details):
        if not self.verbose:
            return

        log_path = os.path.join(
            self.path_logs,
            f"precalibration_log_{self.seed}_alpha{self.alpha}.txt",
        )
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("*" * 10 + "\n")
            f.write(f"[precalibrate] backend = {self.precalibration_backend}\n")
            f.write(f"[precalibrate] fallback u_global = {self.u_global}\n")

            for key, value in self.dict_pattern_and_w.items():
                f.write(f"{key} -> {value}\n")

            f.write("*" * 10 + "\n")

            n_kept = len(self.dict_pattern_and_w)
            n_total = len(unique_pats)
            f.write(
                f"[precalibrate] stored {n_kept}/{n_total} patterns "
                f"(support >= {self.min_samples_per_pattern}).\n"
            )
            f.write(
                f"[precalibrate] tau_from_optimization = {self.tau_from_optimization}\n"
            )

            if details is not None:
                f.write("*" * 20 + "\n")
                f.write("[PRECALIBRATION DETAILS]\n")
                for key, value in details.items():
                    f.write(f"{key}: {value}\n")
            f.write("*" * 20 + "\n")

    def precalibrate(self):
        patterns = self.extracted_patterns
        unique_pats, inv, counts = np.unique(
            patterns,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )

        os.makedirs(self.path_logs, exist_ok=True)
        kept_pat_ids = [
            pat_id
            for pat_id, n_z in enumerate(counts.tolist())
            if n_z >= self.min_samples_per_pattern
        ]

        # Reset stored mappings to avoid keeping stale values from previous calls
        self.dict_pattern_and_w = {}
        self.dict_pattern_to_w_vectors = {}
        self.pattern_counts = {}

        precalibrator_groups = self._build_precalibrator_group(
            unique_pats=unique_pats,
            inv=inv,
            counts=counts,
            kept_pat_ids=kept_pat_ids,
        )
        self.precalibrator_step1_input_path = self._get_precalibrator_paths()["step1_input"]
        self._write_precalibrator_step1_input(
            groups=precalibrator_groups,
            target_path=self.precalibrator_step1_input_path,
        )

        if self.precalibration_backend == "precalibrator":
            precalibrator_paths = self._run_precalibrator_pipeline(
                self.precalibrator_step1_input_path
            )
            calibration_value_on_precal = self._load_precalibrator_step3_results(
                step3_output_path=precalibrator_paths["step3_output"],
                groups=precalibrator_groups,
            )
            details = {
                "status": "PRECALIBRATOR_PIPELINE",
                "probability_threshold": calibration_value_on_precal,
                "evaluation_tau": self.tau_from_optimization,
                "step1_input": precalibrator_paths["step1_input"],
                "step1_output": precalibrator_paths["step1_output"],
                "step2_output": precalibrator_paths["step2_output"],
                "step3_output": precalibrator_paths["step3_output"],
            }
        elif self.precalibration_backend == "solver":
            # Global pattern index:
            # 0 = fallback group containing all low-support patterns
            c_idx_global = np.zeros((self.amount_samples,), dtype=np.int64)
            current_pattern_id = 1
            pat_to_global = {}

            for pat_id in kept_pat_ids:
                pat_to_global[pat_id] = current_pattern_id
                c_idx_global[inv == pat_id] = current_pattern_id
                current_pattern_id += 1
            print(
                f"Number of patterns with at least {self.min_samples_per_pattern} samples: "
                f"{len(kept_pat_ids)}/{len(unique_pats)}",
                flush=True,
            )

            print("Solver instance is created", flush=True)
            optimization_tool = OptimizeWithSolver()
            print("Call solving function...", flush=True)
            best_result = optimization_tool.optimize_w_for_all_patterns_simultaneously(
                p_orig=self.allProbDist[0],
                p_aug=self.allProbDist[1],
                y=self.allTargets,
                index_by_pattern=c_idx_global,
                alpha=self.alpha,
            )

            print("Solver ended solving", flush=True)
            if best_result is None or best_result.get("tau") is None:
                self.u_global = 0.0
                self.tau_from_optimization = None
                if self.verbose:
                    print("[precalibrate] optimization returned no incumbent solution.")
                print("Precalibration done")
                return

            self._store_solver_results(
                unique_pats=unique_pats,
                kept_pat_ids=kept_pat_ids,
                counts=counts,
                pat_to_global=pat_to_global,
                best_result=best_result,
            )
            details = best_result
        else:
            raise NotImplementedError("Method not implemented....")

        self._write_precalibration_log(unique_pats, details)

        print("Precalibration done")

    def create_pattern_1d(self, probDist_orig, probDist_aug):
        """
        probDist_orig, probDist_aug: Array-like mit Form (N, K) oder listen, die zu (N, K) stackbar sind.
        Rückgabe: Pattern-Array mit Form (N, 2K), int.
        """

        prob_orig = np.asarray(probDist_orig)  # (N, K)
        prob_aug = np.asarray(probDist_aug)  # (N, K)

        if prob_orig.ndim == 1:
            prob_orig = prob_orig[None, :]
        if prob_aug.ndim == 1:
            prob_aug = prob_aug[None, :]

        if prob_orig.shape != prob_aug.shape:
            raise ValueError(f"Shape mismatch: orig {prob_orig.shape} vs aug {prob_aug.shape}")

        N, K = prob_orig.shape

        # Argsort entlang Klassenachse (pro Zeile), absteigend
        order_orig = np.argsort(-prob_orig, axis=1, kind="stable")  # (N, K)
        order_aug = np.argsort(-prob_aug, axis=1, kind="stable")  # (N, K)

        # Invertiere Permutation -> Rang pro Klasse
        ranks_per_classes_orig = np.empty_like(order_orig, dtype=np.int64)
        ranks_per_classes_aug = np.empty_like(order_aug, dtype=np.int64)

        row_idx = np.arange(N)[:, None]  # (N, 1)
        col_ranks = np.arange(K)[None, :]  # (1, K)

        ranks_per_classes_orig[row_idx, order_orig] = col_ranks
        ranks_per_classes_aug[row_idx, order_aug] = col_ranks

        # Pattern: (N, 2K)
        return np.hstack((ranks_per_classes_orig, ranks_per_classes_aug), dtype=int)[0]

    def compute_prob_dist(self, probDistOrig: np.ndarray, probDistAug: np.ndarray) -> np.ndarray:

        pat = self.create_pattern_1d(probDistOrig, probDistAug)
        pat_key = tuple(pat.tolist())

        w_eff = float(self.dict_pattern_and_w.get(pat_key, self.u_global))

        weighted_prob = (1.0 - w_eff) * probDistOrig + w_eff * probDistAug

        return weighted_prob


    def calibration(self, calibration_data):

        pass


    def predict(self, probabilityDistributions, calibrationValue):
        assert len(probabilityDistributions) == 2
        resultSet = []
        weighted_prob_dist = self.compute_prob_dist(probDistOrig=probabilityDistributions[0],
                                                    probDistAug=probabilityDistributions[1])

        nextOne = None
        maxClassValue = 0.0
        maxClass = None
        for i in range(len(weighted_prob_dist)):
            if weighted_prob_dist[i]>maxClassValue:
                maxClassValue = weighted_prob_dist[i]
                maxClass = i
            if 1.0-weighted_prob_dist[i]<=calibrationValue:
                resultSet.append(i)
            else:
                if nextOne is None:
                    nextOne = 1.0-weighted_prob_dist[i]
                    assert not nextOne==0.0
                else:
                    nextOne = min(nextOne,1.0-weighted_prob_dist[i])
                    assert not nextOne==0.0
        if len(resultSet)==0:
            resultSet = [maxClass]
        return resultSet,nextOne


    def texInfo(self):
        return """Weighted 2D predictor. We extract patterns and use an optimizer to simulatneously find for each pattern which weight allows to reduce the overall set size."""

    def getShortName(self):
        return "weigted-rank-2D"
