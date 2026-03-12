import os
from logging import raiseExceptions

import numpy as np

from approaches.optimizeWithSolver import OptimizeWithSolver


class WeightedRank2DPredictor:

    def __init__(self, precalibration_data, alpha, path_logs, n_classes:int = 10, seed: int = 2026,
                 min_samples_per_pattern: int = 1, verbose: bool = True):
        self.n_classes = n_classes
        self.seed = seed
        self.dict_pattern_and_w = {}
        self.min_samples_per_pattern = int(min_samples_per_pattern)
        self.alpha = alpha
        self.verbose = verbose
        self.path_logs = path_logs


        tmp_probDistributions, tmp_targets = zip(*precalibration_data)
        self.allProbDist = np.asarray(list(zip(*tmp_probDistributions)))
        self.allTargets = np.asarray(tmp_targets)

        self.amount_samples = self.allTargets.shape[0]

        del tmp_probDistributions, tmp_targets
        self.n_augmentations = len(self.allProbDist[0])-1

        assert self.allProbDist.shape[0] == 2, "At the moment the approach works only with one augmentation"

        # A pattern is a vector that contains the ordered position of the concatened probability distribution for both probDist
        # e.g.: [7 8 3 0 9 1 6 4 2 5 7 9 3 0 8 1 6 4 2 5] -> The highest probDist for Orig correspond to the 4rd class (1-based index)
        #       [0 9 3 6 8 4 1 2 7 5 0 9 3 6 8 4 1 2 7 5] -> The highest probDist for Orig correspond to the 1st class (1-based index)
        self.extracted_patterns = self.create_patterns(self.allProbDist[0], self.allProbDist[1])

        print("Initalization done.")



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

    def precalibrate(self):
        patterns = self.extracted_patterns
        unique_pats, inv, counts = np.unique(
            patterns,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )

        save_path = f"{self.path_logs}precalibration_result_{self.seed}_alpha{self.alpha}.npz"
        kept_pat_ids = [
            pat_id
            for pat_id, n_z in enumerate(counts.tolist())
            if n_z >= self.min_samples_per_pattern
        ]

        # Reset stored mappings to avoid keeping stale values from previous calls
        self.dict_pattern_and_w = {}
        self.dict_pattern_to_w_vectors = {}
        self.pattern_counts = {}

        # Global pattern index:
        # 0 = fallback group containing all low-support patterns
        c_idx_global = np.zeros((self.amount_samples,), dtype=np.int64)
        current_pattern_id = 1
        pat_to_global = {}

        for pat_id in kept_pat_ids:
            pat_to_global[pat_id] = current_pattern_id
            c_idx_global[inv == pat_id] = current_pattern_id
            current_pattern_id += 1

        if os.path.exists(save_path):
            print("Loading weights...")
            data = np.load(save_path, allow_pickle=True)

            best_result = {
                "status": data["status"].item(),
                "objective": float(data["objective"]),
                "u": data["u"],
                "z": data["z"],
                "coverage_proxy": float(data["coverage_proxy"]),
                "avg_set_size_proxy": float(data["avg_set_size_proxy"]),
                "tau": float(data["tau"]) if data["tau"] is not None else None,
                "M": float(data["M"]),
                "runtime": float(data["runtime"]),
                "mip_gap": float(data["mip_gap"]) if data["mip_gap"] is not None else None,
            }

        else:

            optimization_tool = OptimizeWithSolver()
            best_result = optimization_tool.optimize_w_for_all_patterns_simultaneously(
                p_orig=self.allProbDist[0],
                p_aug=self.allProbDist[1],
                y=self.allTargets,
                index_by_pattern=c_idx_global,
                alpha=self.alpha,
            )

        if best_result is None or best_result.get("tau") is None:
            self.u_global = 1.0
            self.tau_from_optimization = None
            if self.verbose:
                print("[precalibrate] optimization returned no incumbent solution.")
            print("Precalibration done")
            return

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

        # if self.verbose:
        #     print("*" * 10)
        #     print(f"[precalibrate] fallback u_global = {self.u_global}")
        #     for key, value in self.dict_pattern_and_w.items():
        #         print(f"{key} -> {value}")
        #     print("*" * 10)
        #     n_kept = len(self.dict_pattern_and_w)
        #     n_total = len(unique_pats)
        #     print(
        #         f"[precalibrate] stored {n_kept}/{n_total} patterns "
        #         f"(support >= {self.min_samples_per_pattern})."
        #     )
        #     print(f"[precalibrate] tau_from_optimization = {self.tau_from_optimization}")
        #
        #     print("*"*20)
        #     print(best_result)
        #     print("*"*20)

        if self.verbose:
            with open(f"{self.path_logs}precalibration_log_{self.seed}_alpha{self.alpha}.txt", "w") as f:
                f.write("*" * 10 + "\n")
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

                f.write("*" * 20 + "\n")
                f.write("[OPTIMIZER RESULTS]\n")

                if best_result is None:
                    f.write("None\n")
                else:
                    for key, value in best_result.items():
                        f.write(f"{key}:\n")

                        if isinstance(value, np.ndarray):
                            f.write(
                                np.array2string(
                                    value,
                                    threshold=np.inf,
                                    max_line_width=np.inf,
                                    separator=", "
                                )
                                + "\n"
                            )
                        else:
                            f.write(f"{value}\n")

                        f.write("\n")

                f.write("*" * 20 + "\n")



            np.savez(
                save_path,
                status=best_result["status"],
                objective=best_result["objective"],
                u=best_result["u"],
                z=best_result["z"],
                coverage_proxy=best_result["coverage_proxy"],
                avg_set_size_proxy=best_result["avg_set_size_proxy"],
                tau=best_result["tau"],
                M=best_result["M"],
                runtime=best_result["runtime"],
                mip_gap=best_result["mip_gap"],
            )

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

        w_eff = self.dict_pattern_and_w.get(pat_key, 1.0)

        if w_eff == 0:
            return probDistOrig

        weighted_prob = w_eff * probDistOrig + (1 - w_eff) * probDistAug

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
