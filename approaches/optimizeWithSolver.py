import math
import numpy as np
from pyscipopt import Model, quicksum

from utils.function_utils import UtilsConformalPrediction


class OptimizeWithSolver:

    def __init__(self, time_limit=60, mip_gap=1e-3, n_threads=6, verbose=False, eps=1e-6):
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.verbose = verbose
        self.eps = eps
        self.n_threads = n_threads 

    def optimize_w_for_all_patterns_simultaneously(self, p_orig, p_aug, y, index_by_pattern, alpha, ):
        one_minus_p_orig = np.asarray(1.0 - p_orig, dtype=float)
        one_minus_p_aug = np.asarray(1.0 - p_aug, dtype=float)
        y = np.asarray(y, dtype=int)
        index_by_pattern = np.asarray(index_by_pattern, dtype=int)

        # Dieser Wert ist so, dass prediction_set = (one_minus_probDist_test <= candidate_calValue).
        candidate_calValue = UtilsConformalPrediction.compute_approximative_calibration_value(
            one_minus_probDist=one_minus_p_orig,
            targets=y, alpha=alpha, )

        if one_minus_p_orig.shape != one_minus_p_aug.shape:
            raise ValueError("p_orig and p_aug must have the same shape.")
        if one_minus_p_orig.ndim != 2:
            raise ValueError("p_orig and p_aug must be 2D arrays of shape (n, K).")
        if y.ndim != 1 or len(y) != one_minus_p_orig.shape[0]:
            raise ValueError("y must be a 1D array of length n.")
        if index_by_pattern.ndim != 1 or len(index_by_pattern) != one_minus_p_orig.shape[0]:
            raise ValueError("index_by_pattern must be a 1D array of length n.")

        n, K = one_minus_p_orig.shape

        if np.min(index_by_pattern) < 0:
            raise ValueError("index_by_pattern must be nonnegative.")
        P = int(np.max(index_by_pattern)) + 1

        if np.min(y) < 0 or np.max(y) >= K:
            raise ValueError("y entries must lie in {0, ..., K-1}.")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError("alpha must lie in (0,1).")

        model = Model("pattern_rank_u_global_tau")
        model.setParam("display/verblevel", 4 if self.verbose else 0)
        # model.setParam("limits/time", float(self.time_limit))
        # model.setParam("limits/gap", float(self.mip_gap))
        model.setParam("parallel/maxnthreads", self.n_threads)
       
        ub_u = 1.0
        M_eff = 1.0 + float(self.eps)  # sufficient since scores and tau are in [0, 1]

        u = {}
        z = {}

        # Ich weiss bereits im Voraus, dass der CalWert
        min_tau = max(0, candidate_calValue - 0.1)
        max_tau = min(1.0, candidate_calValue + 0.1)
        tau = model.addVar(name="global_tau", vtype="C", lb=min_tau, ub=max_tau)

        for c in range(P):
            u[c] = model.addVar(name=f"u_{c}", vtype="C", lb=0.0, ub=ub_u)

        for i in range(n):

            c = int(index_by_pattern[i])

            for k in range(K):
                z[i, k] = model.addVar(name=f"z_{i}_{k}", vtype="B")

                score_expr = (
                        float(one_minus_p_orig[i, k]) +
                        (
                                float(one_minus_p_aug[i, k]) - float(one_minus_p_orig[i, k])
                        ) * u[c]
                )
                # z[i,k] = 1  <=>  score_expr <= tau
                model.addCons(
                    score_expr <= tau + M_eff * (1 - z[i, k]),
                    name=f"memb_ub_{i}_{k}",
                )
                model.addCons(
                    score_expr >= tau - float(self.eps) - M_eff * z[i, k],
                    name=f"memb_lb_{i}_{k}",
                )

        min_covered = int(math.ceil((1.0 - float(alpha)) * n))
        model.addCons(
            quicksum(z[i, int(y[i])] for i in range(n)) >= min_covered,
            name="coverage_constraint",
        )

        model.setObjective(
            quicksum(z[i, k] for i in range(n) for k in range(K)),
            sense="minimize",
        )
        # model.writeProblem(filename="optimization_problem.lp", trans=False, genericnames=False)
        # print("Contruction of MILP completed. see: optimization_problem.lp", flush=True)
        print("Contruction of MILP completed.", flush=True)
        model.optimize()
        print("Optimization completed...", flush=True)

        status_name = str(model.getStatus()).upper()
        best_sol = model.getBestSol()
        has_incumbent = best_sol is not None

        u_sol = np.full(P, np.nan, dtype=float)
        z_sol = np.full((n, K), np.nan, dtype=float)
        tau_sol = None

        objective = None
        coverage_proxy = None
        avg_set_size_proxy = None
        final_gap = None

        if has_incumbent:
            for c in range(P):
                u_sol[c] = model.getSolVal(best_sol, u[c])

            for i in range(n):
                for k in range(K):
                    z_sol[i, k] = model.getSolVal(best_sol, z[i, k])

            z_sol = (z_sol >= 0.5).astype(float)
            tau_sol = model.getSolVal(best_sol, tau)

            objective = float(model.getSolObjVal(best_sol))
            coverage_proxy = float(np.sum(z_sol[np.arange(n), y]))
            avg_set_size_proxy = float(np.sum(z_sol) / n)

            try:
                final_gap = float(model.getGap())
            except Exception:
                final_gap = None

        return {
            "status": status_name,
            "objective": objective,
            "u": u_sol,
            "z": z_sol,
            "coverage_proxy": coverage_proxy,
            "avg_set_size_proxy": avg_set_size_proxy,
            "tau": None if tau_sol is None else float(tau_sol),
            "M": float(M_eff),
            "runtime": float(model.getSolvingTime()),
            "mip_gap": final_gap,
        }

    # First trial of optimization of the MILP problem: Solution almost the same as the one of the original MILP formulation
    # def optimize_w_for_all_patterns_simultaneously(
    #         self,
    #         p_orig,
    #         p_aug,
    #         y,
    #         index_by_pattern,
    #         alpha,
    # ):
    #     one_minus_p_orig = np.asarray(1.0 - p_orig, dtype=float)
    #     one_minus_p_aug = np.asarray(1.0 - p_aug, dtype=float)
    #     y = np.asarray(y, dtype=int)
    #     index_by_pattern = np.asarray(index_by_pattern, dtype=int)
    #
    #     candidate_calValue = UtilsConformalPrediction.compute_approximative_calibration_value(
    #         one_minus_probDist=one_minus_p_orig,
    #         targets=y,
    #         alpha=alpha,
    #     )
    #
    #     if one_minus_p_orig.shape != one_minus_p_aug.shape:
    #         raise ValueError("p_orig and p_aug must have the same shape.")
    #     if one_minus_p_orig.ndim != 2:
    #         raise ValueError("p_orig and p_aug must be 2D arrays of shape (n, K).")
    #     if y.ndim != 1 or len(y) != one_minus_p_orig.shape[0]:
    #         raise ValueError("y must be a 1D array of length n.")
    #     if index_by_pattern.ndim != 1 or len(index_by_pattern) != one_minus_p_orig.shape[0]:
    #         raise ValueError("index_by_pattern must be a 1D array of length n.")
    #
    #     n, K = one_minus_p_orig.shape
    #
    #     if np.min(index_by_pattern) < 0:
    #         raise ValueError("index_by_pattern must be nonnegative.")
    #     P = int(np.max(index_by_pattern)) + 1
    #
    #     if np.min(y) < 0 or np.max(y) >= K:
    #         raise ValueError("y entries must lie in {0, ..., K-1}.")
    #     if not (0.0 < float(alpha) < 1.0):
    #         raise ValueError("alpha must lie in (0,1).")
    #
    #     eps = float(self.eps)
    #
    #     tau_lb = max(0.0, float(candidate_calValue) - 0.1)
    #     tau_ub = min(1.0, float(candidate_calValue) + 0.1)
    #
    #     # Exact score interval over u_c in [0,1]:
    #     # score_{ik}(u_c) = (1-u_c)*orig + u_c*aug
    #     score_min = np.minimum(one_minus_p_orig, one_minus_p_aug)
    #     score_max = np.maximum(one_minus_p_orig, one_minus_p_aug)
    #
    #     # Mark true-label positions
    #     is_true = np.zeros((n, K), dtype=bool)
    #     is_true[np.arange(n), y] = True
    #
    #     # Exact fixing rules for your CURRENT MILP semantics
    #     fixed_one = score_max < (tau_lb - eps)
    #
    #     fixed_zero = (
    #             ((~is_true) & (score_min >= (tau_ub - eps)))  # non-true labels: objective will always choose 0
    #             |
    #             (is_true & (score_min > tau_ub))  # true labels: z=1 infeasible
    #     )
    #
    #     needs_bin = ~(fixed_one | fixed_zero)
    #
    #     n_fixed_one = int(np.sum(fixed_one))
    #     n_fixed_zero = int(np.sum(fixed_zero))
    #     n_bin_used = int(np.sum(needs_bin))
    #     n_bin_full = int(n * K)
    #
    #     min_covered = int(math.ceil((1.0 - float(alpha)) * n))
    #     fixed_true_ones = int(np.sum(fixed_one[np.arange(n), y]))
    #     potential_true_binaries = int(np.sum(needs_bin[np.arange(n), y]))
    #
    #     # Safe early infeasibility check
    #     if fixed_true_ones + potential_true_binaries < min_covered:
    #         return {
    #             "status": "INFEASIBLE_BY_PRESOLVE",
    #             "objective": None,
    #             "u": np.full(P, np.nan, dtype=float),
    #             "z": np.full((n, K), np.nan, dtype=float),
    #             "coverage_proxy": None,
    #             "avg_set_size_proxy": None,
    #             "tau": None,
    #             "M": None,
    #             "runtime": 0.0,
    #             "mip_gap": None,
    #             "n_binary_full": n_bin_full,
    #             "n_binary_used": n_bin_used,
    #             "n_binary_fixed_zero": n_fixed_zero,
    #             "n_binary_fixed_one": n_fixed_one,
    #         }
    #
    #     model = Model("pattern_rank_u_global_tau_presolved")
    #     model.setParam("display/verblevel", 4 if self.verbose else 0)
    #     model.setParam("limits/gap", float(self.mip_gap))
    #     model.setParam("parallel/maxnthreads", self.n_threads)
    #
    #     tau = model.addVar(name="global_tau", vtype="C", lb=tau_lb, ub=tau_ub)
    #
    #     u = {}
    #     for c in range(P):
    #         u[c] = model.addVar(name=f"u_{c}", vtype="C", lb=0.0, ub=1.0)
    #
    #     z = {}
    #     max_M_used = 0.0
    #
    #     for i in range(n):
    #         c = int(index_by_pattern[i])
    #
    #         for k in range(K):
    #             if not needs_bin[i, k]:
    #                 continue
    #
    #             z[i, k] = model.addVar(name=f"z_{i}_{k}", vtype="B")
    #
    #             score_expr = (
    #                     float(one_minus_p_orig[i, k]) +
    #                     (float(one_minus_p_aug[i, k]) - float(one_minus_p_orig[i, k])) * u[c]
    #             )
    #
    #             # Tighter per-variable M values
    #             M_up = max(0.0, float(score_max[i, k]) - tau_lb)
    #             M_lo = max(0.0, tau_ub - eps - float(score_min[i, k]))
    #             max_M_used = max(max_M_used, M_up, M_lo)
    #
    #             model.addCons(
    #                 score_expr <= tau + M_up * (1 - z[i, k]),
    #                 name=f"memb_ub_{i}_{k}",
    #             )
    #             model.addCons(
    #                 score_expr >= tau - eps - M_lo * z[i, k],
    #                 name=f"memb_lb_{i}_{k}",
    #             )
    #
    #     remaining_needed = min_covered - fixed_true_ones
    #     if remaining_needed > 0:
    #         coverage_terms = [z[i, int(y[i])] for i in range(n) if needs_bin[i, int(y[i])]]
    #         model.addCons(
    #             quicksum(coverage_terms) >= remaining_needed,
    #             name="coverage_constraint",
    #         )
    #
    #     objective_terms = [z[i, k] for (i, k) in z.keys()]
    #     if objective_terms:
    #         model.setObjective(quicksum(objective_terms), sense="minimize")
    #     else:
    #         model.setObjective(0.0, sense="minimize")
    #
    #     print("Construction of presolved MILP completed.", flush=True)
    #     print(
    #         f"Binaries: full={n_bin_full}, used={n_bin_used}, "
    #         f"fixed_zero={n_fixed_zero}, fixed_one={n_fixed_one}",
    #         flush=True,
    #     )
    #
    #     model.optimize()
    #     print("Optimization completed...", flush=True)
    #
    #     status_name = str(model.getStatus()).upper()
    #     best_sol = model.getBestSol()
    #     has_incumbent = best_sol is not None
    #
    #     u_sol = np.full(P, np.nan, dtype=float)
    #     z_sol = np.full((n, K), np.nan, dtype=float)
    #     tau_sol = None
    #
    #     objective = None
    #     coverage_proxy = None
    #     avg_set_size_proxy = None
    #     final_gap = None
    #
    #     if has_incumbent:
    #         for c in range(P):
    #             u_sol[c] = model.getSolVal(best_sol, u[c])
    #
    #         z_sol = np.zeros((n, K), dtype=float)
    #         z_sol[fixed_zero] = 0.0
    #         z_sol[fixed_one] = 1.0
    #
    #         for (i, k), var in z.items():
    #             z_sol[i, k] = 1.0 if model.getSolVal(best_sol, var) >= 0.5 else 0.0
    #
    #         tau_sol = model.getSolVal(best_sol, tau)
    #
    #         # Add the constant contribution from variables fixed to 1
    #         objective = float(n_fixed_one + model.getSolObjVal(best_sol))
    #         coverage_proxy = float(np.sum(z_sol[np.arange(n), y]))
    #         avg_set_size_proxy = float(np.sum(z_sol) / n)
    #
    #         try:
    #             final_gap = float(model.getGap())
    #         except Exception:
    #             final_gap = None
    #
    #     return {
    #         "status": status_name,
    #         "objective": objective,
    #         "u": u_sol,
    #         "z": z_sol,
    #         "coverage_proxy": coverage_proxy,
    #         "avg_set_size_proxy": avg_set_size_proxy,
    #         "tau": None if tau_sol is None else float(tau_sol),
    #         "M": float(max_M_used),
    #         "runtime": float(model.getSolvingTime()),
    #         "mip_gap": final_gap,
    #         "n_binary_full": n_bin_full,
    #         "n_binary_used": n_bin_used,
    #         "n_binary_fixed_zero": n_fixed_zero,
    #         "n_binary_fixed_one": n_fixed_one,
    #     }
