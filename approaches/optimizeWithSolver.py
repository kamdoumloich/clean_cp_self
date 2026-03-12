import math
import numpy as np
from pyscipopt import Model, quicksum


class OptimizeWithSolver:

    def __init__(self, time_limit=60, mip_gap=1e-3, verbose=False, eps=1e-6):
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.verbose = verbose
        self.eps = eps

    def optimize_w_for_all_patterns_simultaneously(self, p_orig, p_aug, y, index_by_pattern, alpha, ):
        one_minus_p_orig = np.asarray(1.0 - p_orig, dtype=float)
        one_minus_p_aug = np.asarray(1.0 - p_aug, dtype=float)
        y = np.asarray(y, dtype=int)
        index_by_pattern = np.asarray(index_by_pattern, dtype=int)

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
        model.setParam("limits/gap", float(self.mip_gap))

        ub_u = 1.0
        M_eff = 1.0 + float(self.eps)  # sufficient since scores and tau are in [0, 1]

        u = {}
        z = {}

        tau = model.addVar(name="global_tau", vtype="C", lb=0.0, ub=1.0)

        for c in range(P):
            u[c] = model.addVar(name=f"u_{c}", vtype="C", lb=0.0, ub=ub_u)

        for i in range(n):
            for k in range(K):
                z[i, k] = model.addVar(name=f"z_{i}_{k}", vtype="B")

        for i in range(n):
            c = int(index_by_pattern[i])

            for k in range(K):
                score_expr = (
                        float(one_minus_p_orig[i, k]) * u[c]
                        + float(one_minus_p_aug[i, k]) * (1.0 - u[c])
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

        model.optimize()

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
