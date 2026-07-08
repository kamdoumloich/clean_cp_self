import os
from itertools import product

import numpy as np

from utils.function_utils import UtilsConformalPrediction

EPS = 1e-12


class _MixturePredictorBase:
    """Shared machinery: arithmetic mixture z = (1 - sum w) p + sum_j w_j q_j,
    THR prediction with at-least-one-class rule. Subclasses set self.w."""

    def __init__(self, precalibration_data, alpha, path_logs, n_classes: int = 10,
                 seed: int = 2026, verbose: bool = True, **_compat):
        self.n_classes = int(n_classes)
        self.seed = int(seed)
        self.alpha = float(alpha)
        self.verbose = verbose
        self.path_logs = path_logs

        tmp_dist, tmp_targets = zip(*precalibration_data)
        allProbDist = np.asarray(list(zip(*tmp_dist)))
        if allProbDist.shape[0] < 2:
            raise ValueError("Need original view plus >= 1 augmentation.")
        self.n_views = int(allProbDist.shape[0])
        self.A = self.n_views - 1
        self.p_orig = np.stack(allProbDist[0], axis=0).astype(np.float64)
        self.q_augs = [np.stack(allProbDist[j], axis=0).astype(np.float64) for j in range(1, self.n_views)]
        self.allTargets = np.asarray(tmp_targets, dtype=np.int64)
        self.n = self.allTargets.shape[0]
        del tmp_dist, tmp_targets

        self.w = None  # tuple length A, frozen by precalibrate()

    def _mix_batch(self, p, Qs, w):
        z = (1.0 - float(np.sum(w))) * p
        for wj, q in zip(w, Qs):
            z = z + wj * q
        return z

    def calibration(self, calibration_data):
        pass

    def compute_prob_dist(self, probDistOrig, probDistAugs):
        p = np.asarray(probDistOrig, dtype=np.float64)
        if isinstance(probDistAugs, np.ndarray) and probDistAugs.ndim == 1:
            probDistAugs = [probDistAugs]
        Qs = [np.asarray(q, dtype=np.float64) for q in probDistAugs]
        if len(Qs) != self.A:
            raise ValueError(f"Expected {self.A} augmented views, got {len(Qs)}.")
        return self._mix_batch(p, Qs, self.w)

    def predict(self, probabilityDistributions, calibrationValue):
        assert len(probabilityDistributions) == self.n_views
        z = self.compute_prob_dist(probabilityDistributions[0],
                                   list(probabilityDistributions[1:]))
        resultSet = []
        nextOne = None
        maxClassValue = 0.0
        maxClass = None
        for i in range(len(z)):
            if z[i] > maxClassValue:
                maxClassValue = z[i]; maxClass = i
            if 1.0 - z[i] <= calibrationValue:
                resultSet.append(i)
            else:
                nextOne = (1.0 - z[i]) if nextOne is None else min(nextOne, 1.0 - z[i])
        if len(resultSet) == 0:
            resultSet = [maxClass]
        return resultSet, nextOne


class TTAMeanPredictor(_MixturePredictorBase):
    """Uniform test-time augmentation average: z = mean(p, q_1, ..., q_A),
    followed by standard THR split-CP. Input-independent, weight-free anchor
    in the spirit of TTA-averaging baselines."""

    def precalibrate(self):
        self.w = tuple([1.0 / self.n_views] * self.A)
        if self.verbose:
            print(f"[tta-mean] w = {self.w}", flush=True)

    def texInfo(self):
        return (f"Uniform test-time-augmentation average over the original and "
                f"{self.A} augmented view(s), followed by the standard "
                "threshold conformal predictor.")

    def getShortName(self):
        return f"TTA-Mean-{self.A}A"


class GlobalWeightPredictor(_MixturePredictorBase):
    """Single GLOBAL simplex weight vector w (one scalar per view, no
    per-sample conditioning): z = (1 - sum w) p + sum_j w_j q_j. w is chosen
    on the pre-calibration set as the grid point minimizing the average set
    size at its own recalibrated split-CP quantile (deployment-faithful
    criterion, matching QCM's). This is the ablation that isolates the value
    of per-sample conditioning: it is the in-framework analogue of globally
    weighted aggregation (cf. Luo & Zhou, arXiv:2407.10230, who aggregate at
    the score level with global weights)."""

    def __init__(self, *args, w_step: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        steps = np.round(np.arange(0.0, 1.0 + 1e-9, float(w_step)), 10)
        self.combos = [c for c in product(steps, repeat=self.A) if sum(c) <= 1.0 + 1e-9]

    def precalibrate(self):
        os.makedirs(self.path_logs, exist_ok=True)
        n = self.n
        best = None
        for w in self.combos:
            z = self._mix_batch(self.p_orig, self.q_augs, w)
            tau = UtilsConformalPrediction.compute_approximative_calibration_value(
                one_minus_probDist=(1.0 - z), targets=self.allTargets,
                alpha=self.alpha)
            size = float(((1.0 - z) <= tau).sum(axis=1).mean())
            if best is None or size < best[0] - 1e-12:
                best = (size, w)
        self.w = tuple(float(v) for v in best[1])
        if self.verbose:
            print(f"[global-w] w = {self.w}, precal size = {best[0]:.4f}",
                  flush=True)
        with open(os.path.join(self.path_logs, f"globalw_log_{self.seed}_alpha{self.alpha}.txt"), "w", encoding="utf-8") as f:
            f.write(f"[global-w] w = {self.w}\n")
            f.write(f"[global-w] precal recalibrated avg size = {best[0]}\n")
            f.write(f"[global-w] grid points = {len(self.combos)}\n")

    def texInfo(self):
        return (f"Global-weight mixture over the original and {self.A} "
                "augmented view(s): a single simplex weight vector selected on "
                "the pre-calibration set by minimizing the recalibrated "
                "average set size, followed by the standard threshold "
                "conformal predictor. No per-sample conditioning.")

    def getShortName(self):
        return f"Global-w-{self.A}A"
