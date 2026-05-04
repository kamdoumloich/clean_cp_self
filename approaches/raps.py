import numpy as np


class RAPSConformanceEvaluatorSingleDistribution:
    """
    Regularized Adaptive Prediction Sets (RAPS), adapted to the local predictor interface.

    Reference implementation details were taken from:
    https://github.com/aangelopoulos/conformal_classification
    """

    def __init__(
        self,
        n_classes: int = 10,
        kreg: int = 1,
        lamda: float = 0.01,
        allow_zero_sets: bool = False,
    ):
        if kreg < 0:
            raise ValueError("kreg must be >= 0.")
        if lamda < 0:
            raise ValueError("lamda must be >= 0.")

        self.n_classes = int(n_classes)
        self.kreg = int(kreg)
        self.lamda = float(lamda)
        self.allow_zero_sets = bool(allow_zero_sets)

    def precalibrate(self):
        pass

    def _get_sorted_scores_and_indices(self, probabilityDistributions):
        if len(probabilityDistributions) == 0:
            raise ValueError("RAPS needs at least one probability distribution.")

        probs = np.asarray(probabilityDistributions[0], dtype=np.float64)
        if probs.ndim != 1:
            raise ValueError(
                f"Expected a 1D probability vector, got shape {probs.shape}."
            )

        sorted_indices = np.argsort(-probs, kind="stable")
        sorted_scores = probs[sorted_indices]
        return sorted_scores, sorted_indices

    def _regularized_prefix_scores(self, sorted_scores):
        k = sorted_scores.shape[0]
        cumsum = np.cumsum(sorted_scores)
        positions = np.arange(1, k + 1, dtype=np.float64)
        penalties = self.lamda * np.maximum(0.0, positions - float(self.kreg))
        prefix_scores = cumsum + penalties

        # COMPACT assumes calibration values in [0, 1], with 1 returning all
        # classes. RAPS penalties can push raw prefix scores above 1, so we
        # normalize by the full-set score while preserving their ordering.
        full_set_score = float(prefix_scores[-1])
        if full_set_score <= 0.0:
            return np.zeros_like(prefix_scores)
        return prefix_scores / full_set_score

    def _next_boundary(self, prefix_scores, calibrationValue):
        larger = prefix_scores[prefix_scores > calibrationValue]
        if larger.size == 0:
            return None
        return float(larger[0])

    def predict(self, probabilityDistributions, calibrationValue):
        sorted_scores, sorted_indices = self._get_sorted_scores_and_indices(
            probabilityDistributions
        )

        prefix_scores = self._regularized_prefix_scores(sorted_scores)
        k = prefix_scores.shape[0]

        if calibrationValue >= 1.0:
            return sorted_indices.astype(int).tolist(), None

        # Same deterministic size rule as gcq(..., randomized=False):
        # size = #{m: prefix_score_m <= tau} + 1, clipped to [1, K]
        size = int(np.count_nonzero(prefix_scores <= calibrationValue)) + 1
        size = min(size, k)

        if self.allow_zero_sets and calibrationValue < prefix_scores[0]:
            size = 0

        if size == 0:
            resultSet = []
            return resultSet, self._next_boundary(prefix_scores, calibrationValue)

        resultSet = sorted_indices[:size].astype(int).tolist()
        nextOne = None if size >= k else self._next_boundary(
            prefix_scores,
            calibrationValue,
        )
        return resultSet, nextOne

    def texInfo(self):
        return (
            "RAPS conformal predictor (Regularized Adaptive Prediction Sets) "
            "on the original probability distribution."
        )

    def getShortName(self):
        return f"RAPS(k={self.kreg},lam={self.lamda})"
