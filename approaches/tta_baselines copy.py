import os
import numpy as np
EPS = 1e-12

def _softmax_lastaxis(S):
    S = np.asarray(S, dtype=np.float64)
    Z = S - np.max(S, axis=-1, keepdims=True); np.exp(Z, out=Z)
    Z /= np.clip(Z.sum(axis=-1, keepdims=True), EPS, None)
    return Z


class _MixturePredictorBase:
    """TTA aggregation (Shanmugam Eq.5) + a pluggable conformal score.
    score='aps' -> Romano APS ; score='raps' -> deterministic RAPS. TTA and
    the score are orthogonal, so any subclass x any score is a valid predictor."""

    def __init__(self, precalibration_data, alpha, path_logs, n_classes=1000,
                 seed=2026, verbose=True, inputs_type="logits",
                 score="aps", kreg=1, lamda=0.01):
        if inputs_type not in ("logits", "probs"):
            raise ValueError("inputs_type must be 'logits' or 'probs'.")
        
        if score not in ("aps", "raps", "thr"):
            raise ValueError("score must be 'aps', 'raps', or 'thr'.")
        
        self.inputs_type = inputs_type; self.score = score
        self.kreg = int(kreg); self.lamda = float(lamda)
        self.n_classes = int(n_classes); self.seed = int(seed)
        self.alpha = float(alpha); self.verbose = verbose; self.path_logs = path_logs
        self._rng = np.random.default_rng(self.seed)
        self.p_views_pre, self.targets_pre = self._parse(precalibration_data)
        self.n_views = int(self.p_views_pre.shape[0])
        if self.n_views < 2:
            raise ValueError("Need original view plus >= 1 augmentation.")
        self.A = self.n_views - 1; 
        self.n = int(self.p_views_pre.shape[1])
        if self.inputs_type == "logits" and self._looks_like_probs(self.p_views_pre):
            raise ValueError("inputs_type='logits' but views look like probabilities.")
        self.theta = None; self.qhat = None

    @staticmethod
    def _parse(data):
        print("#"*10)
        print("CALLING PARSE")
        tmp_dist, tmp_targets = zip(*data)
        per_view = list(zip(*tmp_dist))
        V = np.stack([np.stack(v, axis=0) for v in per_view], axis=0).astype(np.float64)
        print("#"*10)
        return V, np.asarray(tmp_targets, dtype=np.int64)
    
    @staticmethod
    def _looks_like_probs(V):
        return bool(np.all(V >= -1e-9) and np.allclose(V.sum(axis=-1), 1.0, atol=1e-3))

    def _to_logits(self, V):
        return V if self.inputs_type == "logits" else np.log(np.clip(V, EPS, None))

    def _aggregate(self, V):
        return _softmax_lastaxis(np.tensordot(self.theta, self._to_logits(V), axes=(0, 0)))
    
    # ---- THR ----
    @staticmethod
    def _thr_true_scores(Z, y):
        """Nonconformity score s(x, y) = 1 - p_y."""
        return 1.0 - Z[np.arange(Z.shape[0]), y]

    @staticmethod
    def _thr_all_scores(z):
        """Score for every possible class."""
        return 1.0 - z

    # ---- APS ----
    def _aps_true_scores(self, Z, y, u):
        piy = Z[np.arange(Z.shape[0]), y]
        return np.sum(Z * (Z > piy[:, None]), axis=1) + u * piy

    @staticmethod
    def _aps_all_scores(z, u):
        return (z[None, :] * (z[None, :] > z[:, None])).sum(axis=1) + u * z

    # ---- RAPS ----
    def _raps_prefix(self, z):
        order = np.argsort(-z, kind="stable"); sp = z[order]
        pre = np.cumsum(sp) + self.lamda * np.maximum(0.0, np.arange(1, z.shape[0] + 1) - self.kreg)
        return pre / (1.0 + self.lamda * max(0.0, z.shape[0] - self.kreg)), order

    def _raps_true_scores(self, Z, y):
        out = np.empty(Z.shape[0])
        for i in range(Z.shape[0]):
            pre, order = self._raps_prefix(Z[i]); s = int(np.flatnonzero(order == y[i])[0])
            out[i] = 0.0 if s == 0 else float(pre[s - 1])
        return out

    def _true_scores(self, Z, y):
        if self.score == "aps":
            return self._aps_true_scores(Z, y, self._rng.uniform(size=Z.shape[0]))
        
        if self.score == "thr":
            return self._thr_true_scores(Z, y)
        
        
        if self.score == "raps":
            return self._raps_true_scores(Z, y)

        raise RuntimeError(f"Unsupported score: {self.score}")

    # def _maybe_tune_raps(self, kregs=(1, 2, 5), lamdas=(0.001, 0.01, 0.05, 0.1)):
    def _maybe_tune_raps(self, kregs=(1, 2), lamdas=(0.001, 0.01, 0.1)):
        if self.score != "raps":
            return
        Z, y = self._aggregate(self.p_views_pre), self.targets_pre
        n, K = Z.shape; best = (np.inf, self.kreg, self.lamda)
        for kreg in kregs:
            for lam in lamdas:
                self.kreg, self.lamda = int(kreg), float(lam)
                sc = self._raps_true_scores(Z, y)
                k = int(np.ceil((n + 1) * (1 - self.alpha)))
                cv = np.inf if k > n else float(np.partition(sc, k - 1)[k - 1])
                sizes = [min(int((self._raps_prefix(Z[i])[0] <= cv).sum()) + 1, K) for i in range(n)]
                avg = float(np.mean(sizes))
                if avg < best[0] - 1e-12:
                    best = (avg, int(kreg), float(lam))
        self.kreg, self.lamda = best[1], best[2]
        if self.verbose:
            print(f"[raps-tune] kreg={self.kreg} lamda={self.lamda} precal_size={best[0]:.3f}", flush=True)

    # ---- calibrate / predict ----
    def calibrate(self, calibration_data, alpha):
        assert self.theta is not None, "precalibrate() first."
        V, y = self._parse(calibration_data); Z = self._aggregate(V)
        sc = self._true_scores(Z, y); n = sc.shape[0]
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        self.qhat = np.inf if k > n else float(np.partition(sc, k - 1)[k - 1])
        return self.qhat

    def compute_prob_dist(self, viewOrig, viewAugs):
        if isinstance(viewAugs, np.ndarray) and viewAugs.ndim == 1:
            viewAugs = [viewAugs]
        views = [np.asarray(viewOrig, dtype=np.float64)] + [np.asarray(q, dtype=np.float64) for q in viewAugs]
        return self._aggregate(np.stack(views, axis=0))

    def predict(self, predictedLogitVector, calibrationValue=None):
        tau = self.qhat if calibrationValue is None else calibrationValue

        if tau is None:
            raise RuntimeError("No calibration value available. Call calibrate() first.")

        z = self.compute_prob_dist(
            predictedLogitVector[0],
            list(predictedLogitVector[1:]),
        )

        if self.score == "aps":
            sc = self._aps_all_scores(z, float(self._rng.uniform()), )

            inc = np.nonzero(sc < tau)[0]

            if inc.size == 0:
                inc = np.array([int(np.argmax(z))])

            exc = sc[sc >= tau]
            return (inc.tolist(), float(exc.min()) if exc.size else None, )

        if self.score == "thr":
            sc = self._thr_all_scores(z)

            # Use <= to match the quantile definition used during calibration.
            inc = np.nonzero(sc <= tau)[0]

            if inc.size == 0:
                inc = np.array([int(np.argmax(z))])

            exc = sc[sc > tau]
            return (inc.astype(int).tolist(), float(exc.min()) if exc.size else None, )

        if self.score == "raps":
            pre, order = self._raps_prefix(z)
            K = pre.shape[0]

            if tau >= 1.0:
                return order.astype(int).tolist(), None

            size = min(int((pre <= tau).sum()) + 1, K)
            larger = pre[pre > tau]
            return (order[:size].astype(int).tolist(), float(larger[0]) if size < K and larger.size else None, )

        raise RuntimeError(f"Unsupported score: {self.score}")


class TTAMeanPredictor(_MixturePredictorBase):
    def precalibrate(self):
        self.theta = np.full(self.n_views, 1.0 / self.n_views); self._maybe_tune_raps()
    def texInfo(self): return f"TTA-Avg + {self.score.upper()}"
    def getShortName(self): return f"TTA-Mean-{self.A}A-{self.score}"

class OriginalOnlyPredictor(_MixturePredictorBase):
    """No TTA: theta selects view 0 -> plain APS / plain RAPS baseline."""
    def precalibrate(self):
        self.theta = np.zeros(self.n_views); self.theta[0] = 1.0; self._maybe_tune_raps()
    def texInfo(self): return f"No-TTA (original view) + {self.score.upper()}"
    def getShortName(self): return f"Orig-{self.score}"

class GlobalWeightPredictor(_MixturePredictorBase):
    def __init__(self, *a, weight_decay=1e-4, max_iters=500, tol=1e-9, **k):
        super().__init__(*a, **k)
        self.weight_decay = float(weight_decay); self.max_iters = int(max_iters); self.tol = float(tol)
    def _ce_and_grad(self, theta, L, onehot, y):
        N = L.shape[1]; Z = _softmax_lastaxis(np.tensordot(theta, L, axes=(0, 0)))
        ce = -np.mean(np.log(np.clip(Z[np.arange(N), y], EPS, None))) + 0.5 * self.weight_decay * float(theta @ theta)
        grad = np.tensordot(L, (Z - onehot) / N, axes=([1, 2], [0, 1])) + self.weight_decay * theta
        return ce, grad
    def precalibrate(self):
        L = self._to_logits(self.p_views_pre); y = self.targets_pre; M, N, K = L.shape
        onehot = np.zeros((N, K)); onehot[np.arange(N), y] = 1.0
        theta = np.full(M, 1.0 / M); ce, grad = self._ce_and_grad(theta, L, onehot, y)
        for _ in range(self.max_iters):
            g2 = float(grad @ grad)
            if g2 < self.tol: break
            t = 1.0
            while True:
                cand = theta - t * grad; ce_new, grad_new = self._ce_and_grad(cand, L, onehot, y)
                if ce_new <= ce - 1e-4 * t * g2 or t < 1e-12: break
                t *= 0.5
            conv = (ce - ce_new < self.tol); theta, ce, grad = cand, ce_new, grad_new
            if conv: break
        self.theta = theta; self._maybe_tune_raps()
    def texInfo(self): return f"TTA-Learned + {self.score.upper()}"
    def getShortName(self): return f"Global-w-{self.A}A-{self.score}"