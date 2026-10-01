"""
tta_baselines.py  --  patched version.

Changes w.r.t. the previous version (all defaults preserve the old numerics):

  * explicit "leading view block" (n_leading_views / leading_views_mode /
    reference_view / leading_pool / leading_others_as_aug);
    self.A is now  n_views - n_leading_views  (identical to n_views - 1 for
    the default n_leading_views=1);
  * one single definition of every conformal score, vectorised over (N, K),
    shared by every subclass:  _thr_all_scores / _aps_all_scores /
    _raps_all_scores / _all_scores / _true_scores;
  * one single conformal quantile: _conformal_quantile;
  * one single "scores -> prediction set" rule: _set_from_scores
    (uses <= everywhere, and honours the framework contract
     "calibrationValue >= 1 returns all classes");
  * calibrate() is now generic; subclasses only override
    _predictive_distribution() or _calibration_true_scores();
  * _tune_raps() is a reusable, vectorised RAPS hyper-parameter search;
    _maybe_tune_raps() keeps the old behaviour.
"""

import numpy as np

EPS = 1e-24

_VALID_SCORES = ("thr", "aps", "raps")
_VALID_LEADING_MODES = ("mean", "individual")
_VALID_LEADING_POOLS = ("arithmetic", "geometric")


def _softmax_lastaxis(S):
    S = np.asarray(S, dtype=np.float64)
    Z = S - np.max(S, axis=-1, keepdims=True)
    np.exp(Z, out=Z)
    Z /= np.clip(Z.sum(axis=-1, keepdims=True), EPS, None)
    return Z


class _MixturePredictorBase:
    """TTA aggregation + a pluggable conformal score.

    The first `n_leading_views` views are the "leading block" (the object that
    plays the role of the original / reference prediction).  All remaining
    views are augmentations.
    """

    # ------------------------------------------------------------------ init
    def __init__(self, precalibration_data, alpha, path_logs,
                 n_augs:int, n_classes=1000,
                 seed=2026, verbose=True, inputs_type="logits",
                 scoring_function="aps", kreg=1, lamda=0.01,
                 n_leading_views=1, leading_views_mode="individual",
                 reference_view=0, leading_pool="arithmetic",
                 leading_others_as_aug=False, aps_randomized=True,
                 score_in_log_space=False):

        if inputs_type not in ("logits", "probs"):
            raise ValueError("inputs_type must be 'logits' or 'probs'.")
        if scoring_function not in _VALID_SCORES:
            raise ValueError(f"score must be one of {_VALID_SCORES}.")
        if leading_views_mode not in _VALID_LEADING_MODES:
            raise ValueError(f"leading_views_mode must be one of {_VALID_LEADING_MODES}.")
        if leading_pool not in _VALID_LEADING_POOLS:
            raise ValueError(f"leading_pool must be one of {_VALID_LEADING_POOLS}.")

        self.inputs_type = inputs_type
        self.scoring_function = scoring_function
        self.kreg = int(kreg)
        self.lamda = float(lamda)
        self.n_classes = int(n_classes)
        self.seed = int(seed)
        self.alpha = float(alpha)
        self.verbose = bool(verbose)
        self.path_logs = path_logs
        self.aps_randomized = bool(aps_randomized)
        self.score_in_log_space = bool(score_in_log_space)
        self._rng = np.random.default_rng(self.seed)
        self.at_least_one_calibration = False

        self.p_views_pre, self.targets_pre = self._parse(precalibration_data)
        if self.p_views_pre.ndim != 3:
            raise ValueError(f"Precalibration views must be (M,N,K), got {self.p_views_pre.shape}.")
        
        self.n_views = int(n_augs) + n_leading_views # original policy added 
        # self.n_views = int(self.p_views_pre.shape[0])
        self.n = int(self.p_views_pre.shape[1])
        self.K = int(self.p_views_pre.shape[2])

        L = int(n_leading_views)
        if not (1 <= L < self.n_views):
            raise ValueError(f"n_leading_views must satisfy 1 <= L < n_views ({self.n_views}), got {L}.")
        self.n_leading_views = L
        self.leading_views_mode = leading_views_mode
        self.leading_pool = leading_pool
        self.leading_others_as_aug = bool(leading_others_as_aug)
        self.reference_view = int(reference_view)

        if self.leading_views_mode == "individual" and not (0 <= self.reference_view < L):
            raise ValueError( "reference_view must lie inside the leading block [0,{L}) when leading_views_mode='individual'.")

        self.aug_indices = self._compute_aug_indices()
        self.A = len(self.aug_indices)
        if self.A < 1:
            raise ValueError("Need at least one augmentation view.")

        if self.inputs_type == "logits" and self._looks_like_probs(self.p_views_pre):
            raise ValueError("inputs_type='logits' but views look like probabilities.")

        self.theta = None
        self.qhat = None

    def _compute_aug_indices(self):
        if self.leading_views_mode == "individual" and self.leading_others_as_aug:
            return [j for j in range(self.n_views) if j != self.reference_view]
        return list(range(self.n_leading_views, self.n_views))

    # ------------------------------------------------------------------ data
    @staticmethod
    def _parse(data):
        # tmp_dist, tmp_targets = zip(*data)
        # per_view = list(zip(*tmp_dist))

        V, y = data

        V = np.asarray(V, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        if V.ndim != 3:
            raise ValueError(f"Expected V with shape (n_augmentations, n_samples, K), got {V.shape}.")

        if y.ndim != 1:
            raise ValueError(f"Expected y to be 1-dimensional, got {y.shape}.")

        if V.shape[1] != y.shape[0]:
            raise ValueError(f"Number of samples in V and y does not match: V has {V.shape[1]} samples, y has {y.shape[0]}.")

        return V, y

        # V = np.stack([np.stack(v, axis=0) for v in per_view], axis=0).astype(np.float64)
        # return V, np.asarray(tmp_targets, dtype=np.int64)

    @staticmethod
    def _looks_like_probs(V):
        return bool(np.all(V >= -1e-12) and np.allclose(V.sum(axis=-1), 1.0, atol=1e-6))

    def _to_logits(self, V):
        return np.asarray(V, dtype=np.float64) if self.inputs_type == "logits" \
            else np.log(np.clip(np.asarray(V, dtype=np.float64), EPS, None))

    def _to_probs(self, V):
        V = np.asarray(V, dtype=np.float64)
        if self.inputs_type == "logits":
            return _softmax_lastaxis(V)
        V = np.clip(V, 0.0, None)
        return V / np.clip(V.sum(axis=-1, keepdims=True), EPS, None)

    # ------------------------------------------------------- view aggregation
    def _leading_theta(self):
        th = np.zeros(self.n_views, dtype=np.float64)
        if self.leading_views_mode == "individual":
            th[self.reference_view] = 1.0
        else:
            th[:self.n_leading_views] = 1.0 / self.n_leading_views
        return th

    def _aggregate(self, V):
        """Log-linear (geometric) pooling of all views with weights theta."""
        return _softmax_lastaxis(np.tensordot(self.theta, self._to_logits(V), axes=(0, 0)))

    def _pool_leading(self, V):
        """Reference ('original') predictive distribution from the leading block."""
        V = np.asarray(V, dtype=np.float64)
        squeeze = (V.ndim == 2)
        if squeeze:
            V = V[:, None, :]
        if V.shape[0] != self.n_views:
            raise ValueError(f"Expected {self.n_views} views, got {V.shape[0]}.")

        L = self.n_leading_views
        if self.leading_views_mode == "individual":
            P = self._to_probs(V[self.reference_view])
        elif self.leading_pool == "arithmetic":
            P = self._to_probs(V[:L]).mean(axis=0)
        else:
            P = _softmax_lastaxis(self._to_logits(V[:L]).mean(axis=0))
        return P[0] if squeeze else P

    def _predictive_distribution(self, V):
        """Hook: distribution the conformal score is applied to."""
        return self._aggregate(V)

    # ---------------------------------------------------------------- scores
    @staticmethod
    def _thr_all_scores(P):
        return 1.0 - np.asarray(P, dtype=np.float64)

    @staticmethod
    def _aps_all_scores(P, u):
        """APS score  sum_{j: p_j > p_k} p_j + u * p_k, exact under ties."""
        P2 = np.asarray(P, dtype=np.float64)
        one_d = (P2.ndim == 1)
        if one_d:
            P2 = P2[None, :]
        N, K = P2.shape

        order = np.argsort(-P2, axis=1, kind="stable")
        Ps = np.take_along_axis(P2, order, axis=1)
        csum = np.cumsum(Ps, axis=1)

        prev = np.zeros_like(csum)
        prev[:, 1:] = csum[:, :-1]

        same = np.zeros((N, K), dtype=bool)
        same[:, 1:] = Ps[:, 1:] == Ps[:, :-1]
        pos = np.broadcast_to(np.arange(K), (N, K))
        run_start = np.maximum.accumulate(np.where(same, -1, pos), axis=1)
        strict = np.take_along_axis(prev, run_start, axis=1)

        u_arr = np.asarray(u, dtype=np.float64)
        u_col = float(u_arr) if u_arr.ndim == 0 else u_arr.reshape(-1, 1)

        S = np.empty_like(Ps)
        np.put_along_axis(S, order, strict + u_col * Ps, axis=1)
        return S[0] if one_d else S

    def _raps_all_scores(self, P, kreg=None, lamda=None):
        """Deterministic RAPS entry score: rank 1 -> 0, rank r -> prefix(r-1)."""
        kreg = self.kreg if kreg is None else int(kreg)
        lamda = self.lamda if lamda is None else float(lamda)

        P2 = np.asarray(P, dtype=np.float64)
        one_d = (P2.ndim == 1)
        if one_d:
            P2 = P2[None, :]
        K = P2.shape[1]

        order = np.argsort(-P2, axis=1, kind="stable")
        Ps = np.take_along_axis(P2, order, axis=1)
        pos = np.arange(1, K + 1, dtype=np.float64)
        pre = np.cumsum(Ps, axis=1) + lamda * np.maximum(0.0, pos[None, :] - float(kreg))
        pre /= (1.0 + lamda * max(0.0, float(K) - float(kreg)))

        entry = np.zeros_like(pre)
        entry[:, 1:] = pre[:, :-1]

        S = np.empty_like(entry)
        np.put_along_axis(S, order, entry, axis=1)
        return S[0] if one_d else S

    def _raps_prefix(self, z, kreg=None, lamda=None):
        """Kept for backward compatibility: normalized prefixes + descending order."""
        kreg = self.kreg if kreg is None else int(kreg)
        lamda = self.lamda if lamda is None else float(lamda)
        z = np.asarray(z, dtype=np.float64)
        order = np.argsort(-z, kind="stable")
        sp = z[order]
        pre = np.cumsum(sp) + lamda * np.maximum(0.0, np.arange(1, z.shape[0] + 1) - kreg)
        return pre / (1.0 + lamda * max(0.0, z.shape[0] - kreg)), order

    def _aps_u(self, n):
        if not self.aps_randomized:
            return 0.0
        return self._rng.uniform(size=int(n))

    def _all_scores(self, P, score=None, kreg=None, lamda=None, u=None, chunk=None):
        """Score matrix for every class. P: (N,K) or (K,)."""
        score = self.scoring_function if score is None else score
        P2 = np.asarray(P, dtype=np.float64)
        one_d = (P2.ndim == 1)
        if one_d:
            P2 = P2[None, :]

        if chunk is not None and P2.shape[0] > int(chunk):
            out = np.empty_like(P2)
            uu = None if u is None else np.asarray(u, dtype=np.float64)
            for a in range(0, P2.shape[0], int(chunk)):
                b = min(a + int(chunk), P2.shape[0])
                u_c = None if uu is None else (uu if uu.ndim == 0 else uu[a:b])
                out[a:b] = self._all_scores(P2[a:b], score, kreg, lamda, u_c, None)
            return out[0] if one_d else out

        if score == "thr":
            S = self._thr_all_scores(P2)
        elif score == "aps":
            if u is None:
                u = self._aps_u(P2.shape[0])
            S = self._aps_all_scores(P2, u)
        elif score == "raps":
            S = self._raps_all_scores(P2, kreg, lamda)
        else:
            raise RuntimeError(f"Unsupported score: {score}")

        if getattr(self, "score_in_log_space", False):
            # Transform to log-space nonconformity: s_log = -log(max(1.0 - S, EPS))
            # For THR: S = 1 - P, so 1 - S = P and s_log = -log(P).
            # For APS: S is cumulative sum in [0, 1], so s_log = -log(1 - s_aps).
            S = -np.log(np.clip(1.0 - S, EPS, 1.0))

        return S[0] if one_d else S

    def _true_scores(self, Z, y, score=None, kreg=None, lamda=None, u=None, chunk=None):
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        y = np.asarray(y, dtype=np.int64)
        S = self._all_scores(Z, score=score, kreg=kreg, lamda=lamda, u=u, chunk=chunk)
        return S[np.arange(Z.shape[0]), y]

    # ------------------------------------------------- quantile / set-building
    @staticmethod
    def _conformal_quantile(scores, alpha):
        assert len(scores.shape) == 1, "Error in function '_conformal_quantile'"
        sc = np.asarray(scores, dtype=np.float64)
        n = sc.shape[0]
        k = int(np.ceil((n + 1) * (1.0 - float(alpha))))
        if k > n:
            return np.inf
        return float(np.partition(sc, k - 1)[k - 1])

    @staticmethod
    def _set_from_scores_predict_only(scores, tau, at_least_one=False, score_in_log_space=False):
        assert len(scores.shape) == 1, "Error in function '_set_from_scores_predict_only'"
        s = np.asarray(scores, dtype=np.float64)
        K = s.shape[0]
        if tau is None:
            raise RuntimeError("No calibration value available.")
        tau = float(tau)
        if not score_in_log_space and tau >= 1.0:
            return list(range(K)), None
        if score_in_log_space and (np.isinf(tau) or tau >= -np.log(EPS)):
            return list(range(K)), None
        inc = np.flatnonzero(s <= tau)
        if inc.size == 0 and at_least_one:
            inc = np.array([int(np.argmin(s))], dtype=np.int64)
        exc = s[s > tau]
        return inc.astype(int).tolist(), (float(exc.min()) if exc.size else None)

    # ------------------------------------------------------- RAPS tuning
    def _tune_raps(self, P, y, kregs=(1, 2), lamdas=(0.001, 0.01, 0.1)):
        """Return (kreg, lamda, avg_size) minimizing avg set size on (P, y)."""
        P = np.atleast_2d(np.asarray(P, dtype=np.float64))
        y = np.asarray(y, dtype=np.int64)
        rows = np.arange(P.shape[0])
        best = (np.inf, int(self.kreg), float(self.lamda))
        for kreg in kregs:
            for lam in lamdas:
                S = self._raps_all_scores(P, kreg, lam)
                q = self._conformal_quantile(S[rows, y], self.alpha)
                sizes = np.maximum((S <= q).sum(axis=1), 1)
                avg = float(np.mean(sizes))
                if avg < best[0] - EPS:
                    best = (avg, int(kreg), float(lam))
        return best[1], best[2], best[0]

    def _maybe_tune_raps(self, kregs=(1, 2), lamdas=(0.001, 0.01, 0.1)):
        if self.scoring_function != "raps":
            return
        Z = self._predictive_distribution(self.p_views_pre)
        self.kreg, self.lamda, avg = self._tune_raps(Z, self.targets_pre, kregs, lamdas)
        if self.verbose:
            print(f"[raps-tune] kreg={self.kreg} lamda={self.lamda} "
                  f"precal_size={avg:.3f}", flush=True)

    # ------------------------------------------------------ calibrate/predict
    def _check_calibration_ready(self):
        assert self.theta is not None, "precalibrate() first."

    def _get_calibration_true_scores(self, V, y):
        Z = self._predictive_distribution(V)
        return self._true_scores(Z, y)

    def calibrate(self, calibration_data, alpha=None):
        self._check_calibration_ready()
        a = self.alpha if alpha is None else float(alpha)
        V, y = self._parse(calibration_data)
        sc = self._get_calibration_true_scores(V, np.asarray(y, dtype=np.int64))
        self.qhat = self._conformal_quantile(sc, a)
        return self.qhat

    def compute_prob_dist(self, viewOrig, viewAugs):
        if isinstance(viewAugs, np.ndarray) and viewAugs.ndim == 1:
            viewAugs = [viewAugs]
        views = [np.asarray(viewOrig, dtype=np.float64)] + \
                [np.asarray(q, dtype=np.float64) for q in viewAugs]
        return self._predictive_distribution(np.stack(views, axis=0))

    def predict(self, predictedLogitVector, calibrationValue=None):
        tau = self.qhat if calibrationValue is None else calibrationValue
        if tau is None:
            raise RuntimeError("No calibration value available. Call calibrate() first.")
        z = self.compute_prob_dist(predictedLogitVector[0], list(predictedLogitVector[1:]))
        u = float(self._rng.uniform()) if (self.scoring_function == "aps" and self.aps_randomized) else 0.0
        sc = self._all_scores(z, score=self.scoring_function, u=u)
        return self._set_from_scores_predict_only(
            sc, tau, at_least_one=self.at_least_one_calibration,
            score_in_log_space=getattr(self, "score_in_log_space", False)
        )


class TTAMeanPredictor(_MixturePredictorBase):
    def precalibrate(self):
        self.theta = np.full(self.n_views, 1.0 / self.n_views)
        self._maybe_tune_raps()

    def texInfo(self):
        log_info = " (log-space)" if getattr(self, "score_in_log_space", False) else ""
        return f"TTA-Avg + {self.scoring_function.upper()}{log_info}"

    def getShortName(self):
        log_tag = "-Log" if getattr(self, "score_in_log_space", False) else ""
        return f"TTA-Mean-{self.A}A-{self.scoring_function}{log_tag}"


class OriginalOnlyPredictor(_MixturePredictorBase):
    """No TTA: uses the leading block only (single view, or its pooled mean)."""

    def precalibrate(self):
        self.theta = self._leading_theta()
        self._maybe_tune_raps()

    def _predictive_distribution(self, V):
        return self._pool_leading(V)

    def texInfo(self):
        log_info = " (log-space)" if getattr(self, "score_in_log_space", False) else ""
        return f"No-TTA (leading views) + {self.scoring_function.upper()}{log_info}"

    def getShortName(self):
        log_tag = "-Log" if getattr(self, "score_in_log_space", False) else ""
        return f"Orig-{self.scoring_function}{log_tag}"


class GlobalWeightPredictor(_MixturePredictorBase):
    def __init__(self, *a, weight_decay=1e-4, max_iters=500, tol=1e-9, **k):
        super().__init__(*a, **k)
        self.weight_decay = float(weight_decay)
        self.max_iters = int(max_iters)
        self.tol = float(tol)

    def _ce_and_grad(self, theta, L, onehot, y):
        N = L.shape[1]
        Z = _softmax_lastaxis(np.tensordot(theta, L, axes=(0, 0)))
        ce = -np.mean(np.log(np.clip(Z[np.arange(N), y], EPS, None))) \
            + 0.5 * self.weight_decay * float(theta @ theta)
        grad = np.tensordot(L, (Z - onehot) / N, axes=([1, 2], [0, 1])) \
            + self.weight_decay * theta
        return ce, grad

    def precalibrate(self):
        L = self._to_logits(self.p_views_pre)
        y = self.targets_pre
        M, N, K = L.shape
        onehot = np.zeros((N, K))
        onehot[np.arange(N), y] = 1.0
        theta = np.full(M, 1.0 / M)
        ce, grad = self._ce_and_grad(theta, L, onehot, y)
        for _ in range(self.max_iters):
            g2 = float(grad @ grad)
            if g2 < self.tol:
                break
            t = 1.0
            while True:
                cand = theta - t * grad
                ce_new, grad_new = self._ce_and_grad(cand, L, onehot, y)
                if ce_new <= ce - 1e-4 * t * g2 or t < 1e-12:
                    break
                t *= 0.5
            conv = (ce - ce_new < self.tol)
            theta, ce, grad = cand, ce_new, grad_new
            if conv:
                break
        self.theta = theta
        self._maybe_tune_raps()

    def texInfo(self):
        log_info = " (log-space)" if getattr(self, "score_in_log_space", False) else ""
        return f"TTA-Learned + {self.scoring_function.upper()}{log_info}"

    def getShortName(self):
        log_tag = "-Log" if getattr(self, "score_in_log_space", False) else ""
        return f"Global-w-{self.A}A-{self.scoring_function}{log_tag}"