from logging import raiseExceptions

import numpy as np
import sys


class WeightedRank2DPredictor:

    def __init__(self, precalibration_data, n_classes:int = 10, seed: int = 2026):
        self.n_classes = n_classes
        self.seed = seed
        self.dict_pattern_and_w = {}


        tmp_probDistributions, tmp_targets = zip(*precalibration_data)
        self.allProbDist = np.asarray(list(zip(*tmp_probDistributions)))
        self.allTargets = np.asarray(tmp_targets)

        del tmp_probDistributions, tmp_targets
        self.n_augmentations = len(self.allProbDist[0])-1

        assert self.allProbDist.shape[0] == 2, "At the moment the approach works only with one augmentation"

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

        return np.hstack(tup=(ranks_per_classes_orig, ranks_per_classes_aug), dtype=int)


    # def compute_weights_per_pattern(self,):
    def precalibrate(self):
        np_unique_patterns, np_iverse_per_pattern, freq_unique_patterns = np.unique(self.extracted_patterns,
                                                                                 axis=0,
                                                                                 return_inverse=True,
                                                                                 return_counts=True)

        mask = freq_unique_patterns > 1
        keep_ids = np.flatnonzero(mask)

        # gefilterte unique patterns und deren Frequenzen
        np_unique_patterns_f = np_unique_patterns[mask]

        # Indizes pro behaltenem Pattern (bezogen auf np_all_patterns)
        indices_per_pattern_f = [np.where(np_iverse_per_pattern == uid)[0] for uid in keep_ids] # n,d

        # Optimierung
        for i, single_pattern in enumerate(np_unique_patterns_f):
            print("Optimierung pattern: ", i+1," /", len(np_unique_patterns_f))
            current_indices = indices_per_pattern_f[i]

            # self.allProbDist

            np_probDist_orig_current_pattern = self.allProbDist[0][current_indices]
            np_probDist_aug_current_pattern = self.allProbDist[1][current_indices]
            np_true_classes = self.allTargets[current_indices]

            # ===================================================================================
            # Pattern extractor for external encoding of the optimization problem for experiments
            # ===================================================================================
            # print("##PATTERN DATA",i,len(np_true_classes))
            # for i in range(len(np_true_classes)):
            #     for j in range(len(np_probDist_orig_current_pattern[i])):
            #         sys.stdout.write(str(np_probDist_orig_current_pattern[i][j])+" ")
            #     for j in range(len(np_probDist_orig_current_pattern[i])):
            #         sys.stdout.write(str(np_probDist_aug_current_pattern[i][j])+" ")
            #     print(np_true_classes[i])
            
            current_w = self.compute_weight_for_pattern(probDist_orig=np_probDist_orig_current_pattern,
                                                        probDist_aug=np_probDist_aug_current_pattern,
                                                        true_classes=np_true_classes)
            key = tuple(single_pattern.tolist())
            self.dict_pattern_and_w[key] = current_w

        print("Precalibration done")



    # 1-p
    def compute_weight_for_pattern(self, probDist_orig: np.ndarray, probDist_aug: np.ndarray, true_classes: np.ndarray):

        best_set_size = np.inf
        best_w = 0

        w_candidates = np.linspace(0, 1, 20)

        for w in w_candidates:
            combined_prob = (1-w) * probDist_orig + w * probDist_aug

            set_size = self.get_setsize_precalibration_baseline(combined_prob, true_classes)
            if best_set_size > set_size:
                best_set_size = set_size
                best_w = w
                print(f"Best set size: {best_set_size} and best w: {best_w}")

        return best_w


    def get_setsize_precalibration_baseline(self, probDist: np.ndarray, true_classes):

        inv_probDist = 1.0 - probDist
        n = inv_probDist.shape[0]

        cal_value_for_max_coverage = np.max(inv_probDist[np.arange(n), true_classes])

        count = (inv_probDist <= cal_value_for_max_coverage).sum(axis=0) # n,1

        return count.sum()

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

        pi = self.dict_pattern_and_w.get(pat_key, 0)

        w_eff = pi

        if w_eff == 0:
            return probDistOrig

        # e.g. default_pi = global prior P(Z=1) for this class, or 0.5
        weighted_prob = (1 - w_eff) * probDistOrig + w_eff * probDistAug

        # print(f"w_eff: {w_eff},\n weighted_prob: {weighted_prob}, probDistOrig: {probDistOrig}, probDistAug: {probDistAug}\n\n")

        weighted_prob = np.maximum(weighted_prob, 0)
        total = np.sum(weighted_prob)
        normalized_prob = weighted_prob / total if total > 0 else probDistOrig

        return normalized_prob


    def calibration(self, calibration_data):

        pass


    def predict(self, probabilityDistributions, calibrationValue):
        assert len(probabilityDistributions) == 2
        resultSet = []
        weighted_prob_dist = self.compute_prob_dist(probDistOrig=probabilityDistributions[0],
                                                    probDistAug=probabilityDistributions[1])
        # print("prediction")
        # print(weighted_prob_dist)

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


        # # ##############################################
        # # # TODO: Predict following sum_max
        # # ##############################################
        # while (probabilityCoveredSoFar <= calibrationValue) and len(restClasses) > 0:
        #     # Search for the remaining class Combination with the highest value

        #     # if len(restClasses) == 0:

        #     if max(weighted_prob_dist) <= 0:
        #         done = True
        #         break
        #     probabilityCoveredSoFar += max(weighted_prob_dist)
        #     resultSet.extend([int(np.argmax(weighted_prob_dist))])
        #     weighted_prob_dist[np.argmax(weighted_prob_dist)] = -10
        #     restClasses.difference_update(resultSet)

        # # # Numerics Fall-Back: If probability sum was already 1 or above, add the rest of the classes
        # if ((calibrationValue == 1.0) or done) and len(restClasses) > 0:
        #     resultSet.extend(restClasses)

        # # NextAfter....
        # return resultSet, math.nextafter(probabilityCoveredSoFar, math.inf)

    def texInfo(self):
        return """Basic 2D predictor. Takes class combination with the highest product until the probability mass of all combinations considered is greater than the calibrated value."""

    def getShortName(self):
        return "weigted-rank-2D"
