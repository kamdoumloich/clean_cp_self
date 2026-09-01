from enum import Enum

import numpy as np
import torchvision
import torch


class Augmentation(Enum):
    ORIGINAL = 'ORIGINAL'
    INCREASE_SHARPNESS = 'INCREASE_SHARPNESS'
    DECREASE_SHARPNESS = 'DECREASE_SHARPNESS'
    AUTOCONTRAST = 'AUTOCONTRAST'
    INVERT = 'INVERT'
    POSTERIZE = 'POSTERIZE'
    SHEAR = 'SHEAR'
    TRANSLATE = 'TRANSLATE'
    COLOR_JITTER = 'COLOR_JITTER'
    RANDOM_CROP = 'RANDOM_CROP'
    HORIZONTAL_FLIP = 'HORIZONTAL_FLIP'
    BLUR = 'BLUR'
    CONTRAST = 'CONTRAST'
    BRIGHTNESS = 'BRIGHTNESS'
    HUE = 'HUE'
    ROTATE = 'ROTATE'

class UtilsAugmentations:


    @staticmethod
    def get_n_evenly_spaced_values_for_augmentation(augmentation_name, lb_degree_aug:float, ub_degree_aug:float, n_amount:int):
        if isinstance(augmentation_name, Augmentation):
            augmentation_name = augmentation_name.name
        else:
            raise NotImplementedError(f"Augmentation: {augmentation_name} is not implemented.")

        if n_amount < 1:
            raise ValueError("n_amount of augmentations must be >= 1.")

        if lb_degree_aug > ub_degree_aug:
            raise ValueError("lb_degree_aug must be <= ub_degree_aug.")
        
        if augmentation_name == Augmentation.HUE.name:
            if lb_degree_aug < -0.5 or ub_degree_aug > 0.5:
                raise ValueError("For hue, values must be in [-0.5, 0.5].")
            
        if augmentation_name == Augmentation.ROTATE.name:
            # https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.affine.html
            # angle (number) – rotation angle in degrees between -180 and 180, clockwise direction.
            if lb_degree_aug <-180 or ub_degree_aug > 180:
                raise ValueError("Rotation angle should be in [-180, 180]")

        if augmentation_name in [Augmentation.BRIGHTNESS.name, Augmentation.CONTRAST.name,]:
            if lb_degree_aug <= 0 or ub_degree_aug == 1.0 or lb_degree_aug == 1.0:
                raise ValueError(f"Bounds must be != 1 (original image) and > 0.")
            
        if n_amount == 1 and lb_degree_aug != ub_degree_aug and augmentation_name != Augmentation.HORIZONTAL_FLIP.name:
            raise ValueError(f"The given setting works only for HORIZONTAL_FLIP")
            
        # n_amount + 1 to include 0
        raw_values = np.linspace(start=lb_degree_aug, stop=ub_degree_aug, num=n_amount+1, endpoint=True).round(3)
        
        if augmentation_name in ["BLUR", "POSTERIZE", "RANDOM_CROP", "HORIZONTAL_FLIP"]:
            raw_values = np.rint(raw_values).astype(int)
        
        augmentation_values = [value for value in raw_values]
        
        # refers to the original image, no need
        if augmentation_name == Augmentation.ROTATE.name and (0.0 in augmentation_values):
            augmentation_values.remove(0.0)

        return augmentation_values
    
    @staticmethod
    def apply_single_augmentation(x, name, rng, degree):
        """x: (1, C, H, W) or (B, C, H, W) float in [0,1]; one parameter draw."""
        import torch
        import torchvision.transforms.functional as TF

        # print(f"Case {name} with value {degree} and rng {rng}", flush=True)

        if name == "ORIGINAL":
            return x
        if name == "HORIZONTAL_FLIP":
            return TF.hflip(x)
        if name == "AUTOCONTRAST":
            # https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.autocontrast.html
            return TF.autocontrast(x)
        if name == "INVERT":
            return TF.invert(x)
        if name in ("INCREASE_SHARPNESS", "DECREASE_SHARPNESS"):
            # https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_sharpness.html
            #  0 gives a blurred image, 1 gives the original image while 2 increases the sharpness by a factor of 2.
            return TF.adjust_sharpness(x, sharpness_factor=degree)
        if name == "POSTERIZE":
            # https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.posterize.html
            x8 = (x.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            return TF.posterize(x8, bits=degree).float().div_(255.0)
        if name == "BLUR":
            return TF.gaussian_blur(x, kernel_size=degree)
        if name == "SHEAR":
            return TF.affine(x, angle=0.0, translate=(0, 0), scale=1.0, shear=[degree, 0.0])
        if name == "ROTATE":
            return TF.affine(x, angle=degree, translate=(0, 0), scale=1.0, shear=[0.0])
        # if name == "ROTATE_OPTIM":
        #     builder = RotationViewBuilder()
        #     return TF.affine(x, angle=degree, translate=(0, 0), scale=1.0, shear=[0.0])
        if name == "BRIGHTNESS":
            return TF.adjust_brightness(x, degree)
        if name == "CONTRAST":
            return TF.adjust_contrast(x, degree)
        if name == "SATURATION":
            return TF.adjust_saturation(x, degree)
        if name == "RANDOM_CROP":
            pad = degree
            h, w = x.shape[-2], x.shape[-1]
            padded = TF.pad(x, [pad, pad, pad, pad])
            top = int(rng.integers(0, 2 * pad + 1))
            left = int(rng.integers(0, 2 * pad + 1))
            return TF.crop(padded, top=top, left=left, height=h, width=w)

        raise NotImplementedError(f"Unknown TTA augmentation: {name}")


# class RotationViewBuilder:
#     """Rotation views with no black corners. Angle 0 reproduces
#     Resize(256) + CenterCrop(224) exactly."""

#     import math
#     import torch
#     import torchvision.transforms.functional as TF
#     from torchvision.transforms import InterpolationMode

#     def __init__(self, out_size=224, resize_short=256, pad_mode="reflect",
#                  interpolation=InterpolationMode.BILINEAR):
#         self.T = int(out_size)
#         self.R = int(resize_short)
#         self.S = int(math.ceil(self.T * math.sqrt(2)))      # 317
#         self.pad_mode = str(pad_mode)
#         self.interpolation = interpolation

#     def buffer(self, pil_image):
#         """PIL -> (3, S, S) float tensor. One resize, one pad, one crop."""
#         x = TF.resize(pil_image, self.R, interpolation=self.interpolation)
#         x = TF.to_tensor(x)                                  # (3, H, W)
#         h, w = x.shape[-2:]
#         ph, pw = max(0, self.S - h), max(0, self.S - w)
#         if ph or pw:
#             if min(h, w) <= max(ph, pw):
#                 raise ValueError("reflect padding needs pad < dimension; "
#                                  "image too small after resize")
#             x = TF.pad(x, [pw // 2, ph // 2, pw - pw // 2, ph - ph // 2],
#                        padding_mode=self.pad_mode)
#         return TF.center_crop(x, [self.S, self.S])

#     def view(self, buf, angle_deg):
#         """(3, S, S) -> (3, T, T) for one angle."""
#         if int(angle_deg) % 360 == 0:
#             return TF.center_crop(buf, [self.T, self.T])     # no resampling at all
#         rotated = TF.affine(buf, angle=float(angle_deg), translate=[0, 0],
#                             scale=1.0, shear=[0.0],
#                             interpolation=self.interpolation)
#         return TF.center_crop(rotated, [self.T, self.T])

#     def all_views(self, pil_image, angles):
#         buf = self.buffer(pil_image)
#         return torch.stack([self.view(buf, a) for a in angles])   # (A, 3, T, T)


# ANGLES = list(range(-179, 181))        # 360 distinct rotations; 0 == ORIGINAL


class UtilsConformalPrediction:

    # @staticmethod
    # def compute_thr_calibration_value(thr_score_form, alpha):
    #     """
    #     Erstellt das Prediction Set basierend auf 1 - p_y.
    #     """
    #     # n = len(one_minus_probDist)
    #     # scores_cal = one_minus_probDist[np.arange(n), targets]
    #     q_level = np.min([1.0, np.ceil((n + 1) * (1 - alpha)) / n])

    #     calValue = np.quantile(thr_score_form, q_level, method='higher')

    #     return calValue
    
# import numpy as np
    @staticmethod
    def _rankdata_average_lastaxis(a, dtype=np.float64):
        """
        Vectorized equivalent of scipy.stats.rankdata(a, axis=-1, nan_policy='raise').
        Default method='average': ties get the mean of the ranks they span.
        Rank 1 = smallest value.
        """
        a = np.asarray(a)
        K = a.shape[-1]

        if np.isnan(a).any():
            raise ValueError("The input contains nan values")

        order = np.argsort(a, axis=-1, kind="stable")
        sv = np.take_along_axis(a, order, axis=-1)

        pos = np.arange(K, dtype=np.int32)

        # start[j] = True iff sorted position j begins a new run of equal values
        start = np.empty(a.shape, dtype=bool)
        start[..., 0] = True
        np.not_equal(sv[..., 1:], sv[..., :-1], out=start[..., 1:])
        del sv

        # lo[j] = sorted index of the first element of j's run
        lo = np.maximum.accumulate(np.where(start, pos, np.int32(0)), axis=-1)

        # end[j] = True iff sorted position j ends a run
        end = np.empty(a.shape, dtype=bool)
        end[..., :-1] = start[..., 1:]
        end[..., -1] = True
        del start

        # hi[j] = sorted index of the last element of j's run
        rev = np.flip(np.where(end, pos, np.int32(K)), axis=-1)
        hi = np.flip(np.minimum.accumulate(rev, axis=-1), axis=-1)
        del end, rev

        ranks_sorted = (lo + hi) * dtype(0.5) + dtype(1.0)
        out = np.empty(a.shape, dtype=dtype)
        np.put_along_axis(out, order, ranks_sorted, axis=-1)
        return out

    @staticmethod
    def collect_true_class_ranks(probabilities, targets, method_tie='average') -> np.ndarray:
        """
        Rank each sample's target class across K classes.

        Rank 1 corresponds to the highest probability. Ties receive their
        average rank.

        Parameters
        ----------
        probabilities : ndarray, shape (M, n_pre, K)
            Class probabilities for M versions.
        targets : ndarray, shape (n_pre,)
            Integer target-class index for each sample.

        Returns
        -------
        ndarray, shape (M, n_pre)
            Target-class ranks, potentially fractional because of ties.
        """
        
        from scipy.stats import rankdata
        
        # probabilities = self.p_views_pre
        # targets = self.targets_pre
        
        probabilities = np.asarray(probabilities)
        targets = np.asarray(targets)

        if probabilities.ndim != 3:
            raise ValueError("probabilities must have shape (M, n_pre, K)")

        n_pre = probabilities.shape[1]

        if targets.shape != (n_pre,):
            raise ValueError(f"targets must have shape ({n_pre},)")

        # Negation makes the highest probability receive rank 1.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
        # all_ranks = rankdata(
        #     -probabilities,
        #     axis=-1,
        #     method=method_tie,
        #     nan_policy='raise'
        # )
        all_ranks = UtilsConformalPrediction._rankdata_average_lastaxis(-probabilities)

        return all_ranks[:, np.arange(n_pre), targets]
    
    @staticmethod
    def get_ranked_data(probabilities, method_tie='average') -> np.ndarray:
        """
        Rank each sample's target class across K classes.

        Rank 1 corresponds to the highest probability. Ties receive their
        average rank.

        Parameters
        ----------
        probabilities : ndarray, shape (M, n_pre, K)
            Class probabilities for M versions.

        Returns
        -------
        ndarray, shape (M, n_pre)
            Target-class ranks, potentially fractional because of ties.
        """
        
        from scipy.stats import rankdata
        
        # probabilities = self.p_views_pre
        # targets = self.targets_pre
        
        probabilities = np.asarray(probabilities)

        if probabilities.ndim != 3:
            raise ValueError("probabilities must have shape (M, n_pre, K)")

        # # Negation makes the highest probability receive rank 1.
        # # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
        # all_ranks = rankdata(
        #     -probabilities,
        #     axis=-1,
        #     method=method_tie,
        #     nan_policy='raise'
        # )

        all_ranks = UtilsConformalPrediction._rankdata_average_lastaxis(-probabilities)

        return all_ranks
    
    @staticmethod
    def compute_calibration_value(scores, alpha):
        """Split-CP threshold: the ceil((n+1)(1-alpha))/n empirical quantile of the
        true-class nonconformity scores. Bounded scores -> 1.0 means 'all classes'."""
        scores = np.asarray(scores, dtype=np.float64).ravel()
        n = scores.shape[0]
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        return 1.0 if k > n else float(np.partition(scores, k - 1)[k - 1])
        # return max(scores) if k > n else float(np.partition(scores, k - 1)[k - 1])
    
    @staticmethod
    def compute_approximative_calibration_value_aps(probDist, targets, alpha):
        """
        APS-style calibration value.

        For each calibration sample i, the APS score is the cumulative probability
        mass of all classes ranked at least as high as the true class y_i:

            score_i = sum_{k: rank_i(k) <= rank_i(y_i)} p_i,k

        Smaller scores mean the true class appears early in the sorted probability list.
        """
        probDist = np.asarray(probDist, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.int64)

        n = len(probDist)

        # Sort classes by decreasing probability
        order = np.argsort(-probDist, axis=1, kind="stable")
        sorted_probs = np.take_along_axis(probDist, order, axis=1)

        # Cumulative probability mass in sorted order
        cumulative_probs = np.cumsum(sorted_probs, axis=1)

        # rank position of the true class in the sorted order
        inverse_order = np.empty_like(order)
        inverse_order[np.arange(n)[:, None], order] = np.arange(probDist.shape[1])[None, :]

        true_class_rank = inverse_order[np.arange(n), targets]

        # APS calibration scores
        scores_cal = cumulative_probs[np.arange(n), true_class_rank]

        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)

        calValue = np.quantile(scores_cal, q_level, method="higher")

        return calValue
    
class UtilsDataset:
    
    @staticmethod
    def generate_three_disjoint_split_idx(n_total, cal_fraction, test_fraction, precal_fraction, seed):
        """Single permutation -> three DISJOINT index arrays."""
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_total)
        n_pre = int(n_total * precal_fraction)
        n_cal = int(n_total * cal_fraction)
        n_test = int(n_total * test_fraction)
        return (perm[:n_cal],
                perm[n_cal:n_cal + n_test],
                perm[n_cal + n_test:n_cal + n_test + n_pre])    
        
    @staticmethod
    def generate_set_and_subset_split_idx(n_total, train_fraction, precal_fraction, seed):
        """Single permutation -> three DISJOINT index arrays."""
        assert train_fraction >= precal_fraction, "The following must hold: train_fraction >= precal_fraction"
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_total)
        n_precal = int(n_total * precal_fraction)
        n_train = int(n_total * train_fraction)
        # n_test = int(n_total * test_fraction)
        idx_train = perm[:n_train]
        return (idx_train, idx_train[:n_precal])
    
    @staticmethod
    def extract_subset_by_idx(dataset, subset_idx:list):
        import torch.utils.data as TA
        
        return TA.Subset(dataset, subset_idx)
    
    
class UtilsGeneral:


    @staticmethod
    def make_json_serializable(obj):
        """
        Convert common non-JSON objects into deterministic JSON-compatible values.
        """

        from enum import Enum
        from pathlib import Path

        if isinstance(obj, Enum):
            return obj.value

        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()

        if isinstance(obj, (np.integer,)):
            return int(obj)

        if isinstance(obj, (np.floating,)):
            return float(obj)

        if isinstance(obj, set):
            return sorted(UtilsGeneral.make_json_serializable(x) for x in obj)

        if isinstance(obj, dict):
            return {
                str(k): UtilsGeneral.make_json_serializable(v)
                for k, v in sorted(obj.items(), key=lambda item: str(item[0]))
            }

        if isinstance(obj, (list, tuple)):
            return [UtilsGeneral.make_json_serializable(x) for x in obj]

        return obj
    
    @staticmethod
    def hash_dict(d, length=16):
        """
        Deterministically hash a dictionary.

        If any key or value changes, the returned hash changes.
        """

        import json
        import hashlib

        serializable_d = UtilsGeneral.make_json_serializable(d)

        encoded = json.dumps(
            serializable_d,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

        full_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

        return full_hash[:length]

    @staticmethod
    def dict_items_to_string(dict_object: dict) -> str:
        string_content = "\n"
        for key, value in dict_object.items():
            string_content += f"{key}: {value}\n"
        string_content += "\n"
        return string_content
    
    @staticmethod
    def set_content_to_string(set_object):
        result = []

        for item in set_object:
            if isinstance(item, (tuple, list)):
                result.append(
                    " -> ".join(str(x) for x in item)
                )
                result.append("\n")
            else:
                result.append(str(item))

        return " ".join(result)

    @staticmethod
    def write_set_to_file(set_object, path):
        with open(f"{path}", "w") as f:
            for item in set_object:
                if isinstance(item, (tuple, list)):
                    f.write(", ".join(str(x) for x in item) + "\n")
                else:
                    f.write(str(item) + "\n")

    @staticmethod
    def read_bin_stat(path):
        data = np.loadtxt(path, delimiter=",", dtype=np.float64)
        return data

    @staticmethod
    def write_dict_to_file(dict_object, path):
        with open(f"{path}", "w") as f:
            for key, item in dict_object.items():
                f.write(f"{key}:\n")
                if isinstance(item, (tuple, list)):
                    f.write(", ".join(str(x) for x in item) + "\n")
                else:
                    f.write(str(item) + "\n")
    
    @staticmethod
    def write_string_to_file(content: str, path: str) -> None:
        """
        Create a file at `path` and write `content` into it.
        Parent directories are created if they do not exist.
        """

        from pathlib import Path

        file_path = Path(path)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8") as f:
            f.write(content)
            
    @staticmethod
    def write_prediction_log(content: str, path_to_file: str) -> None:
        """
        Create a file at `path` and write `content` into it.
        Parent directories are created if they do not exist.
        """

        from pathlib import Path
        import os

        file_path = Path(path_to_file)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("a", encoding="utf-8") as f:
            f.write(content)
            
    @staticmethod
    def compute_softmax_from_lastaxis(S):
        EPS = 1e-9
        S = np.asarray(S, dtype=np.float64)
        Z = S - np.max(S, axis=-1, keepdims=True); np.exp(Z, out=Z)
        Z /= np.clip(Z.sum(axis=-1, keepdims=True), EPS, None)
        return Z