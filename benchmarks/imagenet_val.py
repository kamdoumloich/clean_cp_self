"""
ImageNet-val benchmark for the conformal prediction framework.

No training: a pretrained torchvision model (default ResNet-50,
IMAGENET1K_V1) is evaluated on the 50k validation images. The 50k images
are split DISJOINTLY into pre-calibration / calibration / test via a single
seeded permutation.

Preprocessing order (correctness-critical):
  1. Base transforms: Resize(256) -> CenterCrop(224) -> ToTensor() ([0, 1])
  2. Augmentations (rotation, flip, etc.) applied on [0, 1] tensors
  3. ImageNet normalization (mean/std) applied AFTER augmentation, directly before model.

Cache Architecture:
  Augmentations are deterministic, so the (50000, 1000) logit outputs are
  seed-independent. They are cached as float32 .npz files per augmentation view
  under <cache_dir>. Loading uses memory-mapped slices directly into the disjoint
  splits, preventing multi-gigabyte memory duplication.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torchvision
from torchvision.models import ResNet50_Weights

from utils.function_utils import UtilsAugmentations, UtilsDataset, UtilsGeneral
import utils.tta_policies

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetValBenchmark:
    benchmark_name = "ImageNet-val"

    def __init__(
        self,
        policy_name: Optional[str] = "original_only_policy",
        path_logs: Optional[str] = None,
        list_other_aug: Optional[Dict[str, Any]] = None,
        augmentations: Optional[Dict[str, Any]] = None,
        imagenet_root: str = "/scratch/lkd18/data/imagenet",
        seed: int = 42,
        precalibration_fraction: float = 0.1,
        calibration_fraction: float = 0.5,
        test_fraction: float = 0.4,
        arch: str = "resnet50",
        weights: str = "IMAGENET1K_V1",
        batch_size: int = 128,
        cache_dir: str = "/scratch/lkd18/data/imagenet/cache",
        expected_max_rotation: Optional[int] = None,
    ):
        # Support legacy signature where augmentations dictionary was passed first
        if isinstance(policy_name, dict):
            augmentations = policy_name
            policy_name = None

        if precalibration_fraction + calibration_fraction + test_fraction > 1.0 + 1e-9:
            raise ValueError("precal + cal + test fractions must sum to <= 1.0.")

        resolved_augmentations = {}
        if policy_name is not None:
            resolved_augmentations.update(
                utils.tta_policies.build_augs_dict(policy=policy_name)
            )
        if augmentations is not None:
            resolved_augmentations.update(augmentations)
        if list_other_aug is not None:
            resolved_augmentations.update(list_other_aug)

        self.augmentations = resolved_augmentations
        self.list_augmentation_names = list(self.augmentations.keys())
        self.list_augmentations_keys = self.list_augmentation_names

        if not self.list_augmentation_names or self.list_augmentation_names[0] != "ORIGINAL":
            raise ValueError("Augmentations must start with 'ORIGINAL'.")

        self.expected_max_rotation = expected_max_rotation
        if expected_max_rotation is not None:
            self._validate_rotation_view_order(expected_max_rotation)

        self.seed = int(seed)
        self.rng = np.random.default_rng(seed=seed)
        self.path_logs = path_logs or "./logs/imagenetVal"
        self.imagenet_root = imagenet_root
        self.arch = str(arch)
        self.batch_size = int(batch_size)
        self.precal_fraction = float(precalibration_fraction)
        self.cal_fraction = float(calibration_fraction)
        self.test_fraction = float(test_fraction)

        if weights == "IMAGENET1K_V1":
            self.weights = ResNet50_Weights.IMAGENET1K_V1
        elif isinstance(weights, ResNet50_Weights):
            self.weights = weights
        else:
            raise NotImplementedError(f"Unsupported weights: {weights}")

        os.makedirs(self.path_logs, exist_ok=True)
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.normpath(self.path_logs)), "imagenet_prob_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        # Base cache template for per-augmentation .npz files
        self.cache_path = os.path.join(
            self.cache_dir, f"logits_{self.arch}_{self.weights}_.npz"
        )

        # Load data directly into disjoint splits
        self._load_and_build_splits()

    def _validate_rotation_view_order(self, max_rotation: int) -> np.ndarray:
        """Verify that rotation probes match [-max_rot..-1, 1..max_rot] in exact order."""
        expected_angles = list(range(-max_rotation, 0)) + list(range(1, max_rotation + 1))
        actual_angles = []

        for name in self.list_augmentation_names[1:]:
            if not str(name).startswith("ROTATE_PROBE_"):
                continue
            spec = self.augmentations[name]
            deg = float(spec["degree"])
            if not deg.is_integer():
                raise ValueError(f"Rotation degree must be an integer, got {deg}.")
            actual_angles.append(int(deg))

        if len(actual_angles) == len(expected_angles) and actual_angles != expected_angles:
            raise ValueError(
                f"Incorrect rotation-view order.\n"
                f"Expected: {expected_angles}\n"
                f"Received: {actual_angles}"
            )
        return np.asarray([0] + actual_angles, dtype=np.int64)

    def _get_cache_path_for_aug(self, aug_name: str) -> str:
        """Finds the cache file path for a given augmentation, checking common float/int formats."""
        root, ext = os.path.splitext(self.cache_path)
        safe_aug = str(aug_name).replace("/", "_").replace("\\", "_")
        cand1 = f"{root}.{safe_aug}{ext}"
        if os.path.isfile(cand1):
            return cand1

        # Check alternative formats (e.g. without trailing underscore in root, or .0 on degrees)
        root_no_under = root.rstrip("_")
        cand2 = f"{root_no_under}_{safe_aug}{ext}"
        if os.path.isfile(cand2):
            return cand2

        if safe_aug.startswith("ROTATE_PROBE_"):
            deg_str = safe_aug.replace("ROTATE_PROBE_", "")
            try:
                deg_f = float(deg_str)
                # Try with .0
                alt_aug = f"ROTATE_PROBE_{deg_f:.1f}"
                cand3 = f"{root}.{alt_aug}{ext}"
                if os.path.isfile(cand3):
                    return cand3
                # Try without .0
                alt_aug_int = f"ROTATE_PROBE_{int(deg_f)}"
                cand4 = f"{root}.{alt_aug_int}{ext}"
                if os.path.isfile(cand4):
                    return cand4
            except ValueError:
                pass

        return cand1

    def _load_and_build_splits(self):
        """Loads logits per view directly into disjoint precal, cal, and test splits."""
        missing_augs = []
        cache_paths = {}

        for aug_name in self.list_augmentation_names:
            path = self._get_cache_path_for_aug(aug_name)
            if not os.path.isfile(path):
                missing_augs.append(aug_name)
            else:
                cache_paths[aug_name] = path

        if missing_augs:
            print(
                f"[imagenet] missing caches for {len(missing_augs)} view(s): {missing_augs[:5]}... "
                "Running forward pass over validation set.",
                flush=True,
            )
            self._generate_missing_caches(missing_augs)
            for aug_name in missing_augs:
                cache_paths[aug_name] = self._get_cache_path_for_aug(aug_name)

        # Inspect first view to determine sample count N, class count K, and labels
        first_aug = self.list_augmentation_names[0]
        first_path = cache_paths[first_aug]
        first_classes_path = first_path + ".classes.json"

        with np.load(first_path, mmap_mode="r") as z:
            first_logits = z["logits"]
            labels = np.asarray(z["labels"], dtype=np.int64)

        if first_logits.ndim == 2:
            first_logits = first_logits[None, :, :]
        _, N, K = first_logits.shape
        self.n_classes = int(K)

        if os.path.isfile(first_classes_path):
            with open(first_classes_path, "r", encoding="utf-8") as f:
                self.classes = tuple(json.load(f))
        else:
            self.classes = tuple(str(i) for i in range(K))

        # Generate disjoint split indices
        idx_cal, idx_test, idx_pre = UtilsDataset.generate_three_disjoint_split_idx(
            n_total=N,
            cal_fraction=self.cal_fraction,
            test_fraction=self.test_fraction,
            precal_fraction=self.precal_fraction,
            seed=self.seed,
        )

        print(
            f"[imagenet] split sizes: precal={len(idx_pre)}, "
            f"cal={len(idx_cal)}, test={len(idx_test)} (of {N})",
            flush=True,
        )

        V = len(self.list_augmentation_names)
        pre_views = []
        cal_views = []
        test_views = []

        # Load each view with mmap and slice directly to avoid 140+ GB memory copies
        for aug_name in self.list_augmentation_names:
            aug_path = cache_paths[aug_name]
            with np.load(aug_path, mmap_mode="r") as z:
                aug_logits = z["logits"]
                if aug_logits.ndim == 2:
                    aug_logits = aug_logits[None, :, :]
                pre_views.append(np.array(aug_logits[0, idx_pre, :], dtype=np.float32))
                cal_views.append(np.array(aug_logits[0, idx_cal, :], dtype=np.float32))
                test_views.append(np.array(aug_logits[0, idx_test, :], dtype=np.float32))

        pre_arr = np.stack(pre_views, axis=0)
        cal_arr = np.stack(cal_views, axis=0)
        test_arr = np.stack(test_views, axis=0)

        self.precalibration_data = (pre_arr, labels[idx_pre])
        self.calibration_data = (cal_arr, labels[idx_cal])
        self.testing_data = (test_arr, labels[idx_test])

        # Record accuracies
        self.accuracies = {"Training": [], "Testing": []}
        self.class_accuracies = {"Training": {}, "Testing": {}}

        acc_orig = float((test_arr[0].argmax(axis=1) == labels[idx_test]).mean() * 100.0)
        self.accuracies["Testing"].append(round(acc_orig, 2))
        print(f"[imagenet] top-1 accuracy on test split (ORIGINAL): {acc_orig:.2f}%", flush=True)

        # Log brief summary instead of printing 359 lines
        if V > 1:
            all_accs = [
                float((test_arr[v].argmax(axis=1) == labels[idx_test]).mean() * 100.0)
                for v in range(V)
            ]
            self.accuracies["Testing"] = [round(a, 2) for a in all_accs]
            if V <= 10:
                for v in range(1, V):
                    print(
                        f"[imagenet] top-1 accuracy on test split ({self.list_augmentation_names[v]}): {all_accs[v]:.2f}%",
                        flush=True,
                    )
            else:
                rot_accs = all_accs[1:]
                print(
                    f"[imagenet] {V - 1} rotation probe views loaded: "
                    f"test accuracy range [{min(rot_accs):.2f}% .. {max(rot_accs):.2f}%] (mean {np.mean(rot_accs):.2f}%)",
                    flush=True,
                )

    def _generate_missing_caches(self, missing_augs: List[str]):
        """Generates missing views one by one or in safe sub-batches to prevent CUDA OOM."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
        ])

        try:
            val_dataset = torchvision.datasets.ImageNet(
                root=self.imagenet_root, split="val", transform=transform
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not open ImageNet val at {self.imagenet_root}. Ensure ImageNet validation set is prepared."
            ) from exc

        classes = self._normalize_classes(val_dataset.classes)
        model = torchvision.models.get_model(self.arch, weights=self.weights).to(device)
        model.eval()

        loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=(device == "cuda")
        )

        N = len(val_dataset)
        labels = np.empty(N, dtype=np.int64)

        for aug_name in missing_augs:
            print(f"[imagenet] computing forward pass for missing view '{aug_name}'...", flush=True)
            aug_cache_path = self._get_cache_path_for_aug(aug_name)
            classes_path = aug_cache_path + ".classes.json"
            view_logits = np.empty((1, N, 1000), dtype=np.float32)
            pos = 0

            with torch.no_grad():
                for data, target in loader:
                    data = data.to(device, non_blocking=True)
                    B = data.shape[0]

                    if aug_name == "ORIGINAL":
                        view = data
                    elif aug_name.startswith("ROTATE_PROBE_"):
                        deg = float(self.augmentations[aug_name]["degree"])
                        view = UtilsAugmentations.apply_single_augmentation(
                            data, "ROTATE", rng=None, degree=deg
                        )
                    else:
                        view = utils.tta_policies.apply_tta_aug_policy(data, augmentation_name=aug_name)

                    out = model(self._prepare_model_inputs(view))
                    view_logits[0, pos : pos + B] = out.cpu().numpy().astype(np.float32)
                    labels[pos : pos + B] = target.numpy()
                    pos += B

            assert pos == N
            tmp = aug_cache_path + ".tmp"
            with open(tmp, "wb") as f:
                np.savez(f, logits=view_logits, labels=labels, aug=np.array(str(aug_name)))
            os.replace(tmp, aug_cache_path)

            with open(classes_path, "w", encoding="utf-8") as f:
                json.dump(list(classes), f)
            print(f"[imagenet] wrote cache: {aug_cache_path}", flush=True)

    def _prepare_model_inputs(self, data: torch.Tensor) -> torch.Tensor:
        return torchvision.transforms.functional.normalize(
            data, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)
        )

    @staticmethod
    def _normalize_classes(classes) -> Tuple[str, ...]:
        out = []
        for c in classes:
            if isinstance(c, (tuple, list)):
                out.append(", ".join(str(x) for x in c))
            else:
                out.append(str(c))
        return tuple(out)

    def _can_evaluate_without_training(self) -> bool:
        return True

    def texInfo(self) -> str:
        acc_orig = self.accuracies["Testing"][0] if self.accuracies["Testing"] else "N/A"
        return f"""
        \\begin{{tabular}}{{l|l}} \\
        \\textbf{{Benchmark name:}} & ImageNet-val \\\\ \\hline
        \\textbf{{Model:}} & {self.arch} ({self.weights}, pretrained, no training) \\\\ \\hline
        \\textbf{{Number of views:}} & {len(self.list_augmentation_names)} \\\\ \\hline
        \\textbf{{Top-1 accuracy (test split, ORIGINAL):}} & {acc_orig}\\% \\\\
        \\end{{tabular}}
        (Per-class tables omitted: 1000 classes.)
        """
