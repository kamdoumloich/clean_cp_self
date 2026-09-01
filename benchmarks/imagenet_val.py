"""
ImageNet-val benchmark for the conformance prediction framework.

No training: a pretrained torchvision model (default ResNet-50,
IMAGENET1K_V1) is evaluated on the 50k validation images. The 50k images
are split DISJOINTLY into pre-calibration / calibration / test via a single
seeded permutation (unlike VisionClassificationBenchmark, which draws
independent splits from train and test sets; reusing that logic on a single
set would let precal overlap cal/test).

Preprocessing order (correctness-critical):
#   dataset transform = Resize(256) -> CenterCrop(224) -> ToTensor()  ([0,1])
  augmentations (contrast, hflip, ...) are applied on the [0,1] tensors,
  ImageNet mean/std NORMALIZATION IS APPLIED AFTER THE AUGMENTATION, just
  before the model. Normalizing first would silently corrupt adjust_contrast
  and violate the [0,1] assumption of adjust_hue.

Probability cache:
  Augmentations are deterministic, so the (View, 50000, 1000) softmax outputs
  are seed-INdependent. They are computed once and cached as a float32 .npz
  (~600 MB for 3 views) under <cache_dir>; every seed then only re-indexes
  the cache. Run seeds SEQUENTIALLY the first time (concurrent first runs
  would race on the cache build; the write itself is atomic via os.replace,
  so the race wastes compute but cannot corrupt the file).

"""

import json
import os

from utils.function_utils import UtilsDataset
from utils.function_utils import UtilsGeneral
from utils.function_utils import UtilsAugmentations
from torchvision.models import ResNet50_Weights

import utils.tta_policies

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetValBenchmark:

    benchmark_name = "ImageNet-val"

    def __init__(self, augmentations, path_logs, 
                 imagenet_root='/scratch/lkd18/data/imagenet', 
                 seed=42,
                 precalibration_fraction=0.3, 
                 calibration_fraction=0.3, 
                 test_fraction=0.4,
                 arch="resnet50", weights="IMAGENET1K_V1",
                 batch_size=128, 
                #  num_workers=4, 
                 cache_dir='/scratch/lkd18/data/imagenet/cache',
                 ):
        '''
        type_policy is either 'tta_policy' or 'custom_policy'
        '''
        
        if precalibration_fraction + calibration_fraction + test_fraction > 1.0 + 1e-9:
            raise ValueError("precal + cal + test fractions must be <= 1.")

        self.augmentations = dict(augmentations)
        self.list_augmentations_keys = list(self.augmentations.keys())
        if not self.list_augmentations_keys or self.list_augmentations_keys[0] != "ORIGINAL":
            raise ValueError("augmentations must start with 'ORIGINAL'.")

        self.seed = int(seed)
        self.rng = np.random.default_rng(seed=seed)
        self.path_logs = path_logs
        self.imagenet_root = imagenet_root
        self.arch = str(arch)
        # self.weights = str(weights)
        self.batch_size = int(batch_size)
        # self.num_workers = int(num_workers)
        self.precal_fraction = float(precalibration_fraction)
        self.cal_fraction = float(calibration_fraction)
        self.test_fraction = float(test_fraction)
        
        # self.cache_path_1 = "/scratch/lkd18/data/imagenet/cache/logits_resnet50_ResNet50_Weights.IMAGENET1K_V1_16f851f0346ab6e5.npz"
        # self.cache_path_2 = "/scratch/lkd18/data/imagenet/cache/logits_resnet50_ResNet50_Weights.IMAGENET1K_V1_ae2f9f4a9715ba6f.npz"
        # self.cache_path = "/scratch/lkd18/data/imagenet/cache/cache_merged.npz"
        
        # https://docs.pytorch.org/vision/stable/models.html?highlight=randint
        if weights == "IMAGENET1K_V1":
            # self.preprocess_transform = ResNet50_Weights.IMAGENET1K_V1.transforms()
            self.weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            raise NotImplementedError

        os.makedirs(self.path_logs, exist_ok=True)
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.normpath(self.path_logs)),
                "imagenet_prob_cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)


        ##############################################
        ############## VERY IMPORTANT ################
        aug_spec = "_".join(f"{k}{v}" for k, v in self.augmentations.items())

        hash_of_dict = UtilsGeneral.hash_dict(d=self.augmentations)
        path_to_descrition_file = os.path.join(
            self.cache_dir, f"description_{self.arch}_{self.weights}_{hash_of_dict}.txt")

        UtilsGeneral.write_string_to_file(content=aug_spec, path=path_to_descrition_file)

        ##############################################


        self.cache_path = os.path.join(
            self.cache_dir, f"logits_{self.arch}_{self.weights}_{hash_of_dict}.npz")

        logits, labels, self.classes = self._load_or_build_cache()
        V, N, K = logits.shape
        self.n_classes = int(K)

        idx_cal, idx_test, idx_pre = UtilsDataset.generate_three_disjoint_split_idx(
            n_total=N, cal_fraction=self.cal_fraction, test_fraction=self.test_fraction,
            precal_fraction=self.precal_fraction, seed=self.seed,
            )
        
        print(f"[imagenet] split sizes: precal={len(idx_pre)}, "
              f"cal={len(idx_cal)}, test={len(idx_test)} (of {N})",
              flush=True)

        self.precalibration_data = self._get_entries_by_idx(logits, labels, idx_pre)
        self.calibration_data = self._get_entries_by_idx(logits, labels, idx_cal)
        self.testing_data = self._get_entries_by_idx(logits, labels, idx_test)

        # # per-view top-1 accuracy on the test split (no training happens)
        # self.accuracies = {"Training": [], "Testing": []}
        # self.class_accuracies = {"Training": {}, "Testing": {}}
        # for v, aug_name in enumerate(self.list_augmentations_keys):
        #     acc = float((logits[v, idx_test].argmax(axis=1) == labels[idx_test]).mean() * 100.0)
        #     self.accuracies["Testing"].append(round(acc, 2))
        #     print(f"[imagenet] top-1 accuracy on test split ({aug_name}): {acc:.2f}%", flush=True)

    def _can_evaluate_without_training(self):
        return True


    @staticmethod
    def _get_entries_by_idx(logits, labels, idx):
        V = logits.shape[0]
        return [([logits[v, i] for v in range(V)], int(labels[i])) for i in idx]

    # ------------------------- cache build / load -------------------------
    
    # def _load_cache_file(self, cache_path):
    #     classes_path = cache_path + ".classes.json"

    #     if not os.path.isfile(cache_path):
    #         raise FileNotFoundError(f"Cache file not found: {cache_path}")

    #     with np.load(cache_path, allow_pickle=False) as z:
    #         logits = z["logits"]
    #         labels = z["labels"]
    #         augs = [str(a) for a in z["augs"]]

    #     classes = (
    #         tuple(json.load(open(classes_path, "r", encoding="utf-8")))
    #         if os.path.isfile(classes_path)
    #         else tuple(str(i) for i in range(logits.shape[2]))
    #     )

    #     return {
    #         "logits": logits,
    #         "labels": labels,
    #         "augs": augs,
    #         "classes": classes,
    #     }


    # def _merge_two_caches(self, cache_path_1, cache_path_2, output_path):
    #     cache_1 = self._load_cache_file(cache_path_1)
    #     cache_2 = self._load_cache_file(cache_path_2)
        
    #     from utils.function_utils import Augmentation
        
        
        
    #     list_rotation_360 = UtilsAugmentations.get_n_evenly_spaced_values_for_augmentation(augmentation_name=Augmentation.ROTATE, lb_degree_aug=-10, ub_degree_aug=10, n_amount=21)

    #     dict_aug = {
    #         "ORIGINAL": [0],
    #         # "RANDOM_CROP": [4],
    #         # "HORIZONTAL_FLIP": [True],
    #         # "ROTATE": list_rotation_360,
    #     }
        
    #     for degree in list_rotation_360:
    #         string = "ROTATE_"+str(degree)
    #         dict_aug[string] =  [degree]
            
    #     augs_2 = dict_aug

    #     logits_1 = cache_1["logits"]
    #     logits_2 = cache_2["logits"]

    #     labels_1 = cache_1["labels"]
    #     labels_2 = cache_2["labels"]

    #     augs_1 = cache_1["augs"]
    #     # augs_2 = cache_2["augs"]

    #     classes_1 = cache_1["classes"]
    #     classes_2 = cache_2["classes"]

    #     if logits_1.ndim != 3 or logits_2.ndim != 3:
    #         raise ValueError(
    #             f"Expected logits with shape (V, N, K), got "
    #             f"{logits_1.shape} and {logits_2.shape}."
    #         )

    #     if logits_1.shape[1:] != logits_2.shape[1:]:
    #         raise ValueError(
    #             f"Cannot merge caches with different (N, K): "
    #             f"{logits_1.shape[1:]} vs {logits_2.shape[1:]}."
    #         )

    #     if not np.array_equal(labels_1, labels_2):
    #         raise ValueError("Cannot merge caches: labels are different.")

    #     if classes_1 != classes_2:
    #         raise ValueError("Cannot merge caches: classes are different.")

    #     if len(augs_1) != logits_1.shape[0]:
    #         raise ValueError(
    #             f"Cache 1 has {len(augs_1)} augmentation names but "
    #             f"{logits_1.shape[0]} logit views."
    #         )

    #     if len(augs_2) != logits_2.shape[0]:
    #         raise ValueError(
    #             f"Cache 2 has {len(augs_2)} augmentation names but "
    #             f"{logits_2.shape[0]} logit views."
    #         )

    #     merged_logits = [logits_1[i] for i in range(len(augs_1))]
    #     merged_augs = list(augs_1)

    #     seen = set(merged_augs)

    #     skipped = []
    #     added = []

    #     for i, aug_name in enumerate(augs_2):
    #         if aug_name in seen:
    #             skipped.append(aug_name)
    #             continue

    #         merged_logits.append(logits_2[i])
    #         merged_augs.append(aug_name)
    #         seen.add(aug_name)
    #         added.append(aug_name)

    #     merged_logits = np.stack(merged_logits, axis=0)

    #     tmp = output_path + ".tmp"

    #     with open(tmp, "wb") as f:
    #         np.savez(
    #             f,
    #             logits=merged_logits,
    #             labels=labels_1,
    #             augs=np.array(merged_augs),
    #         )

    #     os.replace(tmp, output_path)

    #     output_classes_path = output_path + ".classes.json"
    #     with open(output_classes_path, "w", encoding="utf-8") as f:
    #         json.dump(list(classes_1), f)

    #     print(
    #         f"[imagenet] merged cache written to {output_path} "
    #         f"({os.path.getsize(output_path) / 1e9:.2f} GB)",
    #         flush=True,
    #     )
    #     print(f"[imagenet] added views: {added}", flush=True)
    #     print(f"[imagenet] skipped duplicate views: {skipped}", flush=True)

    #     return merged_logits, labels_1, classes_1
    
    
    # def _load_or_build_cache(self):
    #     classes_path = self.cache_path + ".classes.json"

    #     if hasattr(self, "cache_path_1") and hasattr(self, "cache_path_2"):
    #         if os.path.isfile(self.cache_path):
    #             print(f"[imagenet] loading merged cache {self.cache_path}", flush=True)

    #             with np.load(self.cache_path, allow_pickle=False) as z:
    #                 logits = z["logits"]
    #                 labels = z["labels"]
    #                 cached_augs = [str(a) for a in z["augs"]]

    #             classes = (
    #                 tuple(json.load(open(classes_path, "r", encoding="utf-8")))
    #                 if os.path.isfile(classes_path)
    #                 else tuple(str(i) for i in range(logits.shape[2]))
    #             )

    #             return logits, labels, classes

    #         return self._merge_two_caches(
    #             cache_path_1=self.cache_path_1,
    #             cache_path_2=self.cache_path_2,
    #             output_path=self.cache_path,
    #         )

    #     if os.path.isfile(self.cache_path):
    #         print(f"[imagenet] loading probability cache {self.cache_path}", flush=True)

    #         with np.load(self.cache_path, allow_pickle=False) as z:
    #             logits = z["logits"]
    #             labels = z["labels"]
    #             cached_augs = [str(a) for a in z["augs"]]

    #         if cached_augs != self.list_augmentations:
    #             raise ValueError(
    #                 f"Cache {self.cache_path} was built for views "
    #                 f"{cached_augs}, requested {self.list_augmentations}. "
    #                 "Delete the cache or change the augmentation dict."
    #             )

    #         classes = (
    #             tuple(json.load(open(classes_path, "r", encoding="utf-8")))
    #             if os.path.isfile(classes_path)
    #             else tuple(str(i) for i in range(logits.shape[2]))
    #         )

    #         return logits, labels, classes

    #     print(
    #         "[imagenet] probability cache not found; running one full "
    #         "forward pass over val.",
    #         flush=True,
    #     )

    #     logits, labels, classes = self._forward_full_val()

    #     tmp = self.cache_path + ".tmp"

    #     with open(tmp, "wb") as f:
    #         np.savez(
    #             f,
    #             logits=logits,
    #             labels=labels,
    #             augs=np.array(self.list_augmentations),
    #         )

    #     os.replace(tmp, self.cache_path)

    #     with open(classes_path, "w", encoding="utf-8") as f:
    #         json.dump(list(classes), f)

    #     print(
    #         f"[imagenet] cache written to {self.cache_path} "
    #         f"({os.path.getsize(self.cache_path) / 1e9:.2f} GB)",
    #         flush=True,
    #     )

    #     return logits, labels, classes
    
    
    #*****************************************************************************

    def _load_or_build_cache(self):
        classes_path = self.cache_path + ".classes.json"
        if os.path.isfile(self.cache_path):
            print(f"[imagenet] loading probability cache {self.cache_path}",
                  flush=True)
            with np.load(self.cache_path) as z:
                logits = z["logits"]
                labels = z["labels"]
                cached_augs = [str(a) for a in z["augs"]]
            if cached_augs != self.list_augmentations_keys:
                raise ValueError(
                    f"Cache {self.cache_path} was built for views "
                    f"{cached_augs}, requested {self.list_augmentations_keys}. "
                    "Delete the cache or change the augmentation dict.")
            classes = tuple(json.load(open(classes_path, "r",
                                           encoding="utf-8"))) \
                if os.path.isfile(classes_path) else tuple(
                    str(i) for i in range(logits.shape[2]))
            return logits, labels, classes

        print("[imagenet] probability cache not found; running one full "
              "forward pass over val (this is the expensive step and is "
              "done ONCE across all seeds).", flush=True)
        logits, labels, classes = self._forward_full_val()
        tmp = self.cache_path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, logits=logits, labels=labels,
                     augs=np.array(self.list_augmentations_keys))
        os.replace(tmp, self.cache_path)
        with open(classes_path, "w", encoding="utf-8") as f:
            json.dump(list(classes), f)
        print(f"[imagenet] cache written to {self.cache_path} "
              f"({os.path.getsize(self.cache_path) / 1e9:.2f} GB)",
              flush=True)
        return logits, labels, classes

    def _forward_full_val(self):
        import torch
        import torchvision

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                    torchvision.transforms.CenterCrop(224),
                    torchvision.transforms.ToTensor(),
            ])
            val_dataset = torchvision.datasets.ImageNet(
                root=self.imagenet_root, split="val", 
                transform=transform
                )
        except Exception as exc:
            raise RuntimeError(
                f"Could not open ImageNet val at {self.imagenet_root}. Run "
                "prepare_data.py --imagenet_root <path> first (the val tar "
                "must be downloaded manually from image-net.org)."
            ) from exc

        classes = self._normalize_classes(val_dataset.classes)
        model = torchvision.models.get_model(self.arch, weights=self.weights).to(device)
        model.eval()

        loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            # num_workers=self.num_workers, 
            pin_memory=(device == "cuda"))

        view_specs = [
            (aug_name, aug_degree)
            for aug_name in self.list_augmentations_keys
            for aug_degree in self.augmentations[aug_name]
        ]
        V = len(view_specs)
        N = len(val_dataset)
        logits = None
        labels = np.empty(N, dtype=np.int64)
        pos = 0

        with torch.no_grad():
            for data, target in loader:
                data = data.to(device, non_blocking=True)
                B = data.shape[0]

                for v, (aug_name, aug_degree) in enumerate(view_specs):
                    view = UtilsAugmentations.apply_single_augmentation(
                        data, name=aug_name, rng=self.seed, degree=aug_degree,
                    )
                    out = model(self._prepare_model_inputs(view))       # [B, K]
                    out = out.cpu().numpy().astype(np.float32)
                    if logits is None:
                        logits = np.empty((V, N, out.shape[1]), dtype=np.float32)
                    logits[v, pos:pos + B] = out

                labels[pos:pos + B] = target.numpy()
                pos += B
                if (pos // self.batch_size) % 20 == 0:
                    print(f"[imagenet] forward {pos}/{N}", flush=True)
        assert pos == N
        return logits, labels, classes

    # --------------------- preprocessing & augmentation ---------------------

    # @staticmethod
    def _prepare_model_inputs(self, data):
        import torchvision
        return torchvision.transforms.functional.normalize(data, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))


    @staticmethod
    def _normalize_classes(classes):
        out = []
        for c in classes:
            if isinstance(c, (tuple, list)):
                out.append(", ".join(str(x) for x in c))
            else:
                out.append(str(c))
        return tuple(out)

    # ------------------------------ reporting ------------------------------

    def texInfo(self):
        accs = " / ".join(
            f"{name}: {acc}\\%" for name, acc in
            zip(self.list_augmentations_keys, self.accuracies["Testing"]))
        return ("""
        \\begin{tabular}{l|l} \\
        \\textbf{Benchmark name:} & ImageNet-val \\\\ \\hline
        \\textbf{Model:} & """ + f"{self.arch} ({self.weights}, pretrained, "
                "no training)" + """ \\\\ \\hline
        \\textbf{Augmentations:} & """ +
                str(list(self.augmentations.values())) + """ \\\\ \\hline
        \\textbf{Top-1 accuracy (test split):} & """ + accs + """ \\
        \\end{tabular}
        (Per-class tables omitted: 1000 classes.)
        """)
