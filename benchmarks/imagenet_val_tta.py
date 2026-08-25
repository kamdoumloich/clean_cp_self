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
from torchvision.models import ResNet50_Weights

import utils.tta_policies
from utils.function_utils import UtilsAugmentations

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetValBenchmarkTTA:

    benchmark_name = "ImageNet-val"

    def __init__(self, policy_name, path_logs, list_other_aug=None, 
                 imagenet_root='/scratch/lkd18/data/imagenet', 
                 seed=42,
                 precalibration_fraction=0.1, 
                 calibration_fraction=0.5, 
                 test_fraction=0.4,
                 arch="resnet50", weights="IMAGENET1K_V1",
                 batch_size=128,
                 cache_dir='/scratch/lkd18/data/imagenet/cache',
                 ):
        '''
        type_policy is either 'tta_policy' or 'custom_policy'
        '''
        
        if precalibration_fraction + calibration_fraction + test_fraction > 1.0 + 1e-9:
            raise ValueError("precal + cal + test fractions must be <= 1.")

        augmentations = utils.tta_policies.build_augs_dict(policy=policy_name)
        self.augmentations = dict(augmentations)
        
        if not list_other_aug is None:
            self.augmentations.update(list_other_aug)
            
        self.list_augmentation_names = list(self.augmentations.keys())
        if not self.list_augmentation_names or self.list_augmentation_names[0] != "ORIGINAL":
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


        # self.cache_path = os.path.join(
        #     self.cache_dir, f"logits_{self.arch}_{self.weights}_{hash_of_dict}.npz")
        self.cache_path = os.path.join(
            self.cache_dir, f"logits_{self.arch}_{self.weights}_.npz")

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

        # per-view top-1 accuracy on the test split (no training happens)
        self.accuracies = {"Training": [], "Testing": []}
        self.class_accuracies = {"Training": {}, "Testing": {}}
        for v, aug_name in enumerate(self.list_augmentation_names):
            acc = float((logits[v, idx_test].argmax(axis=1) == labels[idx_test]).mean() * 100.0)
            self.accuracies["Testing"].append(round(acc, 2))
            print(f"[imagenet] top-1 accuracy on test split ({aug_name}): {acc:.2f}%", flush=True)

    def _can_evaluate_without_training(self):
        return True


    @staticmethod
    def _get_entries_by_idx(logits, labels, idx):
        V = logits.shape[0]
        return [([logits[v, i] for v in range(V)], int(labels[i])) for i in idx]

    # ------------------------- cache build / load -------------------------
    
    def _load_or_build_cache(self):
        """
        Keep one cache per augmentation.

        Example:
            base cache_path:
                /cache/imagenet_probs.npz

            augmentations:
                ["original", "flip", "crop"]

            generated files:
                /cache/imagenet_probs.original.npz
                /cache/imagenet_probs.flip.npz
                /cache/imagenet_probs.crop.npz

        When several augmentations are requested, their caches are loaded
        independently and concatenated along axis=1.
        """

        def cache_path_for_aug(aug_name):
            root, ext = os.path.splitext(self.cache_path)
            safe_aug = str(aug_name).replace("/", "_").replace("\\", "_")
            return f"{root}.{safe_aug}{ext}"

        all_logits = []
        labels = None
        classes = None

        missing_augs = []

        # ------------------------------------------------------------
        # 1. Load every augmentation that already has a cache
        # ------------------------------------------------------------
        for aug_name in self.list_augmentation_names:
            aug_cache_path = cache_path_for_aug(aug_name)
            classes_path = aug_cache_path + ".classes.json"

            if not os.path.isfile(aug_cache_path):
                missing_augs.append(aug_name)
                continue

            print(
                f"[imagenet] loading probability cache "
                f"for augmentation '{aug_name}': {aug_cache_path}",
                flush=True,
            )

            with np.load(aug_cache_path) as z:
                aug_logits = z["logits"]
                aug_labels = z["labels"]

            # Make sure augmentation dimension exists.
            # Expected final shape: [N, 1, C]
            if aug_logits.ndim == 2:
                aug_logits = aug_logits[None, :, :]

            if aug_logits.shape[0] != 1:
                raise ValueError(
                    f"Cache {aug_cache_path} should contain exactly one "
                    f"augmentation, but logits shape is {aug_logits.shape}."
                )

            if labels is None:
                labels = aug_labels
            elif not np.array_equal(labels, aug_labels):
                raise ValueError(
                    f"Labels in cache {aug_cache_path} do not match "
                    "the other augmentation caches."
                )

            if classes is None:
                if os.path.isfile(classes_path):
                    with open(classes_path, "r", encoding="utf-8") as f:
                        classes = tuple(json.load(f))
                else:
                    classes = tuple(
                        str(i) for i in range(aug_logits.shape[-1])
                    )

            all_logits.append((aug_name, aug_logits))

        # ------------------------------------------------------------
        # 2. Build caches that are missing
        # ------------------------------------------------------------
        if missing_augs:
            print(
                "[imagenet] caches missing for augmentations "
                f"{missing_augs}; running forward pass.",
                flush=True,
            )

            # Temporarily run only the missing augmentations.
            original_augs = self.list_augmentation_names

            try:
                self.list_augmentation_names = missing_augs
                new_logits, new_labels, new_classes = self._forward_full_val()
            finally:
                self.list_augmentation_names = original_augs

            if new_logits.ndim == 2:
                new_logits = new_logits[None, :, :]

            if new_logits.shape[0] != len(missing_augs):
                raise ValueError(
                    "_forward_full_val() returned logits with shape "
                    f"{new_logits.shape}, but {len(missing_augs)} "
                    "augmentations were requested."
                )

            if labels is None:
                labels = new_labels
            elif not np.array_equal(labels, new_labels):
                raise ValueError(
                    "Labels returned by _forward_full_val() do not match "
                    "the labels loaded from existing caches."
                )

            if classes is None:
                classes = tuple(new_classes)

            # Save each augmentation separately
            for aug_idx, aug_name in enumerate(missing_augs):
                aug_cache_path = cache_path_for_aug(aug_name)
                classes_path = aug_cache_path + ".classes.json"

                # Keep augmentation dimension:
                # [N, 1, C]
                aug_logits = new_logits[aug_idx:aug_idx + 1, :, :]
                # aug_logits = new_logits[:, aug_idx:aug_idx + 1, :]

                tmp = aug_cache_path + ".tmp"

                with open(tmp, "wb") as f:
                    np.savez(
                        f,
                        logits=aug_logits,
                        labels=new_labels,
                        aug=np.array(str(aug_name)),
                    )

                os.replace(tmp, aug_cache_path)

                with open(classes_path, "w", encoding="utf-8") as f:
                    json.dump(list(new_classes), f)

                print(
                    f"[imagenet] cache written for augmentation "
                    f"'{aug_name}' to {aug_cache_path} "
                    f"({os.path.getsize(aug_cache_path) / 1e9:.2f} GB)",
                    flush=True,
                )

                all_logits.append((aug_name, aug_logits))

        # ------------------------------------------------------------
        # 3. Restore requested augmentation ordering
        # ------------------------------------------------------------
        logits_by_aug = {aug_name: aug_logits for aug_name, aug_logits in all_logits}
        ordered_logits = [logits_by_aug[aug_name] for aug_name in self.list_augmentation_names]
        
        logits = np.concatenate(ordered_logits, axis=0)

        return logits, labels, classes
    
    # Nur als Backup Methode da
    def _strip_cached_view(self, cache_path, view_name="ROTATE_PROBE_0.0"):
        """Drop one view from a logit/probability cache and write a new cache.
    
        Expects the layout produced by ``_load_or_build_cache``:
            logits : (A, N, K) float array, axis 0 indexed by ``augs``
            labels : (N,)      int array
            augs   : (A,)      array of view names
    
        The companion ``"<cache>.classes.json"`` is copied next to the new file
        when it exists, because the loader silently falls back to integer class
        names otherwise.
    
        Returns the path of the file that was written.
        """
        
        novel_aug = self.augmentations
        result = novel_aug.pop(view_name, None)
        if result == None:
            raise ValueError(f"{view_name} is not found")
        aug_spec = "_".join(f"{k}{v}" for k, v in novel_aug.items())

        hash_of_dict = UtilsGeneral.hash_dict(d=novel_aug)

        path_to_descrition_file = os.path.join(
            self.cache_dir, f"description_{self.arch}_{self.weights}_{hash_of_dict}.txt")
        
        UtilsGeneral.write_string_to_file(content=aug_spec, path=path_to_descrition_file)
        
        out_path = os.path.join(
            self.cache_dir, f"logits_{self.arch}_{self.weights}_{hash_of_dict}.npz")
        
        if out_path is None:
            raise ValueError(out_path)
    
        if os.path.exists(out_path):
            raise FileExistsError(f"refusing to overwrite {out_path}")
    
        with np.load(cache_path) as z:
            augs = [str(a) for a in z["augs"]]
    
            if view_name not in augs:
                raise ValueError(f"view {view_name!r} not in cache views {augs}")
            if augs.count(view_name) > 1:
                raise ValueError(f"view {view_name!r} appears {augs.count(view_name)} times")
            if len(augs) < 2:
                raise ValueError("cache holds a single view; nothing would remain")
    
            drop = augs.index(view_name)
            keep = [i for i in range(len(augs)) if i != drop]
    
            logits = z["logits"]
            if logits.ndim != 3 or logits.shape[0] != len(augs):
                raise ValueError(
                    f"logits has shape {logits.shape} but there are {len(augs)} "
                    "view names; expected layout (A, N, K)")
    
            logits = logits[keep]          # peak RAM here is ~2x the full array
            labels = z["labels"]
    
        kept_augs = [augs[i] for i in keep]
    
        tmp = out_path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, logits=logits, labels=labels, augs=np.array(kept_augs))
        os.replace(tmp, out_path)
    
        classes_src = cache_path + ".classes.json"
        classes_dst = out_path + ".classes.json"
        if os.path.isfile(classes_src):
            with open(classes_src, "r", encoding="utf-8") as f:
                classes = json.load(f)
            with open(classes_dst, "w", encoding="utf-8") as f:
                json.dump(classes, f)
        else:
            print(f"[cache] warning: {classes_src} missing, no class names copied",
                flush=True)
    
        print(f"[cache] views {augs} -> {kept_augs}")
        print(f"[cache] logits {logits.shape}, labels {labels.shape}")
        print(f"[cache] written {out_path} "
            f"({os.path.getsize(out_path) / 1e9:.2f} GB)", flush=True)
        return out_path


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

        V = len(self.list_augmentation_names)
        N = len(val_dataset)
        # softmax = torch.nn.Softmax(dim=1)
        logits = None
        labels = np.empty(N, dtype=np.int64)
        pos = 0
        with torch.no_grad():
            for data, target in loader:
                data = data.to(device, non_blocking=True)
                B = data.shape[0]
                
                # views = [utils.tta_policies.apply_tta_aug_policy(data, augmentation_name=name)
                #          for name in self.list_augmentation_names]
                views = []
                for name in self.list_augmentation_names:
                    if name.startswith("ROTATE_PROBE_"):
                        views.append(UtilsAugmentations.apply_single_augmentation(
                            data, "ROTATE", rng=None, degree=self.augmentations[name]["degree"]))
                    else:
                        views.append(utils.tta_policies.apply_tta_aug_policy(data, augmentation_name=name))
                big = torch.cat(views, dim=0)
                # out = softmax(model(self._prepare_model_inputs(big)))
                logit_out = model(self._prepare_model_inputs(big))
                out = logit_out.view(V, B, -1).cpu().numpy().astype(np.float32)
                if logits is None:
                    logits = np.empty((V, N, out.shape[2]), dtype=np.float32)
                logits[:, pos:pos + B] = out
                labels[pos:pos + B] = target.numpy()
                pos += B
                if (pos // self.batch_size) % 20 == 0:
                    print(f"[imagenet] forward {pos}/{N}", flush=True)
        assert pos == N
        return logits, labels, classes

    # --------------------- preprocessing & augmentation ---------------------

    # @staticmethod
    def _prepare_model_inputs(self, data):
        # Because the no transformation has been applied while loading the data.
        # return self.preprocess_transform(data)
        # return data
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
            zip(self.list_augmentation_names, self.accuracies["Testing"]))
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
