"""
Test-time augmentation policies of

    D. Shanmugam, H. Lu, S. Sankaranarayanan, J. Guttag,
    "Test-time Augmentation Improves Efficiency in Conformal Prediction",
    CVPR 2025 (arXiv:2505.22764), Supplement S1.1.

Two policies:
  SIMPLE_POLICY   : identity + random-crop + horizontal-flip (3 views).
  EXPANDED_POLICY : identity + the 12 transforms above (13 views). This is
                    the policy behind their headline results, and the one for
                    which TTA-Learned beats TTA-Avg, because theta can
                    suppress the unhelpful transforms.

Inputs are float tensors in [0, 1], shape (B, C, H, W) -- i.e. AFTER ToTensor
and BEFORE any dataset normalization, matching the ordering used by the
benchmarks in this project. Normalization must stay downstream of these.

Reproduction caveats, flagged rather than guessed:
  * Per-image vs per-batch sampling of the stochastic parameters is not
    specified in the paper. Default here is per_sample=True (one draw per
    image, the standard torchvision dataloader semantics), which costs a
    Python loop; per_sample=False draws once per batch and is much faster.
  * The paper states blur uses "kernel size 5 (and default sigma range of
    [.1, .2])". torchvision's GaussianBlur default sigma range is actually
    (0.1, 2.0). The literal value from the paper is used below.
  * Random crop is described for 256x256 ImageNet inputs (pad 4, crop 256,
    then center-crop 224). Implemented here as pad 4 + random crop back to
    the input's own spatial size, which is the size-agnostic equivalent and
    the standard CIFAR form.
  * Posterize requires uint8 in torchvision; the [0,1] float input is
    quantized to uint8 and back, which is inherent to the transform.

Exchangeability is unaffected by the stochastic transforms as long as the
draws are i.i.d. across examples and the same policy is applied to
pre-calibration, calibration and test data.
"""



import numpy as np

POLICY_PARAMS = {
    "ORIGINAL":           {},
    "INCREASE_SHARPNESS": {"factor": 1.3},                  # deterministic
    "DECREASE_SHARPNESS": {"factor": 0.7},                  # deterministic
    "AUTOCONTRAST":       {},                               # deterministic
    "INVERT":             {},                               # deterministic
    "BLUR":               {"kernel_size": 5,
                        "sigma_range": (0.1, 0.2)},      # stochastic
    "POSTERIZE":          {"bits": 4},                      # deterministic
    "SHEAR":              {"degrees": (-10.0, 10.0)},       # stochastic
    "TRANSLATE":          {"height_fraction": (0.0, 0.1)},  # stochastic
    "COLOR_JITTER":       {"brightness": (0.9, 1.1),
                        "contrast": (0.9, 1.1),
                        "saturation": (0.9, 1.1)},       # stochastic
    "RANDOM_CROP":        {"padding": 4},                   # stochastic
    "HORIZONTAL_FLIP":    {},                               # deterministic
    "ROTATE":             {"degrees": (-10.0, 10.0)},       # stochastic
}

SIMPLE_POLICY = ("ORIGINAL", "RANDOM_CROP", "HORIZONTAL_FLIP")

ONLY_HFLIP_POLICY = ("ORIGINAL", "HORIZONTAL_FLIP")
ORIGINAL_ONLY = ("ORIGINAL",)

EXPANDED_POLICY = (
    "ORIGINAL",
    "INCREASE_SHARPNESS", "DECREASE_SHARPNESS", "AUTOCONTRAST", "INVERT", "HORIZONTAL_FLIP", "POSTERIZE",
    "BLUR", "SHEAR", "TRANSLATE", "COLOR_JITTER", "RANDOM_CROP", "ROTATE",
)

STOCHASTIC = {"BLUR", "SHEAR", "TRANSLATE", "COLOR_JITTER", "RANDOM_CROP", "ROTATE"}

# class UtilsTTA:
    
    # def __init__(self) -> None:
    #     pass


def build_augs_dict(policy='simple_policy'):
    if policy == 'simple_policy':
        policy=SIMPLE_POLICY
    elif policy == 'expanded_policy':
        policy=EXPANDED_POLICY
    elif policy == 'only_hflip_policy':
        policy=ONLY_HFLIP_POLICY
    elif policy == 'original_only_policy':
        policy=ORIGINAL_ONLY
    else:
        raise NotImplementedError("please select either: simple_policy or expanded_policy")
    return {name: POLICY_PARAMS[name] for name in policy}


def _apply_single(x, name, rng):
    """x: (1, C, H, W) or (B, C, H, W) float in [0,1]; one parameter draw."""
    import torch
    import torchvision.transforms.functional as TF

    p = POLICY_PARAMS[name]

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
        return TF.adjust_sharpness(x, sharpness_factor=p["factor"])
    if name == "POSTERIZE":
        # https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.posterize.html
        # Fügt Kanten zu einem Bild hinzu.
        x8 = (x.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        return TF.posterize(x8, bits=p["bits"]).float().div_(255.0)
    if name == "BLUR":
        lo, hi = p["sigma_range"]
        sigma = float(rng.uniform(lo, hi))
        # if p["kernel_size"] % 2 == 0:
        return TF.gaussian_blur(x, kernel_size=p["kernel_size"], sigma=sigma)
    if name == "SHEAR":
        lo, hi = p["degrees"]
        s = float(rng.uniform(lo, hi))
        return TF.affine(x, angle=0.0, translate=(0, 0), scale=1.0, shear=[s, 0.0])
    if name == "ROTATE":
        lo, hi = p["degrees"]
        a = float(rng.uniform(lo, hi))
        return TF.affine(x, angle=a, translate=(0, 0), scale=1.0, shear=[0.0])
    if name == "TRANSLATE":
        lo, hi = p["height_fraction"]
        frac = float(rng.uniform(lo, hi))
        dy = int(round(frac * x.shape[-2]))
        return TF.affine(x, angle=0.0, translate=(0, dy), scale=1.0,
                        shear=[0.0])
    if name == "COLOR_JITTER":
        out = TF.adjust_brightness(x, float(rng.uniform(*p["brightness"])))
        out = TF.adjust_contrast(out, float(rng.uniform(*p["contrast"])))
        return TF.adjust_saturation(out, float(rng.uniform(*p["saturation"])))
    if name == "RANDOM_CROP":
        pad = p["padding"]
        h, w = x.shape[-2], x.shape[-1]
        padded = TF.pad(x, [pad, pad, pad, pad])
        top = int(rng.integers(0, 2 * pad + 1))
        left = int(rng.integers(0, 2 * pad + 1))
        return TF.crop(padded, top=top, left=left, height=h, width=w)

    raise NotImplementedError(f"Unknown TTA augmentation: {name}")

def apply_tta_aug_policy(data, augmentation_name, rng=None, per_sample=True):
    """
    There is no possibility to defince the exact degree of the augmentation. 
    This follows the idea of the paper TTA.
    Apply one policy augmentation to a batch of [0,1] float images.

    data              : (B, C, H, W) float tensor in [0, 1]
    augmentation_name : key of POLICY_PARAMS
    rng               : numpy Generator (seed it for reproducible draws)
    per_sample        : draw stochastic parameters independently per image
                        (True, default) or once for the whole batch (False)
    """
    import torch

    if augmentation_name not in POLICY_PARAMS:
        raise NotImplementedError(
            f"Unknown TTA augmentation: {augmentation_name}")
    if rng is None:
        rng = np.random.default_rng(2026)
    else:
        rng = np.random.default_rng(rng)
        
    if data.dim() != 4:
        raise ValueError(f"Expected (B, C, H, W), got {tuple(data.shape)}.")
    if float(data.min()) < 0.0 or float(data.max()) > 1.0 + 1e-5:
        raise ValueError(f"apply_tta_aug expects inputs in [0, 1]; normalize "
                        f"AFTER augmenting. Min:{data.min()} and Max:{data.max()}")

    if augmentation_name not in STOCHASTIC or not per_sample:
        return _apply_single(data, augmentation_name, rng)

    return torch.cat(
        [_apply_single(data[i:i + 1], augmentation_name, rng)
        for i in range(data.shape[0])], dim=0)