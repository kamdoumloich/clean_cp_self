#!/usr/bin/env python3
"""
One-time data / weight preparation for the experiment suite.

  python3 prepare_data.py --download_weights                 # ResNet-50 IN1k
  python3 prepare_data.py --cifar10 --cifar100               # torchvision
  python3 prepare_data.py --cifar10c                         # Zenodo, ~2.9 GB
  python3 prepare_data.py --imagenet_root                    # extract val

ImageNet: the validation archive CANNOT be downloaded automatically -- it
requires registration at https://image-net.org. Place these two files in
<imagenet_root> yourself, then run this script to extract:
    ILSVRC2012_img_val.tar          (~6.3 GB)
    ILSVRC2012_devkit_t12.tar.gz    (~2.5 MB)
If <imagenet_root> already contains an extracted val/ directory prepared by
torchvision, the script just verifies it.

CIFAR-10-C Zenodo URL is the published record 2535967; the download was NOT
executable from the environment in which this script was written, so if the
URL scheme changed, download CIFAR-10-C.tar manually from
https://zenodo.org/records/2535967 into <data_root> and rerun --cifar10c to
extract.
"""


# Diese Klasse wurde ursprunglich mit Claude-KI generiert und danach bearbeitet.

import argparse
import os
# import sys
import tarfile
# import urllib.request


def do_weights(arch, weights):
    import torchvision
    print(f"[weights] fetching {arch} / {weights} into the torch hub cache "
          f"(TORCH_HOME={os.environ.get('TORCH_HOME', '~/.cache/torch')})")
    torchvision.models.get_model(arch, weights=weights)
    print("[weights] done.")


def do_cifar(data_root, which):
    import torchvision
    cls = (torchvision.datasets.CIFAR10 if which == 10
           else torchvision.datasets.CIFAR100)
    data_root += "/cifar10" if which == 10 else "/cifar100"
    for train in (True, False):
        cls(root=data_root, train=train, download=True)
    print(f"[cifar{which}] downloaded to {data_root}")


def do_cifar10c(data_root):
    out_dir = "/scratch/lkd18/data/cifar10c"
    if os.path.isfile(os.path.join(out_dir, "labels.npy")):
        print(f"[cifar10c] already extracted at {out_dir}")
        return
    tar_path = os.path.join(data_root, "CIFAR-10-C.tar")
    if not os.path.isfile(tar_path):
        raise RuntimeError(f"Please put the tar-file for cifar-10-c into {data_root}.")
    with tarfile.open(tar_path) as tf:
        tf.extractall(path=data_root)
    os.rename(data_root+"/CIFAR-10-C", data_root+"/cifar10c")
    if not os.path.isfile(os.path.join(out_dir, "labels.npy")):
        raise SystemExit("[cifar10c] extraction did not produce "
                         f"{out_dir}/labels.npy; archive layout unexpected.")
    print(f"[cifar10c] ready at {out_dir}")


def do_imagenet(imagenet_root):
    import torchvision
    val_dir = os.path.join(imagenet_root, "val")
    meta = os.path.join(imagenet_root, "meta.bin")
    if os.path.isdir(val_dir) and os.path.isfile(meta):
        n = sum(len(fs) for _, _, fs in os.walk(val_dir))
        print(f"[imagenet] extracted val/ found ({n} files). Verifying via "
              "torchvision...")
        torchvision.datasets.ImageNet(root=imagenet_root, split="val")
        print("[imagenet] ready.")
        return
    val_tar = os.path.join(imagenet_root, "ILSVRC2012_img_val.tar")
    devkit = os.path.join(imagenet_root, "ILSVRC2012_devkit_t12.tar.gz")
    missing = [p for p in (val_tar, devkit) if not os.path.isfile(p)]
    if missing:
        raise SystemExit(
            "[imagenet] cannot proceed; missing:\n  " +
            "\n  ".join(missing) +
            "\nplease download imagent and put it in ", imagenet_root)
    torchvision.datasets.ImageNet(root=imagenet_root, split="val")
    print("[imagenet] ready.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="/scratch/lkd18/data")
    ap.add_argument("--download_weights", action="store_true")
    ap.add_argument("--arch", type=str, default="resnet50")
    ap.add_argument("--weights", type=str, default="IMAGENET1K_V1")
    ap.add_argument("--cifar10", action="store_true")
    ap.add_argument("--cifar100", action="store_true")
    ap.add_argument("--cifar10c", action="store_true")
    ap.add_argument("--imagenet_root", type=str, default="/scratch/lkd18/data/imagenet")
    args = ap.parse_args()

    ran = False
    if args.download_weights:
        do_weights(args.arch, args.weights); ran = True
    if args.cifar10:
        do_cifar(args.data_root, 10); ran = True
    if args.cifar100:
        do_cifar(args.data_root, 100); ran = True
    if args.cifar10c:
        do_cifar10c(args.data_root); ran = True
    if args.imagenet_root is not None:
        do_imagenet(args.imagenet_root); ran = True
    if not ran:
        print(__doc__)


if __name__ == "__main__":
    main()
