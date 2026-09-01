import os

import torch
import torch.nn as nn
import torchvision

from benchmarks.vision_classification import VisionClassificationBenchmark


"""
CIFAR-100 Benchmark for the Conformance Prediction Framework
"""


class ResNet(nn.Module):
    """ResNet-18 or ResNet-50 used for CIFAR-100 recognition."""

    def __init__(self, architecture="resnet18", num_classes=100):
        super(ResNet, self).__init__()

        self.architecture = architecture
        self.model = self._create_resnet(
            architecture=architecture,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _create_resnet(architecture="resnet18", num_classes=100):
        if architecture == "resnet18":
            model = torchvision.models.resnet18(weights=None)
        elif architecture == "resnet50":
            model = torchvision.models.resnet50(weights=None)
        else:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                "Use 'resnet18' or 'resnet50'."
            )

        # Adapt ImageNet ResNet architecture to 32x32 CIFAR images.
        model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        model.maxpool = nn.Identity()

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        return model


class RESNETBenchmark(VisionClassificationBenchmark):
    """
    The CIFAR-100 dataset as data generator for the conformance prediction framework.
    """

    benchmark_name = "CIFAR100"
    model_file_prefix = "resnet_cifar100"

    normalization_mean = (0.5071, 0.4867, 0.4408)
    normalization_std = (0.2675, 0.2565, 0.2761)

    def __init__(
        self,
        augmentations,
        path_logs,
        seed=42,
        robust_training=False,
        train_fraction=1.0,
        test_fraction=0.8,
        calibration_fraction=0.2,
        precalibration_fraction=0.1,
        batch_size=128,
        iterations=0,
        num_workers=2,
        probability_batch_size=128,
        cifar100_root="/scratch/lkd18/data/cifar100",
        architecture="resnet18",
        download=True,
    ):
        self.cifar100_root = cifar100_root
        self.architecture = architecture
        self.download = download

        super().__init__(
            augmentations=augmentations,
            path_logs=path_logs,
            seed=seed,
            robust_training=robust_training,
            train_fraction=train_fraction,
            test_fraction=test_fraction,
            calibration_fraction=calibration_fraction,
            precalibration_fraction=precalibration_fraction,
            batch_size=batch_size,
            iterations=iterations,
            num_workers=num_workers,
            probability_batch_size=probability_batch_size,
        )

    def _load_datasets(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        trainset_full = torchvision.datasets.CIFAR100(
            root=self.cifar100_root,
            train=True,
            download=self.download,
            transform=transform,
        )

        testset_full = torchvision.datasets.CIFAR100(
            root=self.cifar100_root,
            train=False,
            download=self.download,
            transform=transform,
        )

        return trainset_full, testset_full

    def _create_model(self):
        return ResNet(
            architecture=self.architecture,
            num_classes=self.n_classes,
        )

    def _prepare_model_inputs(self, data):
        mean = torch.tensor(
            self.normalization_mean,
            device=data.device,
            dtype=data.dtype,
        ).view(1, -1, 1, 1)

        std = torch.tensor(
            self.normalization_std,
            device=data.device,
            dtype=data.dtype,
        ).view(1, -1, 1, 1)

        return (data - mean) / std