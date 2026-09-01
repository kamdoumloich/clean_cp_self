# import os
# import random
# from enum import Enum

# import numpy as np
# import torch
# import torch.nn as nn
import torchvision
# from numpy.f2py.auxfuncs import throw_error

from benchmarks.vision_classification import VisionClassificationBenchmark
# from utils import function_utils
# from utils.function_utils import Augmentation, UtilsAugmentations

from benchmarks.resnet_architecture import ResNet

"""
CIFAR10 Benchmark for the Conformance Prediction Framework
"""


# Reason to use model.eval() and model.train() in PyTorch
# https://pytorch.org/docs/stable/notes/autograd.html#evaluation-mode-nn-module-eval
# class CIFAR10Net(nn.Module):
#     """Neural network used for CIFAR10 recognition"""

#     def __init__(self):
#         super(CIFAR10Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(256 * 4 * 4, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 10)
#         self.dropout = nn.Dropout(p=0.5)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.pool(x)
#         x = self.relu(self.conv2(x))
#         x = self.pool(x)
#         x = self.relu(self.conv3(x))
#         x = self.pool(x)
#         x = self.flatten(x)
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x


# https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.quantization.resnet18.html
CIFAR10_MEAN = (0.485, 0.456, 0.406)
CIFAR10_STD = (0.229, 0.224, 0.225)

class CIFAR10Benchmark(VisionClassificationBenchmark):
    """
    The CIFAR10 Dataset as data generator for the conformance prediction framework.
    """

    
    def __init__(
        self,
        # augmentations,
        path_logs,
        policy_name,
        expected_max_rotation,
        arch="resnet18",
        weights="IMAGENET1K_V1",
        seed=42,
        list_other_aug=None,
        robust_training=False,
        train_fraction=1.0,
        test_fraction=0.4,
        calibration_fraction=0.5,
        precalibration_fraction=0.1,
        batch_size=1024,
        iterations=5,
        # probability_batch_size=1024,
        pretrained=True,
        heldout_precal=True,
        data_root="/scratch/lkd18/data/cifar10",
    ):
        
        self.benchmark_name = 'CIFAR10'
        self.model_file_prefix = 'resnet18_cifar10'
        self.pretrained = pretrained    
        self.data_root = data_root
        self.arch = arch
        self.weights = weights
        
        super().__init__(
            # augmentations=augmentations,
            path_logs=path_logs,
            policy_name=policy_name,
            list_other_aug=list_other_aug,
            seed=seed,
            robust_training=robust_training,
            train_fraction=train_fraction,
            test_fraction=test_fraction,
            calibration_fraction=calibration_fraction,
            precalibration_fraction=precalibration_fraction,
            batch_size=batch_size,
            iterations=iterations,
            heldout_precal=heldout_precal,
            # num_workers=num_workers,
            probability_batch_size=batch_size,
            pretrained=pretrained,
            expected_max_rotation=expected_max_rotation,
        )

    def _load_datasets(self):
        transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BILINEAR,),
            torchvision.transforms.ToTensor(),
        ])
        
        print(self.data_root)

        trainset_full = torchvision.datasets.CIFAR10(
            root=self.data_root,
            train=True,
            download=True,
            transform=transform,
        )
        testset_full = torchvision.datasets.CIFAR10(
            root=self.data_root,
            train=False,
            download=True,
            transform=transform,
        )

        return trainset_full, testset_full

    def _create_model(self):
        return ResNet(num_classes=10, pretrained=self.pretrained, architecture=self.arch, weights=self.weights)
    
    
    def _prepare_model_inputs(self, data):
        return torchvision.transforms.functional.normalize(data, mean=list(CIFAR10_MEAN), std=list(CIFAR10_STD))
    
    def _can_evaluate_without_training(self):
        return False
    