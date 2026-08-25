# import torch.nn as nn
import torchvision

from benchmarks.vision_classification import VisionClassificationBenchmark
from benchmarks.resnet_architecture import ResNet


"""
CIFAR100 Benchmark for the Conformance Prediction Framework
"""


# class CIFAR100Net(nn.Module):
#     """Neural network used for CIFAR100 recognition."""

#     def __init__(self):
#         super(CIFAR100Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(256 * 4 * 4, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 100)
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
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

class CIFAR100Benchmark(VisionClassificationBenchmark):
    """
    The CIFAR100 Dataset as data generator for the conformance prediction framework.
    """

        
    def __init__(
        self,
        # augmentations,
        path_logs,
        policy_name,
        seed=42,
        arch="resnet18",
        weights="IMAGENET1K_V1",
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
        data_root="/scratch/lkd18/data/cifar100",
    ):
        
        self.pretrained = pretrained    
        self.data_root = data_root
        self.benchmark_name = 'CIFAR100'
        self.model_file_prefix = 'resnet18_cifar100'
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
        )
        
    def _load_datasets(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        trainset_full = torchvision.datasets.CIFAR100(
            root=self.data_root,
            train=True,
            download=True,
            transform=transform,
        )
        testset_full = torchvision.datasets.CIFAR100(
            root=self.data_root,
            train=False,
            download=True,
            transform=transform,
        )

        return trainset_full, testset_full

    def _create_model(self):
        return ResNet(num_classes=100, pretrained=self.pretrained, architecture=self.arch, weights=self.weights)
    
    def _prepare_model_inputs(self, data):
        return torchvision.transforms.functional.normalize(data, mean=list(_MEAN), std=list(_STD))

    def _can_evaluate_without_training(self):
        return False
