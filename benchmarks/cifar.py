import os
import random
# from collections.abc import generator
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torchvision
from numpy.f2py.auxfuncs import throw_error

from utils import function_utils
from utils.function_utils import Augmentation, UtilsAugmentations

"""
CIFAR10 Benchmark for the Conformance Prediction Framework
"""


# Reason to use model.eval() and model.train() in PyTorch
# https://pytorch.org/docs/stable/notes/autograd.html#evaluation-mode-nn-module-eval
class CifarNet(nn.Module):
    """Neural network used for CIFAR10 recognition"""

    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class CIFAR10Benchmark:
    """
    The CIFAR10 Dataset as data generator for the conformance prediction framework.
    """

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
        batch_size=64,
        iterations=2,
    ):
        self.augmentations = augmentations
        self.list_augmentations = list(augmentations.keys())
        self.calibration_data = None
        self.testing_data = None
        self.precalibration_data = None
        self.accuracies = {}
        self.seed = seed
        self.path_logs = path_logs
        self.class_accuracies = {}

        if test_fraction + calibration_fraction > 1.0:
            raise ValueError('test_fraction + calibration_fraction must be <= 1.0')
        if precalibration_fraction > train_fraction:
            raise ValueError('precalibration_fraction must be <= train_fraction')

        self.train_fraction = train_fraction
        self.test_fraction = test_fraction
        self.calibration_fraction = calibration_fraction
        self.precalibration_fraction = precalibration_fraction

        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.DEVICE = DEVICE
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        BATCH_SIZE = batch_size
        ITERATIONS = iterations

        self.classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

        self.basepath = os.path.dirname(__file__)
        os.makedirs(self.path_logs, exist_ok=True)

        model_specification = '_'.join(f'{k}{v}' for k, v in self.augmentations.items())
        dataset_specification = (
            f'train{self.train_fraction}_test{self.test_fraction}'
        )

        if robust_training:
            self.path_saved_model = (
                f'{self.path_logs}cnn_cifar10_robust_{DEVICE}_ITERATION{ITERATIONS}_'
                f'{dataset_specification}_{model_specification}.pt'
            )
        else:
            self.path_saved_model = (
                f'{self.path_logs}cnn_cifar10_{DEVICE}_ITERATION{ITERATIONS}_'
                f'{dataset_specification}.pt'
            )

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])


        trainset_full = torchvision.datasets.CIFAR10(
            root=self.basepath + '/data',
            train=True,
            download=True,
            transform=transform,
        )
        testset_full = torchvision.datasets.CIFAR10(
            root=self.basepath + '/data',
            train=False,
            download=True,
            transform=transform,
        )

        trainset, _ = self._split_dataset_two_parts(
            trainset_full,
            first_fraction=self.train_fraction,
            second_fraction=1.0 - self.train_fraction,
            generator_offset=0,
        )
        precalibration_set, _ = self._split_dataset_two_parts(
            trainset_full,
            first_fraction=self.precalibration_fraction,
            second_fraction=1.0 - self.train_fraction,
            generator_offset=0,
        )
        testing_set, calibration_set = self._split_dataset_two_parts(
            testset_full,
            first_fraction=self.test_fraction,
            second_fraction=self.calibration_fraction,
            generator_offset=1,
        )

        this_generator = torch.Generator().manual_seed(self.seed)

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            generator=this_generator,
        )
        testloader = torch.utils.data.DataLoader(
            testing_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            generator=this_generator,
        )

        print("Size of sets:")
        print(f" - Train: {len(trainset)}/{len(trainset_full)}")
        print(f" - Precal: {len(precalibration_set)}/{len(trainset)} (of extracted training set)")
        print(f" - Test: {len(testing_set)}/{len(testset_full)}")
        print(f" - Calib: {len(calibration_set)}/{len(testset_full)}")


        model = CifarNet().to(DEVICE)

        if not os.path.isfile(self.path_saved_model):
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

            for epoch in range(ITERATIONS):
                model.train()
                for i, (data, target) in enumerate(trainloader):
                    if robust_training and len(self.list_augmentations) > 0:
                        random_aug_name = self.list_augmentations[
                            random.randint(0, len(self.list_augmentations) - 1)
                        ]
                    else:
                        random_aug_name = Augmentation.ORIGINAL.value

                    data = self.apply_aug(data=data, augmentation_name=random_aug_name)
                    inputs, labels = data.to(DEVICE), target.to(DEVICE)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    if i % 100 == 0:
                        print(f'[{epoch}, batch:{i}] loss: {loss.item():.3f}', flush=True)
                scheduler.step()

            torch.save(model.state_dict(), self.path_saved_model)
        else:
            print('Trained model is loaded from:', self.path_saved_model)
            state_dict = torch.load(self.path_saved_model, map_location=DEVICE)
            model.load_state_dict(state_dict)

        model.eval()

        self.calibration_data = []
        self.testing_data = []
        self.precalibration_data = []

        softmax = torch.nn.Softmax(dim=1)
        i = 1

        for (inputSet, outputList) in [
            (calibration_set, self.calibration_data),
            (testing_set, self.testing_data),
            (precalibration_set, self.precalibration_data),
        ]:
            print(f"Loading of probability distributions ({i}/3)..."); i += 1
            self._collect_probability_data(
                inputSet=inputSet,
                outputList=outputList,
                model=model,
                softmax=softmax,
            )

        for (loader, dataset) in [(trainloader, 'Training'), (testloader, 'Testing')]:
            self._evaluate_accuracies(loader=loader, dataset=dataset, model=model)

    def _split_dataset_two_parts(self, dataset, first_fraction, second_fraction, generator_offset=0):
        total_size = len(dataset)
        first_size = int(total_size * first_fraction)
        second_size = int(total_size * second_fraction)
        unused_size = total_size - first_size - second_size

        lengths = [first_size, second_size, unused_size]
        generator = torch.Generator().manual_seed(self.seed + generator_offset)
        first_set, second_set, _ = torch.utils.data.random_split(dataset, lengths, generator=generator)
        return first_set, second_set

    def _collect_probability_data(self, inputSet, outputList, model, softmax):
        loader = torch.utils.data.DataLoader(
            inputSet,
            batch_size=128,
            shuffle=False,
            num_workers=1,
        )

        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.DEVICE, non_blocking=True)
                target_np = target.cpu().numpy()

                B = data.shape[0]
                A = len(self.list_augmentations)

                aug_batches = []
                for augmentation_name in self.list_augmentations:
                    dataAug = self.apply_aug(data=data, augmentation_name=augmentation_name)
                    aug_batches.append(dataAug)

                big_batch = torch.cat(aug_batches, dim=0)
                outputs = softmax(model(big_batch))
                outputs_np = outputs.view(A, B, -1).cpu().numpy()

                for j in range(B):
                    these_distributions = [outputs_np[a, j, :] for a in range(A)]
                    outputList.append((these_distributions, target_np[j]))

    def _evaluate_accuracies(self, loader, dataset, model):
        correct_pred = {
            aug_name: {classname: 0 for classname in self.classes}
            for aug_name in self.list_augmentations
        }
        total_pred = {
            aug_name: {classname: 0 for classname in self.classes}
            for aug_name in self.list_augmentations
        }

        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.DEVICE, non_blocking=True)
                labels = target.to(self.DEVICE)
                B = data.shape[0]
                A = len(self.list_augmentations)

                aug_batches = []
                for augmentation_name in self.list_augmentations:
                    dataAug = self.apply_aug(data=data, augmentation_name=augmentation_name)
                    aug_batches.append(dataAug)

                big_batch = torch.cat(aug_batches, dim=0)
                outputs = model(big_batch)
                _, predictions = torch.max(outputs, 1)
                predictions = predictions.view(A, B)

                for a_idx, augmentation_name in enumerate(self.list_augmentations):
                    preds_a = predictions[a_idx]
                    for label, prediction in zip(labels, preds_a):
                        cls_name = self.classes[label.item()]
                        if label == prediction:
                            correct_pred[augmentation_name][cls_name] += 1
                        total_pred[augmentation_name][cls_name] += 1

        self.accuracies[dataset] = []
        self.class_accuracies[dataset] = {}

        for augmentation_name in self.list_augmentations:
            print(
                '==========[Accuracy on the', dataset,
                'dataset (Augmentation:', augmentation_name, ')]=========='
            )

            self.class_accuracies[dataset][augmentation_name] = []

            for cls_name in self.classes:
                c = correct_pred[augmentation_name][cls_name]
                t = total_pred[augmentation_name][cls_name]
                accuracy = round(100.0 * c / t, 2) if t > 0 else 0.0

                print('Class', cls_name, ':', accuracy, '% (', t, 'cases)')
                self.class_accuracies[dataset][augmentation_name].append((cls_name, accuracy, t))

            total_correct = sum(correct_pred[augmentation_name].values())
            total_cases = sum(total_pred[augmentation_name].values())
            overall = round(100.0 * total_correct / total_cases, 2) if total_cases > 0 else 0.0

            self.accuracies[dataset].append(overall)
            print('Overall accuracy:', overall, '% (', total_cases, 'cases)')

        print('\n', flush=True)

    def apply_aug(self, data, augmentation_name):
        if augmentation_name not in self.list_augmentations:
            raise NotImplementedError(f'Unknown augmentation: {augmentation_name}')

        if augmentation_name == Augmentation.ORIGINAL.value:
            return data
        if augmentation_name == Augmentation.HORIZONTAL_FLIP.value:
            return UtilsAugmentations.apply_horizontal_flip(data)
        if augmentation_name == Augmentation.BLUR.value:
            return UtilsAugmentations.apply_gaussian_blur(data, self.augmentations[augmentation_name])
        if augmentation_name == Augmentation.CONTRAST.value:
            return UtilsAugmentations.apply_contrast(data, contrast_factor=self.augmentations[augmentation_name])
        if augmentation_name == Augmentation.BRIGHTNESS.value:
            return UtilsAugmentations.apply_brightness(data, brightness_factor=self.augmentations[augmentation_name])
        if augmentation_name == Augmentation.HUE.value:
            return UtilsAugmentations.apply_hue(data, hue_factor=self.augmentations[augmentation_name])
        if augmentation_name == Augmentation.ROTATION.value:
            return UtilsAugmentations.apply_rotation(data, angle_deg=self.augmentations[augmentation_name])

    def _texClassAccuracies(self, dataset: str) -> str:
        data = self.class_accuracies.get(dataset, {})
        augs = list(self.list_augmentations)

        acc_map = {aug: {} for aug in augs}
        for aug in augs:
            for cls, acc, n in data.get(aug, []):
                acc_map[aug][cls] = (float(acc), int(n))

        classes = list(getattr(self, 'classes', []))
        if not classes:
            classes = sorted({cls for aug in augs for cls, _, _ in data.get(aug, [])})

        colspec = 'l|r|' + '|'.join(['r'] * len(augs))

        header = []
        header.append(f'\\begin{{tabular}}{{{colspec}}}')
        header.append('\\hline')
        header.append(
            '\\textbf{Class} & \\textbf{N} & '
            + f'\\multicolumn{{{len(augs)}}}{{c}}{{\\textbf{{Accuracies}}}} \\\\ \\hline'
        )
        header.append(
            ' &  & ' + ' & '.join([f'\\textbf{{{aug}}}' for aug in augs]) + ' \\\\ \\hline'
        )

        rows = []
        for cls in classes:
            n_val = ''
            for aug in augs:
                if cls in acc_map[aug]:
                    n_val = str(acc_map[aug][cls][1])
                    break

            acc_vals = []
            for aug in augs:
                if cls in acc_map[aug]:
                    acc_vals.append(f'{acc_map[aug][cls][0]:.2f}\\%')
                else:
                    acc_vals.append('--')

            rows.append(f'{cls} & {n_val} & ' + ' & '.join(acc_vals) + ' \\\\ \\hline')

        if not rows:
            rows = [f'\\multicolumn{{{2 + len(augs)}}}{{c}}{{No class accuracies available}} \\\\ \\hline']

        return '\n'.join(header + rows + ['\\end{tabular}'])

    def texInfo(self):
        text1 = """
        \\begin{{tabular}}{{l|l}} \\
        \\textbf{{Benchmark name:}} & CIFAR10 \\\\ \\hline
        \\textbf{{Augmentations:}} & {0} \\\\ \\hline
        \\textbf{{Accuracy Training set:}} & {1} \\% \\\\ \\hline
        \\textbf{{Accuracy Testing set:}} & {2} \\% \\
        \\end{{tabular}}
        """.format(list(self.augmentations.values()), self.accuracies['Training'], self.accuracies['Testing'])

        final = """
        \\section{Accuracies Per Class on Training Set}
        """
        final += self._texClassAccuracies('Training')

        final += """
        \\newpage
        \\section{Accuracies Per Class on Test Set}
        """

        final += self._texClassAccuracies('Testing')
        return text1 + final
