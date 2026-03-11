import random
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import os
import torchvision

"""
MNIST Benchmark for the Conformace Prediction Framework
"""


class MNistNet(nn.Module):    
    """Neural network used for MNIST digit reconition"""
    def __init__(self):
        super(MNistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, (3,3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(40, 32, (3,3), stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*7*7, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        # soft = F.softmax(x, dim=1)  # Apply softmax to the output
        return x

class Augmentation(Enum):
    ORIGINAL = 'ORIGINAL'
    ROTATION = 'ROTATION'
    TRANSLATION = 'TRANSLATION'
    RANDOM_RESIZED_CROP = 'RANDOM-RESIZED-CROP'
    HORIZONTAL_FLIP = 'HORIZONTAL-FLIP'
    VERTICAL_FLIP = 'VERTICAL-FLIP'
    MIXUP = 'MIXUP'
    BLUR = 'BLUR'
    CONTRAST = 'CONTRAST'
    BRIGHTNESS = 'BRIGHTNESS'

class MNISTBenchmark:
  """
    The MNIST Dataset as data generator for the conformance prediction framework

    Attributes:
       augmentations        A list of all augmentation values used when generating the probability vectors
       calibration_data     A list of pairs of probability distributions (one for each augmentation) and the corresponding classes, calibration set for the conformance predictor
       testing_data         A list of pairs of probability distributions (one for each augmentation) and the corresponding classes, testing set for the conformance predictor
       classes              Names for all classes

    Maybe later support for cross-validation is added 
  """
  def __init__(self, augmentations, seed=42, robust_training=False):

    # Define fields
    self.augmentations=augmentations # Copied
    self.list_augmentations = list(augmentations.keys())
    self.calibration_data = None
    self.testing_data = None
    self.accuracies = {}
    self.seed = seed

    # Base settings
    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    BATCH_SIZE = 64
    ITERATIONS = 2
    PROB_CALIBRATION = 0.2

    self.classes = ('zero', 'One', 'Two', 'Three', 'Four','Five','Six', 'Seven', 'Eight', 'Nine')

    self.basepath = os.path.dirname(__file__)

    model_specification = "_" + "_".join(f"{k}{v}" for k, v in self.augmentations.items())

    if robust_training:
        self.path_saved_model = f"benchmarks/cnn_mnist_robust_{DEVICE}_SEED{seed}{model_specification}.pt"
    else:
        self.path_saved_model = f"benchmarks/cnn_mnist_{DEVICE}_SEED{seed}{model_specification}.pt"
    
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])

    # Read Dataset: Training & Testing
    trainset = torchvision.datasets.MNIST(root=self.basepath+'/data', train=True,
                                            download=True,transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                                   # pin_memory=True,
                                              shuffle=False, num_workers=2)

    testset = torchvision.datasets.MNIST(root=self.basepath+'/data', train=False,
                                           download=True,transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                                   # pin_memory=True,
                                             shuffle=False, num_workers=2)


    # Train Model
    model = MNistNet().to(DEVICE)

    if not os.path.isfile(self.path_saved_model):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        for epoch in range(ITERATIONS):  # loop over the dataset multiple times
            # model = model.to(DEVICE)
            for i, (data, target) in enumerate(trainloader):
                if robust_training:
                    random_aug_name = self.list_augmentations[random.randint(0, len(self.list_augmentations)-1)]
                else:
                    random_aug_name = Augmentation.ORIGINAL.value
                # print(random_aug_name)
                data = self.apply_aug(data=data,
                                      augmentation_name=random_aug_name)
                inputs, labels = data.to(DEVICE), target.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # # print statistics
                if i % 100 == 0:
                    print(f'[{epoch}, batch:{i}] loss: {loss.item():.3f}')# and accuracy: {accur}')
            scheduler.step()
        torch.save(model.state_dict(), self.path_saved_model)
    else:
        print("Trained model is loaded from: ", self.path_saved_model)
        state_dict = torch.load(self.path_saved_model, weights_only=True)
        model.load_state_dict(state_dict)

    model.eval()

    # Split Testing set into conformance calibration & conformance testing set
    generator = torch.Generator().manual_seed(self.seed)
    calibration_set, rest_of_testing_set = torch.utils.data.random_split(testset, [PROB_CALIBRATION,1.0-PROB_CALIBRATION],generator)


    # Compute probability distribution/class 
    self.calibration_data = []
    self.testing_data = []
    self.precalibration_data = []

    # TODO: Only for test purpose. I should remove it afterwards. I should use the whole training set for precalibration.

    # subset_size = int(0.0001 * len(trainset))  # 600
    # indices = torch.randperm(len(trainset))[:subset_size]

    # trainset_small = torch.utils.data.Subset(trainset, indices)

    # trainloader_small = torch.utils.data.DataLoader(trainset_small, batch_size=BATCH_SIZE,
    #                                           # pin_memory=True,
    #                                           shuffle=True, num_workers=2)


    softmax = torch.nn.Softmax(dim=1)
    model.eval()

    for (inputSet, outputList) in [
        (calibration_set, self.calibration_data),
        (rest_of_testing_set, self.testing_data),
        (trainset, self.precalibration_data),
    ]:
        loader = torch.utils.data.DataLoader(
            inputSet,
            batch_size=128,  # >1 für Effizienz, nach Bedarf anpassen
            # pin_memory=True,
            shuffle=False,
            num_workers=1,  # ggf. anpassen
        )

        with torch.no_grad():
            for data, target in loader:
                data = data.to(DEVICE, non_blocking=True)
                target_np = target.cpu().numpy()

                B = data.shape[0]
                A = len(self.list_augmentations)

                aug_batches = []
                for augmentation_name in self.list_augmentations:
                    dataAug = self.apply_aug(data=data, augmentation_name=augmentation_name)
                    # print('dataAug: ', dataAug.shape)
                    aug_batches.append(dataAug)  # jede: (B, C, H, W)

                # 2) Augmentierte Batches konkatenieren und Modell einmal ausführen
                big_batch = torch.cat(aug_batches, dim=0)  # (A*B, C, H, W)
                outputs = softmax(model(big_batch))  # (A*B, K)

                # 3) In Form (A, B, K) bringen und auf CPU/Numpy
                outputs_np = outputs.view(A, B, -1).cpu().numpy()

                # 4) Gleiche Struktur wie vorher: pro Sample eine Liste von Verteilungen
                for j in range(B):
                    these_distributions = [outputs_np[a, j, :] for a in range(A)]
                    outputList.append((these_distributions, target_np[j]))

    model.eval()


    for (loader, dataset) in [(trainloader, "Training"), (testloader, "Testing")]:
        # Zähler pro Augmentation und Klasse
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
                # data: (B, C, H, W), target: (B,)
                data = data.to(DEVICE, non_blocking=True)
                labels = target.to(DEVICE)
                B = data.shape[0]
                A = len(self.list_augmentations)

                # 1) Alle Augmentierungen auf den gesamten Batch anwenden
                aug_batches = []
                for augmentation_name in self.list_augmentations:
                    dataAug = self.apply_aug(data=data, augmentation_name=augmentation_name)
                    aug_batches.append(dataAug)  # jede: (B, C, H, W)

                # 2) Zu einem großen Batch zusammenfassen und Modell einmal ausführen
                big_batch = torch.cat(aug_batches, dim=0)  # (A*B, C, H, W)
                outputs = model(big_batch)  # (A*B, K)
                _, predictions = torch.max(outputs, 1)  # (A*B,)

                # 3) In (A, B) umformen: pro Augmentation ein Vorhersage-Vektor
                predictions = predictions.view(A, B)

                # 4) Zählen pro Augmentation und Klasse (wie vorher, nur gebündelt)
                for a_idx, augmentation_name in enumerate(self.list_augmentations):
                    preds_a = predictions[a_idx]  # (B,)
                    for label, prediction in zip(labels, preds_a):
                        cls_name = self.classes[label]
                        if label == prediction:
                            correct_pred[augmentation_name][cls_name] += 1
                        total_pred[augmentation_name][cls_name] += 1

        # 5) Auswertung / Ausgabe wie vorher, aber pro Augmentation
        self.accuracies[dataset] = []

        # init / reset class_accuracies for this dataset
        self.class_accuracies = getattr(self, "class_accuracies", {})
        self.class_accuracies[dataset] = {}  # reset to avoid duplicates across runs

        for augmentation_name in self.list_augmentations:
            print("==========[Accuracy on the", dataset,
                  "dataset (Augmentation:", augmentation_name, ")]==========")

            # init list for this augmentation
            self.class_accuracies[dataset][augmentation_name] = []

            for cls_name in self.classes:
                c = correct_pred[augmentation_name][cls_name]
                t = total_pred[augmentation_name][cls_name]

                if t > 0:
                    accuracy = round(100.0 * c / t, 2)
                else:
                    accuracy = 0.0

                print("Class", cls_name, ":", accuracy, "% (", t, "cases)")

                self.class_accuracies[dataset][augmentation_name].append((cls_name, accuracy, t))

            total_correct = sum(correct_pred[augmentation_name].values())
            total_cases = sum(total_pred[augmentation_name].values())
            overall = round(100.0 * total_correct / total_cases, 2) if total_cases > 0 else 0.0

            self.accuracies[dataset].append(overall)
            print("Overall accuracy:", overall, "% (", total_cases, "cases)")

        print("\n")


  def apply_aug(self, data, augmentation_name):

      if augmentation_name in self.list_augmentations:
          if augmentation_name == Augmentation.ROTATION.value:
              return self._apply_rotation(data, self.augmentations[augmentation_name])
          if augmentation_name == Augmentation.TRANSLATION.value:
              return self._apply_translation(data, self.augmentations[augmentation_name])
          if (augmentation_name == Augmentation.ORIGINAL.value or
              (augmentation_name == Augmentation.HORIZONTAL_FLIP.value and not self.augmentations[augmentation_name]) or
              (augmentation_name == Augmentation.VERTICAL_FLIP.value and not self.augmentations[augmentation_name])):
              return data
          if augmentation_name == Augmentation.HORIZONTAL_FLIP.value and self.augmentations[augmentation_name]:
              return self._apply_horizontal_flip(data)
          if augmentation_name == Augmentation.VERTICAL_FLIP.value and self.augmentations[augmentation_name]:
              return self._apply_vertical_flip(data)
          if augmentation_name == Augmentation.BLUR.value:
              return self._apply_gaussian_blur(data, self.augmentations[augmentation_name])
          if augmentation_name == Augmentation.RANDOM_RESIZED_CROP.value:
              return self._apply_random_resized_cropping(data, self.augmentations[augmentation_name])
          if augmentation_name == Augmentation.MIXUP.value:
              return self._apply_mixup(data)
          if augmentation_name == Augmentation.CONTRAST.value:
              return self._apply_contrast(data, contrast_factor=self.augmentations[augmentation_name])
          if augmentation_name == Augmentation.BRIGHTNESS.value:
              return self._apply_brightness(data, brightness_factor=self.augmentations[augmentation_name])
      else:
         raise NotImplementedError


  def _apply_mixup(self, data):
      angle_deg = self.augmentations[Augmentation.ROTATION.value]
      translation_factor = self.augmentations[Augmentation.TRANSLATION.value]
      return torchvision.transforms.functional.affine(img=data, translate=translation_factor, angle=angle_deg, shear=0.0, scale=1.0)

  def _apply_gaussian_blur(self, data, kernel_size):
      # return torchvision.transforms.v2.functional.gaussian_blur(inpt=data, kernel_size=kernel_size)
      return torchvision.transforms.functional.gaussian_blur(img=data, kernel_size=kernel_size)

  def _apply_rotation(self, x: torch.Tensor, angle_deg: float) -> torch.Tensor:
      return torchvision.transforms.functional.affine(img=x, translate=(0, 0), angle=angle_deg, shear=0.0, scale=1.0)

  def _apply_translation(self, x: torch.Tensor, translation_factor) -> torch.Tensor:
      # B, C, H, W = x.shape
      # return torchvision.transforms.v2.functional.affine(inpt=x, translate=translation_factor,angle=0.0, shear=0.0, scale=1.0)
      return torchvision.transforms.functional.affine(img=x, translate=translation_factor,angle=0.0, shear=0.0, scale=1.0)

  def _apply_horizontal_flip(self, x: torch.Tensor) -> torch.Tensor:
      # return torchvision.transforms.v2.functional.horizontal_flip(inpt=x)
      return torchvision.transforms.functional.hflip(img=x)

  def _apply_vertical_flip(self, x: torch.Tensor) -> torch.Tensor:
      # return torchvision.transforms.v2.functional.vertical_flip(inpt=x)
      return torchvision.transforms.functional.vflip(img=x)

  def _apply_random_resized_cropping(self, x: torch.Tensor, new_shape) -> torch.Tensor:
      # return torchvision.transforms.v2.RandomResizedCrop(size=new_shape,)(x)
      return torchvision.transforms.RandomResizedCrop(size=new_shape,)(x)

  def _apply_contrast(self, data: torch.Tensor, contrast_factor: float) -> torch.Tensor:
      '''0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by a factor of 2.'''
      return torchvision.transforms.functional.adjust_contrast(img=data, contrast_factor=contrast_factor)

  def _apply_brightness(self, data: torch.Tensor, brightness_factor: float) -> torch.Tensor:
      '''0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by a factor of 2.'''
      return torchvision.transforms.functional.adjust_brightness(img=data, brightness_factor=brightness_factor)


  def _texClassAccuracies(self, dataset: str) -> str:
      data = self.class_accuracies.get(dataset, {})
      augs = list(self.list_augmentations)

      # Build lookup: acc_map[aug][cls] = (acc, n)
      acc_map = {aug: {} for aug in augs}
      for aug in augs:
          for cls, acc, n in data.get(aug, []):
              acc_map[aug][cls] = (float(acc), int(n))

      # Determine class order
      classes = list(getattr(self, "classes", []))
      if not classes:
          # fallback: infer from data
          classes = sorted({cls for aug in augs for cls, _, _ in data.get(aug, [])})

      # Column spec: Class | N | Aug1 | Aug2 | ...
      colspec = "l|r|" + "|".join(["r"] * len(augs))

      # Header with a grouped "Accuracies" spanning all augmentation columns
      header = []
      header.append(f"\\begin{{tabular}}{{{colspec}}}")
      header.append("\\hline")
      header.append(
          "\\textbf{Class} & \\textbf{N} & "
          + f"\\multicolumn{{{len(augs)}}}{{c}}{{\\textbf{{Accuracies}}}} \\\\ \\hline"
      )
      header.append(
          " &  & " + " & ".join([f"\\textbf{{{aug}}}" for aug in augs]) + " \\\\ \\hline"
      )

      # Rows
      rows = []
      for cls in classes:
          # Pick N from the first augmentation where it's available (counts should usually match)
          n_val = ""
          for aug in augs:
              if cls in acc_map[aug]:
                  n_val = str(acc_map[aug][cls][1])
                  break

          acc_vals = []
          for aug in augs:
              if cls in acc_map[aug]:
                  acc_vals.append(f"{acc_map[aug][cls][0]:.2f}\\%")
              else:
                  acc_vals.append("--")

          rows.append(f"{cls} & {n_val} & " + " & ".join(acc_vals) + " \\\\ \\hline")

      if not rows:
          rows = [f"\\multicolumn{{{2 + len(augs)}}}{{c}}{{No class accuracies available}} \\\\ \\hline"]

      return "\n".join(header + rows + ["\\end{tabular}"])

  def texInfo(self):
    text1 =  """
        \\begin{{tabular}}{{l|l}} \\\\
        \\textbf{{Benchmark name:}} & MNIST \\\\ \\hline
        \\textbf{{Augmentations:}} & {0} \\\\ \\hline
        \\textbf{{Accuracy Training set:}} & {1} \\% \\\\ \\hline
        \\textbf{{Accuracy Testing set:}} & {2} \\% \\\\ 
        \\end{{tabular}}
    """.format(list(self.augmentations.values()),self.accuracies["Training"],self.accuracies["Testing"])

    final = """
    \\section{Accuracies Per Class on Training Set}
     """
    final += self._texClassAccuracies("Training")

    final += """
    \\newpage
    \\section{Accuracies Per Class on Test Set}
     """

    final += self._texClassAccuracies("Testing")

    return text1 + final
