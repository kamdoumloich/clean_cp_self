import random

import numpy as np
import torch
import torch.nn as nn
import os
import torchvision
import numpy
import matplotlib.pyplot
"""
MNIST Benchmark for the Conformace Prediction Framework
"""

# Reason to use model.eval() and model.train() in PyTorch
# https://pytorch.org/docs/stable/notes/autograd.html#evaluation-mode-nn-module-eval
class CIFARNet(nn.Module):
    def __init__(self):
        super(CIFARNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class CIFARBenchmark:
    """
    The CIFAR10 Dataset as data generator for the conformance prediction framework

    Attributes:
        augmentations        A list of all augmentation values used when generating the probability vectors
        calibration_data     A list of pairs of probability distributions (one for each augmentation) and the corresponding classes, calibration set for the conformance predictor
        testing_data         A list of pairs of probability distributions (one for each augmentation) and the corresponding classes, testing set for the conformance predictor
        classes              Names for all classes

    Maybe later support for cross-validation is added 
    """

    def __init__(self, augmentations, seed=777):

        self.augmentations = augmentations  # Copied
        self.calibration_data = None
        self.testing_data = None
        self.accuracies = {}
        self.seed = seed

        # Base settings
        DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
        # torch.manual_seed(0)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        BATCH_SIZE = 64
        ITERATIONS = 2
        PROB_CALIBRATION = 0.2
        self.classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
        self.path_saved_model = "benchmarks/cnn_cifar10.pt"
        self.distribution_per_sets = {"Training":[], "Testing":[]}
        self.class_accuracies_by_aug_dataset = {}

        self.basepath = os.path.dirname(__file__)
        self.path_saved_model = f"benchmarks/cnn_cifar_{seed}.pt"

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                ])

        # Read Dataset: Training & Testing
        trainset = torchvision.datasets.CIFAR10(root=self.basepath+'/data', train=True,
                                                download=True,transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                                        pin_memory=True,
                                                    shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=self.basepath+'/data', train=False,
                                                download=True,transform=transform)

        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                                        pin_memory=True,
                                                    shuffle=False, num_workers=2)
        
        # self._execute_functions()

        # Train Model

        model = CIFARNet().to(DEVICE)

        if not os.path.isfile(self.path_saved_model):
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

            for epoch in range(ITERATIONS):  # loop over the dataset multiple times
                net = model.to(DEVICE)
                for i, (data, target) in enumerate(trainloader):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data.to(DEVICE), target.to(DEVICE)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # # print statistics
                    if i % 100 == 0:
                        print(f'[{epoch}, batch:{i}] loss: {loss.item():.3f}')  # and accuracy: {accur}')
                scheduler.step()
            torch.save(model.state_dict(), self.path_saved_model)
        else:
            print("Trained model is loaded from: ", self.path_saved_model)
            state_dict = torch.load(self.path_saved_model, weights_only=True)
            model.load_state_dict(state_dict)

        model.eval()

        # Checking Accuracy
        for (loader, dataset) in [(trainloader, "Training"), (testloader, "Testing")]:
            correct_pred = {classname: 0 for classname in self.classes}
            total_pred = {classname: 0 for classname in self.classes}
            for i, (data, target) in enumerate(loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data.to(DEVICE), target.to(DEVICE)
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[self.classes[label]] += 1
                    total_pred[self.classes[label]] += 1
            print("==========[Accuracy on the", dataset, "dataset]==========")
            for (a, b) in correct_pred.items():
                print("Class", a, ": ", 100 * b / total_pred[a], "% (", total_pred[a], "cases)")
            self.accuracies[dataset] = 100 * sum([b for (a, b) in correct_pred.items()]) / sum(
                [b for (a, b) in total_pred.items()])
            print("Overall accuracy:", self.accuracies[dataset], "% (", sum([b for (a, b) in total_pred.items()]), "cases)")

            # Split Testing set into conformance calibration & conformance testing set
        generator = torch.Generator().manual_seed(2025)
        calibration_set, rest_of_testing_set = torch.utils.data.random_split(testset,
                                                                             [PROB_CALIBRATION, 1.0 - PROB_CALIBRATION],
                                                                             generator)

        # Compute probability distribution/class
        self.calibration_data = []
        self.testing_data = []
        for (inputSet, outputList) in [(calibration_set, self.calibration_data), (rest_of_testing_set, self.testing_data)]:
            loader = torch.utils.data.DataLoader(inputSet, batch_size=1,
                                                 pin_memory=True, shuffle=False, num_workers=1)
            for i, (data, target) in enumerate(loader):
                these_distributions = []
                for augmentation in augmentations:
                    # Augment
                    with torch.no_grad():
                        dataAug = (1.0 - augmentation) * data + augmentation  # Brightness augmentation
                        inputs, labels = dataAug.to(DEVICE), target.to(DEVICE)
                        outputs = torch.nn.Softmax(dim=1)(model(inputs))
                        assert outputs.shape[0] == 1
                        these_distributions.append(outputs.cpu().numpy()[0, :])
                outputList.append((these_distributions, target.cpu().numpy()[0]))

        # Compute probability distribution/class PRECALIBRATION
        self.precalibration_data = []
        # for (inputSet,outputList) in [(calibration_set,self.precalibration_data)]:
        # TODO: Here I must use the training set
        for (inputSet, outputList) in [(trainset, self.precalibration_data)]:
            loader = torch.utils.data.DataLoader(inputSet, batch_size=1,
                                                 pin_memory=True, shuffle=False, num_workers=1)
            for i, (data, target) in enumerate(loader):
                these_distributions = []
                for augmentation in augmentations:
                    # Augment
                    with torch.no_grad():
                        dataAug = (1.0 - augmentation) * data + augmentation  # Brightness augmentation
                        inputs, labels = dataAug.to(DEVICE), target.to(DEVICE)
                        outputs = torch.nn.Softmax(dim=1)(model(inputs))
                        assert outputs.shape[0] == 1
                        these_distributions.append(outputs.cpu().numpy()[0, :])
                outputList.append((these_distributions, target.cpu().numpy()[0]))


    def texInfo(self):
        return """
              \\begin{{tabular}}{{l|l}} \\\\
              \\textbf{{Benchmark name:}} & MNIST \\\\ \\hline
              \\textbf{{Augmentations:}} & {0} \\\\ \\hline
              \\textbf{{Accuracy Training set:}} & {1} \\% \\\\ \\hline
              \\textbf{{Accuracy Testing set:}} & {2} \\% \\\\ 
              \\end{{tabular}}
          """.format(self.augmentations, self.accuracies["Training"], self.accuracies["Testing"])
