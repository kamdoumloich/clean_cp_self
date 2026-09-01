import os

import torch
import torch.nn as nn
import torchvision

from torchvision.models import ResNet50_Weights, ResNet18_Weights

# from benchmarks.vision_classification import VisionClassificationBenchmark

class ResNet(nn.Module):
    ''' The pretrained model outputs logits, not directly probability distributions.
        ResNet50_Weights.IMAGENET1K_V1: --- The accuracies correspond to those of TTA paper on page 4.
        acc@1 (on ImageNet-1K)  76.13
        acc@5 (on ImageNet-1K)  92.862
        https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
    '''
    def __init__(self, num_classes, pretrained=True, architecture='resnet50', weights="IMAGENET1K_V1"):
        super(ResNet, self).__init__()
        if architecture not in ['resnet18', 'resnet50']:
            raise NotImplementedError
        
        self.model = self._create_resnet(pretrained=pretrained, architecture=architecture, weights=weights)

        if num_classes != 1000:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
    @staticmethod
    def _create_resnet(pretrained=True, architecture='resnet50', weights="IMAGENET1K_V1"):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        if weights =="IMAGENET1K_V1" and architecture=='resnet50' and pretrained:
            return torchvision.models.get_model(architecture, weights=ResNet50_Weights.IMAGENET1K_V1)
        elif weights =="IMAGENET1K_V1" and architecture=='resnet18' and pretrained:
            return torchvision.models.get_model(architecture, weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ModuleNotFoundError
