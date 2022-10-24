import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# net = torchvision.models.vgg16(pretrained=True)
# net.classifier.add_module('add_linear', nn.Linear(1000, 10))

densenet = torchvision.models.vgg16()
densenet.classifier[6].out_features = 2
print(densenet)
x = torch.rand(1, 3, 1800, 1800)
print(densenet.classifier[6](x).shape)