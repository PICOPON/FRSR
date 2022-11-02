import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

net = torchvision.models.resnet50(pretrained=False)
print(net)
x = torch.rand(1, 3, 46, 46)
print(net(x))

print(round(11.5))