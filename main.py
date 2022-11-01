import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


net = torchvision.models.vgg16(pretrained=True)

x = torch.rand(1, 3, 32, 32)
print(net(x).shape)