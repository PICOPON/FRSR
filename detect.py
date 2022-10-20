import torchvision
import torch

net = torchvision.models.vgg16(pretrained=True)
for name, module in net.named_modules():
    if name == 'features':
        new_net = module
        print(module)
        x = torch.randn(1, 3, 416, 416)
        print(new_net(x).shape)


