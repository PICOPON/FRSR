import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image, ImageReadMode


or_img_tensor = (read_image('test_m.jpg', ImageReadMode.RGB) / 255.).reshape(1, 3, 256, 256).to(device='cuda:0')
net = torch.load('srcnn.pth').cuda()

out = net(or_img_tensor)

plt.imshow(out[0,...].permute(1,2,0).detach().cpu())
plt.show()

