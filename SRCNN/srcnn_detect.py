from srcnn import SRCNN
import torch
import cv2
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(40, 20))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
net = SRCNN(num_channel=3)

image = torch.tensor(cv2.imread("./test.jpg", cv2.IMREAD_UNCHANGED)).permute(2,0,1) /255.
image = image.view(1, *image.size())

net.load_state_dict(torch.load('srcnn_saved.pth'))

y = net(image)

ax1.imshow(image[0, 0, ...])
ax2.imshow(y[0,0,...].detach().numpy())
plt.show()