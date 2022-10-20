import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

from FRPN import FRPN
from SRCNN.SRCNN import SRCNN
from datasets import CoCo


# FRSR主模型结构构建
class FRSR(nn.Module):
    def __init__(self):
        super(FRSR, self).__init__()
        self.rpn_front = FRPN()
        self.sr_cnn = SRCNN(3)
        # self.dt_end = YOLO() # YOLO 等单步探测模型

    def forward(self, x):
        _, _, _, rois = self.rpn_front(x)
        roi_imgs = []
        for n in rois:
            for b in n:
                roi_x0, roi_y0, roi_x1, roi_y1 = int(b[1]), int(b[0]), int(b[3]), int(b[2])
                x_patch = x[:, :, roi_x0:roi_x1, roi_y0:roi_y1]
                roi_img = self.sr_cnn(x_patch)
                roi_imgs.append(roi_img)
        # out_box = self.dt_end(roi_imgs)
        return roi_imgs  # out_box


# 数据定义加载
coco_dataset = CoCo('../../datasets/Mines.v2i.yolov5pytorch/train/images/',
                    '../../datasets/Mines.v2i.yolov5pytorch/train/labels/')

coco_loader = DataLoader(coco_dataset, 1)


# 模型定义
net = FRSR()
print(net)

# 冻结部分模型参数
net.rpn_front.requires_grad = False
net.sr_cnn.requires_grad = False

# 误差梯度反向传播
# optim = optim.SGD(net.dt_end.parameters(), lr=0.001, momentum=0.9)

net.train()
for e in range(10):
    for img, bboxes in coco_loader:
        if bboxes.shape[1]:
            net.zero_grad()
            # 损失计算
            y = net(img)
            plt.matshow(y[0][0, 0, :, :].detach().numpy())  # 第一个roi推荐区域的
            plt.show()
            break
        break
    break
