import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
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
        self.cls_end = torchvision.models.densenet121(pretrained=False)
        self.cls_end.classifier.add_module('out', nn.Linear(1000, 2))
        self.threshold = 0.9

    def forward(self, x):
        _, _, _, rois = self.rpn_front(x)
        obj_rois = []
        roi_imgs = []
        for n in range(len(rois)):
            for b in rois[n]:
                roi_x0, roi_y0, roi_x1, roi_y1 = int(b[1]), int(b[0]), int(b[3]), int(b[2])
                x_patch = x[n, :, roi_x0:roi_x1, roi_y0:roi_y1]
                roi_img = self.sr_cnn(x_patch.reshape(1, x_patch.shape[0],  x_patch.shape[1],  x_patch.shape[2]))
                out = self.cls_end(roi_img)
                if out[0] > self.threshold:
                    obj_rois.append(roi_img)

        return obj_rois


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
            plt.matshow(y[0][0, 0, :, :].detach().numpy())   # 第一个roi推荐区域的显示
            plt.show()
            break
        break
    break
