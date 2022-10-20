import torch
import torch.nn as nn
from SRCNN.SRCNN import SRCNN
from FRPN import FRPN
from datasets import CoCo
from torch.utils.data import DataLoader
import torch.optim as optim


class FRSR(nn.Module):
    def __init__(self):
        super(FRSR, self).__init__()
        self.rpn_front = FRPN()
        self.sr_cnn = SRCNN(3)
        self.dt_end = None

    def forward(self, x):
        _, _, _, rois = self.rpn_front(x)
        out = None
        for n in rois:
            for b in n:
                roi_x0, roi_y0, roi_x1, roi_y1 = int(b[1]), int(b[0]), int(b[3]), int(b[2])
                x_patch = x[:, :, roi_x0:roi_x1, roi_y0:roi_y1]
                out = self.sr_cnn(x_patch)
                break
        return out


coco_dataset = CoCo('../datasets/Mines.v2i.yolov5pytorch/train/images/',
                    '../datasets/Mines.v2i.yolov5pytorch/train/labels/')

coco_loader = DataLoader(coco_dataset, 1)

# img = torch.randn(1, 3, 640, 640)
# bboxes = torch.tensor([[[100, 200, 30, 30], [100, 100, 100, 100], [200, 200, 100, 100]]])

# 模型定义
net = FRSR()

x = torch.rand(1, 3, 416, 416)
y = net(x)
print(y.shape)

'''
# 冻结部分模型参数
net.backbone.requires_grad = False

# 误差梯度反向传播
optim = optim.SGD(net.rpn.parameters(), 0.001)

net.train()
for e in range(10):
    for img, bboxes in coco_loader:
        if bboxes.shape[1]:
            net.zero_grad()
            # 损失计算
            rpn_fg_scores, rpn_locs, anchors, rois = net(img)
'''