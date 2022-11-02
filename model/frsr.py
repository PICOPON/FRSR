import torch
import torch.nn as nn
import torchvision
from RPN.rpn import FRPN
from SRCNN.srcnn import SRCNN
import torch.nn.functional as F

import cv2
import numpy as np
import matplotlib.pyplot as plt

# FRSR主模型结构构建
class FRSR(nn.Module):
    def __init__(self):
        super(FRSR, self).__init__()
        self.rpn_front = FRPN()
        self.rpn_front.load_state_dict(torch.load("../RPN/rpn_saved.pth"))
        self.sr_cnn = SRCNN(num_channel=3)
        self.sr_cnn.load_state_dict(torch.load('../SRCNN/srcnn_saved.pth'))
        self.cls_end = Head()

        self.cls_threshold = 0.2

    def forward(self, x):
        _, _, _, rois = self.rpn_front(x)
        obj_rois = []
        for n in range(len(rois)):
            for roi in rois[n]:
                roi_x0, roi_y0, roi_x1, roi_y1 = round(roi[1]), round(roi[0]), round(roi[3]), round(roi[2])
                x_patch = x[n, :, roi_x0:roi_x1, roi_y0:roi_y1]
                x_patch = x_patch.reshape(1, x_patch.shape[0],  x_patch.shape[1],  x_patch.shape[2])

                # roi区域全部resize到固定大小
                x_patch = F.interpolate(x_patch, size=(64,64), mode='bicubic', align_corners=False)   # [1,3,64,64]

                # roi 区域超分增强特征
                roi_img = self.sr_cnn(x_patch)

                # roi 区域特征分类
                cls_pred = self.cls_end(roi_img)
                if cls_pred[0, 0] > self.cls_threshold:
                    obj_rois.append(roi_img)

        return obj_rois


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.fc_cls = nn.Linear(1000, 2) # 2个类别，类别概率

    def forward(self, x):
        x = self.resnet50(x)
        return F.softmax(self.fc_cls(x), dim=1)



