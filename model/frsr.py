import torch
import torch.nn as nn
import torchvision
from RPN.rpn import FRPN
from SRCNN.srcnn import SRCNN


# FRSR主模型结构构建
class FRSR(nn.Module):
    def __init__(self):
        super(FRSR, self).__init__()
        self.rpn_front = FRPN()
        self.rpn_front.load_state_dict(torch.load("../RPN/rpn_saved.pth"))
        self.sr_cnn = SRCNN(num_channel=3)
        self.sr_cnn.load_state_dict(torch.load('../SRCNN/srcnn_saved.pth'))
        self.cls_end = torchvision.models.densenet121(pretrained=True)
        self.cls_end.classifier.add_module('out', nn.Linear(1000, 2))
        self.threshold = 0.9

    def forward(self, x):
        _, _, _, rois = self.rpn_front(x)
        obj_rois = []
        for n in range(len(rois)):
            for b in rois[n]:
                roi_x0, roi_y0, roi_x1, roi_y1 = int(b[1]), int(b[0]), int(b[3]), int(b[2])
                x_patch = x[n, :, roi_x0:roi_x1, roi_y0:roi_y1]
                roi_img = self.sr_cnn(x_patch.reshape(1, x_patch.shape[0],  x_patch.shape[1],  x_patch.shape[2]))
                out = self.cls_end(roi_img)
                if out[0] > self.threshold:
                    obj_rois.append(roi_img)
        return obj_rois

