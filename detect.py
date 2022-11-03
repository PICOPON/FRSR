from MTHead.MTHead import MTHead
from RPN.rpn import FRPN
from SRCNN.srcnn import SRCNN
import torch


def f(roi, cls_pred, loc_pred):
    return [0,0,0,0]


if __name__ == '__main__':
    img = torch.randn(1, 3, 416, 416)
    frpn_net = FRPN()
    srcnn_net = SRCNN(num_channel=3)
    MTHead_net = MTHead()
    _, _, _, rois = frpn_net(img)
    for roi in rois:
        cls_pred, loc_pred = MTHead_net(srcnn_net(img(roi)))
        out_box = f(roi, cls_pred, loc_pred)
