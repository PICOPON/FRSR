from MTHead.head import MTHead
from RPN.rpn import FRPN
from SRCNN.srcnn import SRCNN
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    img = torch.randn(1, 3, 416, 416)

    frpn_net = FRPN()
    frpn_net.load_state_dict(torch.load("./RPN/rpn_saved.pth"))

    srcnn_net = SRCNN(num_channel=3)
    srcnn_net.load_state_dict(torch.load("./SRCNN/srcnn_saved.pth"))

    MTHead_net = MTHead()
    MTHead_net.load_state_dict(torch.load("./MTHead/head_saved.pth"))


    _,  _,  _, n_rois = frpn_net(img)

    for n in range(len(n_rois)):   # 第n张图的rois
        plt.matshow(img[n, 0, ...])
        for roi in n_rois[n]:
            roi_x0, roi_y0, roi_x1, roi_y1 = round(roi[1]), round(roi[0]), round(roi[3]), round(roi[2])
            roi_patch = img[n, :, roi_x0:roi_x1, roi_y0:roi_y1]
            roi_patch = roi_patch.reshape(1, roi_patch.shape[0], roi_patch.shape[1], roi_patch.shape[2])

            if roi_patch.shape[1] and roi_patch.shape[2]:
                # roi区域全部resize到固定大小
                roi_patch = F.interpolate(roi_patch, size=(64, 64), mode='bicubic', align_corners=False)
                # [1, 3, 64, 64]

                # roi 区域超分增强特征
                roi_sr_patch = srcnn_net(roi_patch)

                # roi 区域特征分类
                cls_pred, loc_pred = MTHead_net(roi_sr_patch)
                print(cls_pred, loc_pred)

                xa, ya, wa, ha = roi[1], roi[0], roi[3] - roi[1], roi[2] - roi[0]
                tx, ty, tw, th = loc_pred[0, 0].detach().numpy(), loc_pred[0, 1].detach().numpy(), loc_pred[0, 2].detach().numpy(), loc_pred[0, 3].detach().numpy()
                x_t, y_t, w_t, h_t = xa + wa * tx, ya + ha * ty, wa * np.exp(tw), ha * np.exp(th)

                roi_pred = (y_t, x_t, y_t + h_t, x_t + w_t)

                plt.gca().add_patch(plt.Rectangle((roi_pred[1], roi_pred[0]), roi_pred[3] - roi_pred[1],
                                                  roi_pred[2] - roi_pred[0], fill=False,
                                                  edgecolor='r', linewidth=3))
        plt.show()

