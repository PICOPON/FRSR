import sys
sys.path.append("../RPN")
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from head import MTHead
from SRCNN.srcnn import SRCNN
from datasets import BBoxData

from RPN.rpn_train import iou_compute
from RPN.rpn import FRPN

# 计算硬件
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# MTHead网络 loss 计算
def MTHead_Loss_Compute(bboxes, roi, cls_pred, loc_pred, iou_threshold, loss_fn = torch.nn.MSELoss()):
    # bboxes[m, 5] , roi[4], cls_pred [1,?], loc_pred [1,4]
    m, _ = bboxes.shape
    roi_label = torch.zeros_like(cls_pred)
    cls_loss, loc_loss = torch.tensor(0.), torch.tensor(0.)

    xa, ya, wa, ha = roi[1], roi[0], roi[3] - roi[1], roi[2] - roi[0]
    tx, ty, tw, th = loc_pred[0, 0], loc_pred[0, 1], loc_pred[0, 2], loc_pred[0, 3]
    x_t, y_t, w_t, h_t = xa + wa * tx, ya + ha * ty, wa * torch.exp(tw), ha * torch.exp(th)

    roi_pred = (y_t, x_t, y_t + h_t, x_t + w_t)

    ious_m = []
    for i in range(m):
        ious_m.append(iou_compute(roi, bboxes[i, 1:]))
    index_max_iou = ious_m.index(max(ious_m))
    if  ious_m[index_max_iou] > iou_threshold:
        roi_label[0, int(bboxes[index_max_iou, 0])] = 1
        loc_loss = (1 - ious_m[index_max_iou])**2

    cls_loss = loss_fn(roi_label, cls_pred)

    return cls_loss, loc_loss


# 数据定义加载
BboxData_dataset = BBoxData('E:\\dataset\\DJI_0030\\DJI_0030\\images',
                            'E:\\dataset\\DJI_0030\\DJI_0030\\labels')

BboxData_loader = DataLoader(BboxData_dataset, 1)

# 已训练模型
frpn_net = FRPN().cuda(device=device)
frpn_net.load_state_dict(torch.load("../RPN/rpn_saved.pth"))

srcnn_net = SRCNN(num_channel=3).cuda(device=device)
srcnn_net.load_state_dict(torch.load('../SRCNN/srcnn_saved.pth'))

# 待训练
MTHead_net = MTHead().cuda(device=device)

# 误差梯度反向传播
optim = optim.SGD(MTHead_net.parameters(), lr=0.001, momentum=0.9)

cls_iou_threshold = 0.2

MTHead_net.train()
for e in range(10):
    for b_i, (img, bboxes) in enumerate(BboxData_loader):
        img, bboxes = img.to(device, dtype=torch.float32), bboxes.to(device, dtype=torch.float32)
        if bboxes.shape[1]:     # 如果有目标
            MTHead_net.zero_grad()
            _, _, _, rois = frpn_net(img)
            cls_n_loss, loc_n_loss = torch.tensor(0.).cuda(device=device), torch.tensor(0.).cuda(device=device)
            for n in range(len(rois)):
                # 第n张图的rois
                for roi in rois[n]:
                    roi_x0, roi_y0, roi_x1, roi_y1 = round(roi[1]), round(roi[0]), round(roi[3]), round(roi[2])
                    roi_patch = img[n, :, roi_x0:roi_x1, roi_y0:roi_y1]
                    roi_patch = roi_patch.reshape(1, *roi_patch.size())

                    if roi_patch.shape[1] and roi_patch.shape[2]:
                        # roi区域全部resize到固定大小
                        roi_patch = F.interpolate(roi_patch, size=(64, 64), mode='bicubic', align_corners=False)
                        # [1, 3, 64, 64]

                        # roi 区域超分增强特征
                        roi_sr_patch = srcnn_net(roi_patch)

                        # roi 区域特征分类
                        cls_pred, loc_pred = MTHead_net(roi_sr_patch)

                        # 损失计算
                        cls_loss, loc_loss = MTHead_Loss_Compute(bboxes[n, ...], roi, cls_pred, loc_pred,
                                                                 cls_iou_threshold)
                        cls_n_loss += cls_loss ** 2
                        loc_n_loss += loc_loss ** 2

            print(f"batch index: {b_i} cls_n_loss: {cls_n_loss}, loc_n_loss: {loc_n_loss}")
            loss = cls_n_loss**2 + loc_n_loss**2

            loss.backward()
            optim.step()

        if b_i % 20 == 0:
            print("model saved")
            torch.save(MTHead_net.state_dict(), 'head_saved.pth')






