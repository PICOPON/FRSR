import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from MTHead import MTHead
from RPN.rpn import FRPN
from SRCNN.srcnn import SRCNN
from datasets import BBoxData
from RPN.rpn_train import iou_compute


# MTHead网络 loss 计算
def MTHead_Loss_Compute(bboxes, roi, cls_pred, loc_pred, iou_threshold, loss_fn = torch.nn.MSELoss()):
    # bboxes[m, 5] , roi[4], cls_pred [1,2], loc_pred [1,4]
    m, _ = bboxes.shape
    label = torch.ones_like(cls_pred)
    cls_loss, loc_loss = torch.tensor(0), torch.tensor(0)
    roi_pred = torch.tensor(roi) + loc_pred[0, ...]
    ious_m = []
    for i in range(m):
        ious_m.append(iou_compute(roi_pred, bboxes[i, 1:]))
    index_max_iou = ious_m.index(max(ious_m))
    if  index_max_iou > iou_threshold:
        label[0, bboxes[index_max_iou, 0]] = 1
        cls_loss += loss_fn(label, cls_pred)
        loc_loss += 1-ious_m[index_max_iou]
    return cls_loss, loc_loss

# 数据定义加载
BboxData_dataset = BBoxData('../../datasets/Mines.v2i.yolov5pytorch/train/images/',
                            '../../datasets/Mines.v2i.yolov5pytorch/train/labels/')

BboxData_loader = DataLoader(BboxData_dataset, 1)

# 已训练模型
frpn_net = FRPN()
frpn_net.load_state_dict(torch.load("../RPN/rpn_saved.pth"))

srcnn_net = SRCNN(num_channel=3)
srcnn_net.load_state_dict(torch.load('../SRCNN/srcnn_saved.pth'))

# 待训练
MTHead_net = MTHead()

# 误差梯度反向传播
optim = optim.SGD(MTHead_net.parameters(), lr=0.001, momentum=0.9)

cls_threshold = 0.2

MTHead_net.train()
for e in range(10):
    for img, bboxes in BboxData_loader:
        if bboxes.shape[1]:     # 如果有目标
            MTHead_net.zero_grad()

            _, _, _, rois = frpn_net(img)
            obj_rois = []
            for n in range(len(rois)):   #第n批次rois
                for roi in rois[n]:
                    roi_x0, roi_y0, roi_x1, roi_y1 = round(roi[1]), round(roi[0]), round(roi[3]), round(roi[2])
                    roi_patch = img[n, :, roi_x0:roi_x1, roi_y0:roi_y1]
                    roi_patch = roi_patch.reshape(1, roi_patch.shape[0], roi_patch.shape[1], roi_patch.shape[2])

                    # roi区域全部resize到固定大小
                    roi_patch = F.interpolate(roi_patch, size=(64, 64), mode='bicubic', align_corners=False)  # [1,3,64,64]

                    # roi 区域超分增强特征
                    roi_sr_patch = srcnn_net(roi_patch)

                    # roi 区域特征分类
                    cls_pred, loc_pred = MTHead_net(roi_sr_patch)

                    # 损失计算
                    cls_loss, loc_loss = MTHead_Loss_Compute(bboxes[n, ...], roi, cls_pred, loc_pred)
                    loss = cls_loss + loc_loss

                    loss.backward()
                    optim.step()







