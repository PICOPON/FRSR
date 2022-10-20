import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FRPN(nn.Module):
    def __init__(self):
        super(FRPN, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        for name, module in vgg16.named_modules():
            if name == 'features':
                self.backbone = module
        self.rpn = RPN(inChannels=512, im_info=32)

    def forward(self, x):
        features = self.backbone(x)
        rpn_score, rpn_locs, anchors, rois = self.rpn(features)
        return rpn_score, rpn_locs, anchors, rois


class RPN(nn.Module):
    def __init__(self, inChannels, im_info):
        super(RPN, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, inChannels, kernel_size=3, stride=1, padding=1)   # 3*3 conv 提取特征
        self.score = nn.Conv2d(inChannels, 18, kernel_size=1, stride=1)                      # 1*1 conv 分类
        self.loc = nn.Conv2d(inChannels, 36, kernel_size=1, stride=1)                        # 1*1 conv 回归
        self.feat_stride = im_info  # 用于记录下采样的放缩比例，生成anchor和rois

    def forward(self, x):
        h = F.relu(self.conv1(x))  # 输入H*W*512特征图 (N,512,H,W)
        n, _, hh, ww = h.shape  # 特征图的batch_size N, 高, 宽
        rpn_locs = self.loc(h).permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  # (N, 9*H*W, 4) 每个预测框的位置
        rpn_scores = self.score(h)  # 分类 (N,512,H,W) -> (N,2*9,H,W)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()  # (N,2*9,H,W)-> (N,H,W,2*9)
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, 9, 2), dim=4)
        # (N,H,W,2*9)->(N,H,W,9,2) 并在dim=4基础上去计算softmax
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()  # (N,H,W,9) 1对应是前景的置信度
        rpn_fg_scores = rpn_fg_scores.view(n, -1)  # (N,9*H*W) 每个预测框的置信度

        # rpn_scores = rpn_scores.view(n, -1, 2)  # (N,H,W,2*9) ->(N,9*H*W,2)
        # anchors生成
        anchors_base = self.generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[1, 2, 4])
        anchors = self._enumerate_shifted_anchor(anchors_base, feat_stride=self.feat_stride,
                                                 height=hh, width=ww)  # (9*H*W, 4)  原图的anchor坐标
        # anchors范围限制 clamp操作
        H, W = hh * self.feat_stride, ww * self.feat_stride
        anchors[anchors < 0] = 0
        anchors[:, 0][anchors[:, 0] > H] = H
        anchors[:, 1][anchors[:, 1] > W] = W
        anchors[:, 2][anchors[:, 2] > H] = H
        anchors[:, 3][anchors[:, 3] > W] = W

        # rois生成
        b_rois = []  # (N, 9*H*W, 4) 取1个anchor box
        for i in range(n):
            rois = self.proposal_layer(
                anchors,
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                (hh, ww),
                self.feat_stride
            )
            b_rois.append(rois)

        return rpn_fg_scores, rpn_locs, anchors, b_rois

    # 函数功能：
    # 以 base_size=(16*16)为基础anchor，按照ratios和scale 生成基本的9个anchor
    def generate_anchor_base(self, base_size=16, ratios=[0.5, 1, 2],
                             anchor_scales=[8, 16, 32]):
        '''
        :param base_size:  anchor基础大小
        :param ratios:  anchor形状长宽比
        :param anchor_scales: anchor尺寸放缩比率
        :return: yxyx
        '''

        py = base_size / 2.  # 8.0 base box的中心y坐标
        px = base_size / 2.  # 8.0 base box的中心x坐标

        # 生成shape为(9,4)，值为0的二维数据
        anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                               dtype=np.float32)  # shape is (9,4)
        # 通过计算法则
        # 同一种scale的anchor，面积相等，三个anchor的w:h分别是1:1,1:2,2:1
        # 同一种ratio的anchor，形状相同，三个anchor的面积分别是128^2,256^2,512^2
        # 进而可以计算得到以base box的中心为中心的9种不同的anchor
        for i in range(len(ratios)):
            for j in range(len(anchor_scales)):
                h = base_size * anchor_scales[j] * np.sqrt(ratios[i])  # 16 * scale_8 * sqrt(0.5)
                w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

                index = i * len(anchor_scales) + j
                anchor_base[index, 0] = py - h / 2.  # 左上角y
                anchor_base[index, 1] = px - w / 2.  # 左上角x
                anchor_base[index, 2] = py + h / 2.  # 右下角y
                anchor_base[index, 3] = px + w / 2.  # 右下角x
        # anchor_bash.shape = (9,4)
        return anchor_base

    # 函数功能:
    # 将anchor_base也就是特征图中每个位置上9个anchors的坐标与特征图中每个位置的坐标进行结合
    # 进而生成特征图中每个点都有其对应的9个anchors
    def _enumerate_shifted_anchor(self, anchor_base, feat_stride, height, width):
        """
        参数说明:
             anchor_base:
                 生成的9个尺寸各异的anchors的坐标 shape = (9,4)
             feat_stride:
                 原始图像尺寸经过backbone之后得到的feature map 与原始图像之间的比例
             height:
                 feature map's height
             width:
                 feature map's width
        """
        # 以0为起点 ，height * feat_stride为终点   feat_stride为步长 得到一个np.array
        # 如array [0,16,32,..]
        # 这里对height和width采用了相同的步长，在torchvision中是区别对待了的
        shift_y = np.arange(0, height * feat_stride, feat_stride)
        shift_x = np.arange(0, width * feat_stride, feat_stride)

        # 这里的代码与torchvision的类似，torch是用其内置方法，这里使用的np的方法
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        # 其中np.ravel()方法的效果就是生成一个连续的一维序列，其效果等价与reshape(-1)
        # 得到shift.shape = (height*width,4)
        # 得到一个从原图(注意是原图)左上角第一个位置(0,0) -> 原图右下角((height-1)*feat_stride,(width-1)*feat_stride)
        # 的最后一个位置 得到一个基础坐标
        shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                          shift_y.ravel(), shift_x.ravel()), axis=1)

        # anchor_base.shape = (9,4)
        A = anchor_base.shape[0]  # A=9
        K = shift.shape[0]  # K=height*width

        # shift得到的是特征图的每个点在原始图片中的位置坐标，而anchor_base是特征图中每个点生成9个尺寸各异的anchor的坐标
        # 将两者按广播机制进行相加,就完成了特征图中以每个点为中心生成9个anchor的操作，所以最终的anchor.shape = (height*width*9,4)
        anchor = anchor_base.reshape((1, A, 4)) + \
                 shift.reshape((1, K, 4)).transpose((1, 0, 2))
        # anchor.shape = (k*A,4)=(height*width*9,4)
        anchor = anchor.reshape((K * A, 4)).astype(np.float32)
        return anchor

    # 函数功能:
    # 计算筛选生成的候选框
    def proposal_layer(self, anchors, rpn_locs, rpn_fg_scores, fmap_size, scale):
        '''
        :param anchors:   锚框 (9*H*W, 4)
        :param rpn_locs:  锚框位置偏移
        :param rpn_fg_scores:  锚框类置信度 (9*H*W, )
        :return:  roi 生成1个候选框 yxyx
        '''
        t_anchors = np.zeros_like(anchors)
        px, py, pw, ph = anchors[:, 1], anchors[:, 0], anchors[:, 3] - anchors[:, 1], anchors[:, 2] - anchors[:, 0]
        dx, dy, dw, dh = rpn_locs[:, 0], rpn_locs[:, 1], rpn_locs[:, 2], rpn_locs[:, 3]
        gx, gy, gw, gh = px + dx * pw, py + dy * ph, pw * np.exp(dw), ph * np.exp(dh)
        t_anchors[:, 0], t_anchors[:, 1], t_anchors[:, 2], t_anchors[:, 3] = gy, gx, gy + gh, gx + gw
        # 分数筛选
        # clamp
        H, W = fmap_size[0] * scale, fmap_size[1] * scale
        t_anchors[t_anchors < 0] = 0
        t_anchors[:, 0][t_anchors[:, 0] > H] = H
        t_anchors[:, 1][t_anchors[:, 1] > W] = W
        t_anchors[:, 2][t_anchors[:, 2] > H] = H
        t_anchors[:, 3][t_anchors[:, 3] > W] = W

        dict_t_fg_score = dict()
        for id, t_fg_score in enumerate(rpn_fg_scores):
            dict_t_fg_score[id] = t_fg_score

        max_t_fg_scores = sorted(dict_t_fg_score.items(), key=lambda x: x[1], reverse=True)
        max_index_anchors = [key for key, value in max_t_fg_scores][:10]  # 取前128个iou对应的anchor序号
        rois = []
        for m_i in max_index_anchors:
            rois.append(t_anchors[m_i, :])
        # nms
        return rois


# 损失函数定义
def loss_compute(rpn_fg_scores, rpn_locs, anchors, bboxes):
    # anchors [9*H*W, 4],  rpn_scores (N,9*H*W,2)  rpn_locs (N, 9*H*W, 4) bboxes (N, m, 4)
    # anchors 筛选掉越界的
    N, m, _ = bboxes.shape
    rpn_cls_loss = 0
    rpn_loc_loss = 0
    for n in range(N):
        # cls loss 计算
        gt_index_max128_ious = []  # (m, 128)  iou最大对应的anchor序号
        for j in range(m):  # m个bbox
            ious = dict()
            for i in range(len(anchors)):
                ious[i] = iou_compute(anchors[i, ...], bboxes[n, j, :])  # 求出每个bbox和所有anchors的iou (9*H*W, 1)
            n_ious = sorted(ious.items(), key=lambda x: x[1], reverse=True)
            n_ious_max = [key for key, value in n_ious][:128]     # 取前128个iou对应的anchor序号
            gt_index_max128_ious.append(n_ious_max)

            # loc loss 计算
            for n_i in n_ious_max:
                rpn_loc_loss += rpn_loc_loss_compute(rpn_locs[n, n_i, :],
                                                     anchors[n_i, :], bboxes[n, j, :])

        gt_fg_scores = gt_fg_scores_generator(anchors, gt_index_max128_ious)  # 生成 [9*H*W, 1] 所有anchor的标签
        rpn_cls_loss += rpn_cls_loss_compute(gt_fg_scores, rpn_fg_scores[n, ...])  # cls_loss

        '''
        # loc loss 计算
        for gt_i in gt_index_max_ious:
            # 对每个gt对应的前景anchor锚框求loc损失
            rpn_bbox_loss = []
            for l in range(m):
                rpn_bbox_loss.append(rpn_loc_loss_compute(rpn_locs[n, gt_i, :],
                                                          anchors[gt_i, :], bboxes[n, l, :]).detach())
            #
            rpn_loc_loss += rpn_loc_loss_compute(rpn_locs[n, gt_i, :],
                                                 anchors[gt_i, :], bboxes[n, np.argmax(rpn_bbox_loss), :])
        '''

    return rpn_cls_loss, rpn_loc_loss


def iou_compute(anchor, bbox):
    # (1, 4) anchor, bbox : yxyx
    box1, box2 = anchor, bbox
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = inter / union
    return iou


def gt_fg_scores_generator(anchors, gt_index_max_ious):
    # [9*H*W, 4], (m, 1)
    gt_rpn_label = torch.zeros(anchors.shape[0])  #
    for gt_index128 in gt_index_max_ious:
        for gt_i in gt_index128:
            gt_rpn_label[gt_i] = 1  # 前景为1 背景为0
    return gt_rpn_label


def rpn_cls_loss_compute(gt_fg_scores, rpn_fg_scores, loss_fn=nn.L1Loss()):
    # 目标 gt_fg_scores =[0,0,...,1, ..., 0] <= rpn_fg_scores
    return loss_fn(gt_fg_scores, rpn_fg_scores)


def rpn_loc_loss_compute(rpn_locs, anchors, bbox):
    # (1, 4) rpn_locs tx ty tw th, anchors yxyx bbox yxyx
    """ rpn locs 损失计算
    """
    xa, ya, wa, ha = anchors[1], anchors[0], anchors[3] - anchors[1], anchors[2] - anchors[0]
    tx, ty, tw, th = rpn_locs[0], rpn_locs[1], rpn_locs[2], rpn_locs[3]
    x_t, y_t, w_t, h_t = xa + wa * tx, ya + ha * ty, wa * math.exp(tw), ha * math.exp(th)
    iou = iou_compute((y_t, x_t, y_t + h_t, x_t + w_t), bbox)
    return iou


# 数据集
from datasets import CoCo
from torch.utils.data import DataLoader

coco_dataset = CoCo('../datasets/Mines.v2i.yolov5pytorch/train/images/',
                    '../datasets/Mines.v2i.yolov5pytorch/train/labels/')

coco_loader = DataLoader(coco_dataset, 1)

# img = torch.randn(1, 3, 640, 640)
# bboxes = torch.tensor([[[100, 200, 30, 30], [100, 100, 100, 100], [200, 200, 100, 100]]])

# 模型定义
net = FRPN()
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

            print(f'roi_nums: {len(rois[0])}')
            # 损失计算
            rpn_cls_loss, rpn_loc_loss = loss_compute(rpn_fg_scores, rpn_locs, torch.tensor(anchors), bboxes)
            #
            print(f'rpn_loc_loss: {rpn_loc_loss}, rpn_cls_loss: {rpn_cls_loss} ')
            loss_fn = nn.MSELoss()
            rpn_loss = loss_fn(rpn_cls_loss, rpn_loc_loss)

            rpn_loss.backward()
            optim.step()
            print(f'rpn_loc_loss: {rpn_loc_loss}, rpn_cls_loss: {rpn_cls_loss} ')

            #
            plt.matshow(img[0, 0, ...])

            plt.gca().add_patch(plt.Rectangle((bboxes[0, 0, 1], bboxes[0, 0, 0]), bboxes[0, 0, 3] - bboxes[0, 0, 1],
                                              bboxes[0, 0, 2] - bboxes[0, 0, 0], fill=False,
                                              edgecolor='w', linewidth=3))
            for roi in rois[0]:
                plt.gca().add_patch(plt.Rectangle((roi[1], roi[0]), roi[3] - roi[1],
                                                  roi[2] - roi[0], fill=False,
                                                  edgecolor='r', linewidth=3))
            plt.show()
