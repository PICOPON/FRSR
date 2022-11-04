import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from rpn import FRPN


# 损失函数定义
def loss_compute(rpn_fg_scores, rpn_locs, anchors, bboxes):
    # anchors [9*H*W, 4],  rpn_scores (N,9*H*W,2)  rpn_locs (N, 9*H*W, 4)  bboxes (N, m, 5)
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
                ious[i] = iou_compute(anchors[i, ...], bboxes[n, j, 1:])    # 求出每个bbox和所有anchors的iou (9*H*W, 1)
            n_ious = sorted(ious.items(), key=lambda x: x[1], reverse=True)
            n_ious_max = [key for key, value in n_ious][:128]               # 取前128个iou对应的anchor序号
            gt_index_max128_ious.append(n_ious_max)

            # loc loss 计算
            for n_i in n_ious_max:
                rpn_loc_loss += (rpn_loc_loss_compute(rpn_locs[n, n_i, :],
                                                      anchors[n_i, :], bboxes[n, j, 1:]))**2

        gt_fg_scores = gt_fg_scores_generator(anchors, gt_index_max128_ious)        # 生成 [9*H*W, 1] 所有anchor的标签
        rpn_cls_loss += (rpn_cls_loss_compute(gt_fg_scores, rpn_fg_scores[n, ...]))**2  # cls_loss

    return rpn_cls_loss, rpn_loc_loss


# iou计算
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
    gt_fg_scores, rpn_fg_scores = gt_fg_scores.reshape(1, -1), rpn_fg_scores.reshape(1, -1)
    return loss_fn(gt_fg_scores, rpn_fg_scores)


def rpn_loc_loss_compute(rpn_locs, anchors, bbox):
    # (1, 4) rpn_locs tx ty tw th, anchors yxyx bbox yxyx
    """ rpn locs 损失计算 """
    xa, ya, wa, ha = anchors[1], anchors[0], anchors[3] - anchors[1], anchors[2] - anchors[0]
    tx, ty, tw, th = rpn_locs[0], rpn_locs[1], rpn_locs[2], rpn_locs[3]
    x_t, y_t, w_t, h_t = xa + wa * tx, ya + ha * ty, wa * torch.exp(tw), ha * torch.exp(th)
    iou = iou_compute((y_t, x_t, y_t + h_t, x_t + w_t), bbox)
    return iou


if __name__ == '__main__':
    # 数据集
    from datasets import BBoxData
    from torch.utils.data import DataLoader

    rpn_dataset = BBoxData(img_path="E:\\CVATData\\DJI_0030\\images",
                           label_path="E:\\CVATData\\DJI_0030\\labels")

    rpn_dataset_loader = DataLoader(rpn_dataset, 1)

    # 模型定义
    net = FRPN()

    # 冻结部分模型参数
    net.backbone.requires_grad = False

    # 误差梯度反向传播
    optim = optim.SGD(net.rpn.parameters(), lr=0.001, momentum=0.95)

    net.train()

    for e in range(3):
        for img, bboxes in rpn_dataset_loader:
            if bboxes.shape[1]:
                net.zero_grad()
                # 损失计算
                rpn_fg_scores, rpn_locs, anchors, rois = net(img)

                print(f'roi_nums: {len(rois[0])}')
                # 损失计算
                rpn_cls_loss, rpn_loc_loss = loss_compute(rpn_fg_scores, rpn_locs, torch.tensor(anchors), bboxes)
                #
                rpn_loss = rpn_cls_loss + rpn_loc_loss

                rpn_loss.backward()
                optim.step()

                print(f'epoch: {e}, rpn_loc_loss: {rpn_loc_loss}, rpn_cls_loss: {rpn_cls_loss} ')
                #
                # 可视化rpn网络推荐框的迭代过程
                plt.matshow(img[0, 0, ...])

                plt.gca().add_patch(plt.Rectangle((bboxes[0, 0, 2], bboxes[0, 0, 1]), bboxes[0, 0, 4] - bboxes[0, 0, 2],
                                                  bboxes[0, 0, 3] - bboxes[0, 0, 1], fill=False,
                                                  edgecolor='w', linewidth=3))
                for roi in rois[0]:
                    plt.gca().add_patch(plt.Rectangle((roi[1], roi[0]), roi[3] - roi[1],
                                                      roi[2] - roi[0], fill=False,
                                                      edgecolor='r', linewidth=3))
                plt.show()

    torch.save(net.state_dict(), 'rpn_saved.pth')
