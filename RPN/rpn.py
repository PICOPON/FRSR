import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class FRPN(nn.Module):
    def __init__(self):
        super(FRPN, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)   # 加载预训练权重
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
        self.feat_stride = im_info  # 采样的放缩比例，用于在输入图中生成anchor和rois

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
        # anchors生成  未范围限制的anchors
        anchors_base = self.generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[1, 2, 4])
        anchors = self.shifted_anchor_base(anchors_base, feat_stride=self.feat_stride,
                                                 height=hh, width=ww)  # (9*H*W, 4)  原图的anchor坐标

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

    # 以 base_size=(16*16)为基础anchor，按照ratios和 scale 生成基本的9个anchor
    def generate_anchor_base(self, base_size=16, ratios=[0.5, 1, 2],
                             anchor_scales=[8, 16, 32]):
        '''
        :param base_size:  anchor基础大小
        :param ratios:  anchor形状长宽比
        :param anchor_scales: anchor尺寸放缩比率
        :return: 生成anchors 坐标值：yxyx
        '''

        py = base_size / 2.  # 8.0 base box的中心y坐标
        px = base_size / 2.  # 8.0 base box的中心x坐标

        # 生成shape为(9,4)，值为0的二维数据
        anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                               dtype=np.float32)

        # 同一种scale的anchor，面积相等，三个anchor的w:h分别是1:1,1:2,2:1
        # 同一种ratio的anchor，形状相同，三个anchor的面积分别是128^2,256^2,512^2
        for i in range(len(ratios)):
            for j in range(len(anchor_scales)):
                h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
                w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

                index = i * len(anchor_scales) + j
                anchor_base[index, 0] = py - h / 2.  # 左上角y
                anchor_base[index, 1] = px - w / 2.  # 左上角x
                anchor_base[index, 2] = py + h / 2.  # 右下角y
                anchor_base[index, 3] = px + w / 2.  # 右下角x

        return anchor_base  # shape (9,4)

    # 将anchor_base也就是特征图中每个位置上9个anchors的坐标与特征图中每个位置的坐标进行结合
    # 进而生成特征图中每个点都有其对应的9个anchors
    def shifted_anchor_base(self, anchor_base, feat_stride, height, width):
        """
        参数说明:
             anchor_base:
                 生成的9个尺寸各异的anchors的坐标 shape = (9,4)
             feat_stride:
                 原始图像尺寸经过backbone之后得到的feature map 与原始图像之间的比例　org/ fmap
             height:
                 feature map's height
             width:
                 feature map's width
        """
        # 以0为起点 ，height * feat_stride为终点   feat_stride为步长 得到np.array
        shift_y = np.arange(0, height * feat_stride, feat_stride)
        shift_x = np.arange(0, width * feat_stride, feat_stride)

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

    # 计算筛选生成的候选框
    def proposal_layer(self, anchors, rpn_locs, rpn_fg_scores, fmap_size, scale):
        '''
        :param anchors:   锚框 (9*H*W, 4)  yxyx
        :param rpn_locs:  锚框位置偏移
        :param rpn_fg_scores:  锚框类置信度 (9*H*W, 1)
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
        max_index_anchors = [key for key, value in max_t_fg_scores][:10]  # 取前10个iou对应的anchor序号
        rois = []
        for m_i in max_index_anchors:
            rois.append(t_anchors[m_i, :])
        # nms
        return rois