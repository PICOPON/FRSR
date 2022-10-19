import torch.nn as nn
import numpy as np


class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head,
                loc_normalize_mean = (0., 0., 0., 0.),
                loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    ):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor # 特征提取网络
        self.rpn = rpn # rpn网络
        self.head = head # roi 分类回归

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

    def forward(self, x, scale=1.):
        img_size = x.shape[2:] # 原始图片的H，W

        h = self.extractor(x) # 经过特征提取网络之后得到feature map
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn(h, img_size, scale) # rpn网络处理
        roi_cls_locs, roi_scores = self.head(
            h, rois, roi_indices) # head网络处理

        return roi_cls_locs, roi_scores, rois, roi_indices

    # use_preset方法主要是根据不同的情况设置一些阈值，如nms阈值 和最后的score阈值等
    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')


class RegionProposalNetwork(nn.Module):
    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        # anchor_base.shape = (9,4)
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0] #9
        # 这三个Conv2d 都没有改变tensor的channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1) # rpn网络的第一个3*3 卷积
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0) # 分类部分 (N,512,H,W) -> (N,2*9,H,W)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)  #回归部分   (N,512,H,W) -> (N,4*9,H,W)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        # 这里的x就是输入的features特征图
        n, _, hh, ww = x.shape # 得到特征图的N,C,H,W
        # 关于anchor的生成 见后文
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)
        # anchor.shape = (height*width*9,4) 这里的height和width是特征图的height和width
        n_anchor = anchor.shape[0] // (hh * ww) # 9
        h = F.relu(self.conv1(x)) # rpn网络的第一个3*3 卷积

        rpn_locs = self.loc(h) # 回归 (N,512,H,W) -> (N,4*9,H,W)
        # UNNOTE: check whether need contiguous
        # A: Yes
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  # (N,4*9,H,W) -> (N,9*H*W,4)

        rpn_scores = self.score(h)  # 分类 (N,512,H,W) -> (N,2*9,H,W)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous() # (N,2*9,H,W)-> (N,H,W,2*9)
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        # (N,H,W,2*9)->(N,H,W,9,2) 并在dim=4基础上去计算softmax
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous() # (N,H,W,9)
        rpn_fg_scores = rpn_fg_scores.view(n, -1) # (N,9*H*W)
        rpn_scores = rpn_scores.view(n, -1, 2) # (N,H,W,2*9) ->(N,9*H*W,2)

        rois = list()
        roi_indices = list()
        for i in range(n):
            # ProposalCreator 根据得到的anchors和对应的分类回归信息
            # 做一些处理生成一个基础的roi
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        # 得到五个数据
        # rpn 的回归数据rpn_locs,(N, H*W*A, 4)
        # rpn的分类数据 rpn_scores  (N, H*W*A, 2)
        # N个矩阵的roi数据 组成的rois (N*R,4)
        # roi_indices 对应着roi的编号以及长度
        # anchor (H*W*A,4)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


# 函数功能：
# 以 base_szie=(16*16)为基础anchor，按照ratios和scale 生成基本的9个anchor
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):

    py = base_size / 2.  #8.0 base box的中心y坐标
    px = base_size / 2.  #8.0 base box的中心x坐标

    # 生成shape为(9,4)，值为0的二维数据
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)  #  shape is (9,4)
    # 通过计算法则
    # 同一种scale的anchor，面积相等，三个anchor的w:h分别是1:1,1:2,2:1
    # 同一种ratio的anchor，形状相同，三个anchor的面积分别是128^2,256^2,512^2
    # 进而可以计算得到以base box的中心为中心的9种不同的anchor
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i]) # 16 * scale_8 * sqrt(0.5)
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2. # 左上角y
            anchor_base[index, 1] = px - w / 2. # 左上角x
            anchor_base[index, 2] = py + h / 2. # 右下角y
            anchor_base[index, 3] = px + w / 2. # 右下角x
    # anchor_bash.shape = (9,4)
    return anchor_base

"""
函数功能:
     将anchor_base也就是特征图中每个位置上9个anchors的坐标与特征图中每个位置的坐标进行结合
     进而生成特征图中每个点都有其对应的9个anchors
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

def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    import numpy as np
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