import collections
import torch.nn as nn

# 四种loss
LossTuple = collections.namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):
    """wrapper for conveniently training. return losses

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn # 首先由一个fasterrcnn的实例来初始化
        self.rpn_sigma = opt.rpn_sigma #3
        self.roi_sigma = opt.roi_sigma #1

        # target creator create gt_bbox gt_label etc as training targets.
        self.anchor_target_creator = AnchorTargetCreator() # 创建anchor Target
        self.proposal_target_creator = ProposalTargetCreator() # 创建Proposal Target

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean # 归一化的均值
        self.loc_normalize_std = faster_rcnn.loc_normalize_std # 归一化的方差

        self.optimizer = self.faster_rcnn.get_optimizer() # 优化器 在faster_rcnn这个类中有具体的实现
        # 以下就是有关可视化的实现，笔者在自己复现时并没有使用这些操作
        # 所以这里就不过多说明，大家有兴趣可以自行尝试
        # visdom wrapper
        self.vis = Visualizer(env=opt.env)
        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        n = bboxes.shape[0] # N
        # 作者的代码只针对batchsize =1 做了实现，对于多batchsize需要对源码做一些修改
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape #图像的尺寸
        img_size = (H, W)

        # 得到特征图
        features = self.faster_rcnn.extractor(imgs)

        # rpn 返回五个数据
        # rpn 的回归数据rpn_locs ,shape=(N, features_H*features_W*A, 4)
        # rpn的分类数据 rpn_scores , shape=(N, features_H*features_W*A, 2)
        # roi数据 ,shape=(N*R,4)
        # roi数据对应的索引, shape=(N*R,)
        # 所有的anchor ,shape=(A*features_H*features_W,4)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)
        """
        这里代码说明一下，主要是batchsize=1 注意这一点
        那么 bboxses.shape = (1,R,4)
             label.shape = (1,R)
             rpn_score.shape = (1,features_H*features_W*A,2)
             roi.shape = (R,4)
             roi_indices.shape = (R)
        """
        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0] # shape = (R',4)
        label = labels[0] # shape = (R',)
        rpn_score = rpn_scores[0] # (features_H*features_W*A,2)
        rpn_loc = rpn_locs[0] # (features_H*features_W*A,4)
        roi = rois #(R,4)

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois,
        # consider them as constant input
        # 由rpn得到的roi ，以及由dataset得到的真值box，label传入proposal_target_creator
        # proposal_target_creator做了一个这样的事
        # 通过给定的bbox label 以及roi得到满足阈值的  roi(sample_roi) roc_loc的偏移量(gt_roi_loc) 以及roi的label(gt_roi_label)
        # 关于proposal_target_creator的具体实现，参考后续对ProposalTargetCreator的说明
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)

        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        # head 计算得到最终roi的分类和回归
        # head的操作参考后续对head的说明
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # 计算loss
        # ------------------ RPN losses -------------------#
        # 同样将由dataset得到的box，以及rpn网络模块中生成的anchor传入anchor_target_creator
        # 关于anchor_target_creator的具体操作实现，见后续对AnchorTargetCreator的说明
        # gt_rpn_loc 为rpn真值偏移量
        # gt_rpn_label 为rpn真值label
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        # 第一个loss rpn回归loss rpn_loc_loss
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        # 第二个loss rpn分类loss rpn_cls_loss
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        # 第三个loss roi回归loss roi_loc_loss
        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        # 第四个loss roi分类loss roi_cls_loss
        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        # 返回的loss包括前面四个loss 和所有的loss的sum totalloss
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    # 训练step，作者这里训练的相关操作放到了这里，包括loss的back 优化器的优化执行等
    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses
   # 这里还需要说明一下:
   # 除了上述的几个方法以外，FasterRCNNTrainer还封装了几个方法包括save方法用来保存模型权重、load方法用来load模型参数、和其他几个有关可视化的方法
   # 与训练的核心无关，所以为了简化文章这里不赘述
   # 而关于模型的predict方法详见 4 预测&验证流程