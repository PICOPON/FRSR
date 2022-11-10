import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from MTHead.head import MTHead
from RPN.rpn import FRPN
from SRCNN.srcnn import SRCNN

from datasets import BBoxData

# 计算硬件
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')


if __name__ == '__main__':

    # 数据加载
    BboxData_dataset = BBoxData('E:\\dataset\\DJI_0030\\DJI_0030\\images',
                                'E:\\dataset\\DJI_0030\\DJI_0030\\labels')

    BboxData_loader = DataLoader(BboxData_dataset, 1)


    for e in range(1):
        for img, bboxes in BboxData_loader:
            img, bboxes = img.to(device, dtype=torch.float32), bboxes.to(device, dtype=torch.float32)
            # 已训练模型
            frpn_net = FRPN().cuda(device=device)
            frpn_net.load_state_dict(torch.load("./RPN/rpn_saved.pth"))

            srcnn_net = SRCNN(num_channel=3).cuda(device=device)
            srcnn_net.load_state_dict(torch.load('./SRCNN/srcnn_saved.pth'))

            MTHead_net = MTHead().cuda(device=device)
            MTHead_net.load_state_dict(torch.load("./MTHead/head_saved.pth"))

            if bboxes.shape[1]:  # 如果有目标
                MTHead_net.zero_grad()
                _, _, _, rois = frpn_net(img)
                obj_rois = []
                for n in range(len(rois)):
                    cls_n_loss, loc_n_loss = 0, 0
                    plt.imshow(img[n, ...].permute(1, 2, 0).cpu())  # 绘制第n张图
                    obj_rois = []  # x, y, w, h
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

                            if any(cls_pred[0, ...]):
                                loc_pred = loc_pred.detach()
                                xa, ya, wa, ha = roi[1], roi[0], roi[3] - roi[1], roi[2] - roi[0]
                                tx, ty, tw, th = loc_pred[0, 0], loc_pred[0, 1], loc_pred[0, 2], loc_pred[0, 3]
                                x_t, y_t, w_t, h_t = xa + wa * tx, ya + ha * ty, wa * torch.exp(tw), ha * torch.exp(th)
                                obj_rois.append((cls_pred[0, ...].detach(), x_t, y_t, w_t, h_t))

                    # 类别阈值处理
                    dnst_cls_1_obj_rois = sorted(obj_rois, key=lambda x: x[0][0], reverse=True)
                    dnst_cls_2_obj_rois = sorted(obj_rois, key=lambda x: x[0][1], reverse=True)

                    # 对 obj_rois 进行 nms 处理 筛选iou较大的重复框
                    # obj_rois = nms(obj_rois)

                    # 置信度阈值0.5 显示最终检测目标
                    out_obj_box = dnst_cls_1_obj_rois[0]
                    if out_obj_box[0][0] > 0.5:
                        plt.gca().add_patch(
                            plt.Rectangle(xy=(out_obj_box[1].cpu(), out_obj_box[2].cpu()), width=out_obj_box[3].cpu(),
                                          height=out_obj_box[4].cpu(),
                                          edgecolor='r',
                                          fill=False, linewidth=2
                                          ))
                        plt.show()
