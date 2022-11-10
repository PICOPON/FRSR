import torch
from torch.utils.data import DataLoader
import torchvision
from datasets import BBoxData

# 计算硬件
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')


if __name__ == '__main__':

    # 数据加载
    BboxData_dataset = BBoxData('E:\\dataset\\DJI_0030\\DJI_0030\\images',
                                'E:\\dataset\\DJI_0030\\DJI_0030\\labels')

    BboxData_loader = DataLoader(BboxData_dataset, 1)
    for img,  bboxes in BboxData_loader:
        vgg16 = torchvision.models.vgg16(pretrained=True)  # 加载预训练权重
        print(vgg16(img))
        break