from datasets import BBoxData
from torch.utils.data import DataLoader
from frsr import FRSR
import matplotlib.pyplot as plt


# 数据定义加载
BboxData_dataset = BBoxData('../../datasets/Mines.v2i.yolov5pytorch/train/images/',
                            '../../datasets/Mines.v2i.yolov5pytorch/train/labels/')

BboxData_loader = DataLoader(BboxData_dataset, 1)

# 模型定义
net = FRSR()

print(net.state_dict())

'''
# 冻结部分模型参数
net.rpn_front.requires_grad = False
net.sr_cnn.requires_grad = False

# 误差梯度反向传播
# optim = optim.SGD(net.dt_end.parameters(), lr=0.001, momentum=0.9)

net.train()
for e in range(10):
    for img, bboxes in BboxData_loader:
        if bboxes.shape[1]:
            net.zero_grad()
            # 损失计算
            y = net(img)
            plt.matshow(y[0][0, 0, :, :].detach().numpy())   # 第一个roi推荐区域的显示
            plt.show()
            break
        break
    break
'''