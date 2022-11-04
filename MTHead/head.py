import torch.nn as nn
import torch.nn.functional as F
import torchvision


class MTHead(nn.Module):
    def __init__(self):
        super(MTHead, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.fc_cls = nn.Linear(1000, 2)    # 2个类别，类别概率
        self.fc_loc = nn.Linear(1000, 4)    # 用于微调预测框输出

    def forward(self, x):
        x = self.resnet50(x)
        cls = F.softmax(self.fc_cls(x), dim=1)
        loc = self.fc_loc(x)
        return cls, loc



