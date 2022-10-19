import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self, num_channel=1):
        super(SRCNN, self).__init__()
        self.CONV1 = nn.Conv2d(num_channel, 64, (9, 9), padding=9 // 2)
        self.CONV2 = nn.Conv2d(64, 32, (5, 5), padding=5 // 2)
        self.CONV3 = nn.Conv2d(32, num_channel, (5, 5), padding=5 // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.CONV1(x))
        x = self.relu(self.CONV2(x))
        return self.CONV3(x)


