
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datasets import MyData


# model
class Generator(nn.Module):
    def __init__(self, z):
        super(Generator, self).__init__()
        self.Z = z
        self.LN1 = nn.Linear(100, 1024*4*4)
        self.DECONV = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, (4, 4), stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self):
        Z_conv = torch.reshape(self.LN1(self.Z), (-1, 1024, 4, 4))
        out = self.DECONV(Z_conv)
        return out

    def updateZ(self, z):
        self.Z = z


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.CONV = nn.Sequential(
            nn.Conv2d(3, 128, (4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.LN = nn.Sequential(
            nn.Linear(512*8*8, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.CONV(input)
        input = input.view((-1, 512*8*8))
        return self.LN(input)


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('GPU', torch.cuda.get_device_name())
    data = DataLoader(MyData(img_path="data/bird/images"), batch_size=50)
    G = Generator(z=torch.rand((50, 100)).cuda()).cuda()
    D = Discriminator().cuda()

    # 权重初始化
    # G.apply(weight_init)
    # D.apply(weight_init)
    #
    for epoch in range(20):
        D.train()
        G.train()
        for e in range(20):
            for X in data:
                G.updateZ(z=torch.rand((50, 100)).cuda())
                # D training
                optim_D = optim.SGD(D.parameters(), lr=0.01, momentum=0.9)
                optim_D.zero_grad()

                loss_D = torch.mean(-torch.log(1-D(G())) - torch.log(D(X.cuda())))
                loss_D.backward()
                optim_D.step()
                # print('loss_D:%.5f  ' % loss_D)

        for e in range(200):
            G.updateZ(z=torch.rand((50, 100)).cuda())
            # G training
            optim_G = optim.SGD(G.parameters(), lr=0.01, momentum=0.9)
            optim_G.zero_grad()

            loss_G = torch.mean(-torch.log(D(G())))
            loss_G.backward()
            optim_G.step()
            # print('loss_G: %.5f ' % loss_G)

        print('epoch:%d---------------------------------------------'%epoch)

    torch.save(G, 'Gen.pth')

    plt.matshow(G().detach().cpu().numpy()[0, 0, :, :])
    plt.show()