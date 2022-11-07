import torch.nn
import torch.optim as optim
from datasets import SRData
from torch.utils.data import DataLoader
from srcnn import SRCNN

# 计算硬件
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# 超参数
batch_size = 16
lr = 0.01
epochs = 200
# 数据
train = SRData(fpath='train.h5')
test = SRData(fpath='test.h5')
trainSet = DataLoader(train, batch_size=batch_size)
testSet = DataLoader(test, batch_size=batch_size)

# 模型
net = SRCNN(num_channel=3).cuda()

# 模型参数初始化
# net.apply(weight_init)

# 损失优化函数
loss_fn = torch.nn.MSELoss()
optim = optim.SGD([
    {'params': net.CONV1.parameters(), 'lr': 0.01},
    {'params': net.CONV2.parameters(), 'lr': 0.01},
    {'params': net.CONV3.parameters(), 'lr': 0.01},
], lr=0.01, weight_decay=0.001)

# 训练

for e in range(epochs):
    net.train()
    for batch, (x, y) in enumerate(trainSet):
        x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
        optim.zero_grad()
        pred = net(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optim.step()
        if batch % 10 == 0:
            print(f'epoch: {e}  train loss: {loss}')
    net.eval()
    with torch.no_grad():
        avg_loss = 0
        for x, y in testSet:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
            pred = net(x)
            avg_loss += loss_fn(pred, y)
        print(f'avg test loss: { avg_loss / len(testSet) }')
    # torch.onnx.export(net, torch.randn(1, 3, 256, 256).to(device), 'srcnn.onnx', input_names=['input'],
    #                  output_names=['output'])

torch.save(net.state_dict(), 'srcnn_saved.pth')