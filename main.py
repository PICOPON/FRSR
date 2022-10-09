import torch.nn
import torch.optim as optim
from datasets import MyData
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from SRCNN import SRCNN

# 计算硬件
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')
# 超参数
batch_size = 16
lr = 0.01
epochs = 50
# 数据
train = MyData(fpath='train.h5')
test = MyData(fpath='test.h5')
trainSet = DataLoader(train, batch_size=batch_size)
testSet = DataLoader(test, batch_size=batch_size)

# 模型
net = SRCNN(num_channel=3).cuda()

# 模型参数初始化
# net.apply(weight_init)

# 损失优化函数
loss_fn = torch.nn.MSELoss()
optim = optim.SGD([
    {'params': net.CONV1.parameters()},
    {'params': net.CONV2.parameters(), 'lr': 0.005},
    {'params': net.CONV3.parameters(), 'lr': 0.005},
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
        if batch % 20 == 0:
            print(f'epoch: {e}  train loss: {loss}')
    net.eval()
    with torch.no_grad():
        loss = 0
        for x, y in testSet:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
            pred = net(x)
            loss += loss_fn(pred, y)
        print(f'avg test loss: {loss / len(testSet)}')
    # torch.onnx.export(net, torch.randn(1, 3, 256, 256).to(device), 'srcnn.onnx', input_names=['input'], output_names=['output'])
    torch.save(net, 'srcnn.pth')


'''
image = cv2.imread('data/Set5/bird.png')
small = cv2.resize(image, (int(image.shape[0]*0.5), int(image.shape[1]*0.5)), interpolation=cv2.INTER_CUBIC)
big = cv2.resize(small, (int(small.shape[0]*2), int(small.shape[1]*2)), interpolation=cv2.INTER_CUBIC)

print(image.shape)

mse = np.mean(np.square(image - big))
psnr = 10*np.log10(255*255/mse)
print(psnr)

cv2.imshow('org', image)
cv2.imshow('sma', small)
cv2.waitKey(0)
'''