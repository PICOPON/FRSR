import h5py
from torch.utils.data import Dataset
import numpy as np


class MyData(Dataset):
    def __init__(self, fpath='data.h5', trans=None):
        super(MyData, self).__init__()
        self.f = h5py.File(fpath, mode='r')
        self.hrs = self.f["hr"]
        self.lrs = self.f["lr"]
        self.trans = trans

    def __getitem__(self, index):
        return np.transpose(self.lrs[index, ...]/255., (2, 0, 1)), np.transpose(self.hrs[index, ...]/255., (2, 0, 1))

    def __len__(self):
        return self.lrs.shape[0]

# dataset
'''
class MyData(Dataset):
    def __init__(self, img_path, label_path=None, trans=None, scale=2):
        super(MyData, self).__init__()
        self.scale = scale
        self.imgs_path = glob.glob('{}/*'.format(img_path))
        # self.labels_path = os.path.join(label_path, "*.txt")
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(100)
        ])
        # 可用于图片的裁剪

    def __getitem__(self, item):
        img = cv2.imread(self.imgs_path[item], cv2.IMREAD_COLOR)
        hr_size = (img.shape[0] // self.scale) * self.scale, (img.shape[1] // self.scale) * self.scale
        hr = cv2.resize(img, hr_size, interpolation=cv2.INTER_CUBIC)
        lr = cv2.resize(img, (img.shape[0] // self.scale, img.shape[1] // self.scale), interpolation=cv2.INTER_CUBIC)
        lr = cv2.resize(lr, hr_size, interpolation=cv2.INTER_CUBIC)
        return self.trans(lr),  self.trans(hr)

    def __len__(self):
        return len(self.imgs_path)
'''