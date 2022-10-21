import glob
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


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


class CoCo(Dataset):
    def __init__(self, img_path, label_path=None, trans=None):
        super(CoCo, self).__init__()
        self.imgs_path = glob.glob('{}/*'.format(img_path))
        self.labels_path = glob.glob('{}/*'.format(label_path))
        self.trans = trans
        # 可用于图片的裁剪

    def __getitem__(self, item):
        img = cv2.imread(self.imgs_path[item], cv2.IMREAD_COLOR)
        h, w, _ = img.shape
        labels = []
        with open(self.labels_path[item], 'r') as f:
            for line in f:
                x_, y_, w_, h_ = [float(x) for x in line.strip('\n').split(' ')[1::]]
                x1 = w * x_ - 0.5 * w * w_
                x2 = w * x_ + 0.5 * w * w_
                y1 = h * y_ - 0.5 * h * h_
                y2 = h * y_ + 0.5 * h * h_
                labels.append([y1, x1, y2, x2])

        return torch.tensor(img/255., dtype=torch.float32).permute(2, 0, 1), torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.imgs_path)


