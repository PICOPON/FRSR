import glob
import h5py
import cv2


img_path = 'data/Set5'
f = h5py.File('test.h5', mode='w')
scale = 2

lr_patches = []
hr_patches = []
patch_size = 200
stride = 10

for item_path in glob.glob('{}/*'.format(img_path)):
    img = cv2.imread(item_path, cv2.IMREAD_COLOR)
    hr_size = (img.shape[0] // scale) * scale, (img.shape[1] // scale) * scale
    hr = cv2.resize(img, hr_size, interpolation=cv2.INTER_CUBIC)
    lr = cv2.resize(img, (img.shape[0] // scale, img.shape[1] // scale), interpolation=cv2.INTER_CUBIC)
    lr = cv2.resize(lr, hr_size, interpolation=cv2.INTER_CUBIC)

    for i in range(0, lr.shape[0]-patch_size+1, stride):
        for j in range(0, lr.shape[1]-patch_size+1, stride):
            lr_patches.append(lr[i:i+patch_size, j:j+patch_size])
            hr_patches.append(hr[i:i+patch_size, j:j+patch_size])

f.create_dataset('lr', data=lr_patches)
f.create_dataset('hr', data=hr_patches)
f.close()
