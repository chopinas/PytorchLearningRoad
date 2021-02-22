#数据加载和处理
from __future__ import print_function, division
import os
import torch
import pandas as pd #用于更容易地进行csv解析
# from skimage import io, transform #用于图像的IO和变换
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
# 忽略警告
import warnings
warnings.filterwarnings("ignore")
plt.ion() # interactive mode

#读取数据集
landmarks_frame = pd.read_csv('./data/faces/face_landmarks.csv')
n = 65
img_name = landmarks_frame.iloc[n, 0]
#df.as_matrix()改写成df.values
landmarks = landmarks_frame.iloc[n, 1:].values
landmarks = landmarks.astype('float').reshape(-1, 2)
print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

#编写函数来展示一张图片和它对应的标注点
def show_landmarks(image, landmarks):
 """显示带有地标的图片"""
 plt.imshow(image)
 plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
 plt.pause(0.001) # pause a bit so that plots are updated

#数据集类
class FaceLandmarksDataset(Dataset):
    """面部标记数据集."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        :param csv_file:带注释的csv文件的路径。
        :param root_dir: 包含所有图像的目录。
        :param transform:：一个样本上的可用的可选变换.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """返回数据集尺寸"""
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
         img_name = os.path.join(self.root_dir,
         self.landmarks_frame.iloc[idx, 0])
         image = io.imread(img_name)
         landmarks = self.landmarks_frame.iloc[idx, 1:]
         landmarks = np.array([landmarks])
         landmarks = landmarks.astype('float').reshape(-1, 2)
         sample = {'image': image, 'landmarks': landmarks}
         if self.transform:
             sample = self.transform(sample)
         return sample


#实例化类遍历数据样本
face_dataset = FaceLandmarksDataset(csv_file='./data/faces/face_landmarks.csv',root_dir='./data/faces/')
fig = plt.figure()
for i in range(len(face_dataset)):
    sample = face_dataset[i]
    print(i, sample['image'].shape, sample['landmarks'].shape)
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)
    if i == 3:
        plt.show()
        break

#数据变换,把图片变换成同样的尺寸， * Rescale ：缩放图片 * RandomCrop随机裁剪。* ToTensor ：把numpy格式图片转为torch格式图片 (我们需要交换坐标轴)
class Rescale(object):
    """将样本中的图像重新缩放到给定大小。.
     Args:
     output_size（tuple或int）：所需的输出大小。 如果是元组，则输出为
         与output_size匹配。 如果是int，则匹配较小的图像边缘到output_size保持纵横比相同。
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]
        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """随机裁剪样本中的图像.
     Args:
     output_size（tuple或int）：所需的输出大小。 如果是int，方形裁剪是。
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h,
        left: left + new_w]
        landmarks = landmarks - [left, top]
        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    """将样本中的ndarrays转换为Tensors."""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # 交换颜色轴因为
        # numpy包的图片是: H * W * C
        # torch包的图片是: C * H * W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

#使用上面三个变换类
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
 RandomCrop(224)])
# 在样本上应用上述的每个变换。
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
 transformed_sample = tsfrm(sample)
 ax = plt.subplot(1, 3, i + 1)
 plt.tight_layout()
 ax.set_title(type(tsfrm).__name__)
 show_landmarks(**transformed_sample)
plt.show()
