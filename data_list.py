#from __future__ import print_function, division    全是

import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path

def make_dataset(image_list, labels):   #根据给定的图像列表和标签信息，生成一个包含图像路径和对应标签的数据集。 
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:                               # 根据数据集./data目录下的所有数据，全是路径上自带标签，所以不需要自己输入标签，自带     
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 

def l_loader(path):       # 加载图像并转换为灰度格式 ，根据ImageList的mode选择RGB或L   line 65/67
     
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')   

class ImageList(Dataset):         
    """A generic data loader where the images are arranged in this way:::   通用数据加载器，图像按以下方式排列
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:参数
        root (string): Root directory path.   根目录路径
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
                  (可调用对象，可选)   ：一个函数/变换，接收 PIL 图像并返回变换后的版本
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        loader (callable, optional): A function to load an image given its path.  根据图像路径加载图像的函数。

     Attributes:属性
        classes (list): List of the class names. 列表[class_name]
        class_to_idx (dict): Dict with items (class_name, class_index).  字典（key=class_name, value=class_index）
        imgs (list): List of (image path, class_index) tuples   包含（图像路径，类别索引）元组的列表
    """

    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        #if len(imgs) == 0:                  # 在imgs列表为空时，抛出一个运行时错误，这是在Linux上运行的代码
            # raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
            #                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)

class ImageValueList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=rgb_loader):
        imgs = make_dataset(image_list, labels)
        # if len(imgs) == 0:
        #     raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
        #                        "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.values = [1.0] * len(imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def set_values(self, values):
        self.values = values

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)