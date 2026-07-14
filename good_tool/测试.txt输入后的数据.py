import argparse
import os, random, pdb, math, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import Dataset
from PIL import Image

# print(torch.__version__)        # 应输出 '1.1.0'
# print(torch.version.cuda)      # 应输出 '9.0'
# print(torch.cuda.is_available())  # 应输出 True  测试cuda是否可用

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
    with open(path, 'rb') as f:        # 打开路径path下的图像，并且转化位三通道RGB
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):       # 加载图像并转换为灰度格式 ，根据ImageList的mode选择RGB或L   line 65/67
     
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')   

def image_train(resize_size=256, crop_size=224):   # resize_size指定图像在裁剪前的尺寸，crop_size随机裁剪后图片的尺寸
    return  transforms.Compose([                   
        transforms.Resize((resize_size, resize_size)),    # 调整输入图像的尺寸为256
        transforms.RandomCrop(crop_size),                 # 随即裁剪
        transforms.RandomHorizontalFlip(),                # 0.5概率翻转图像
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class ImageList(Dataset):  
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels) # 自定义函数，根据给定的图像列表和标签信息，生成一个包含图像路径和对应标签的数据集列表，列表元素是元组，imgs是ImageList的实例属性，可以通过dsets['source'].imgs直接调用查看
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):        # 从路径读取图片的关键代码
        path, target = self.imgs[index]      #  获取图像路径和标签
        img = self.loader(path)               # 🎯 通过路径加载图片的核心操作self.loader----根据mode选择了rbg_loader
        if self.transform is not None:
            img = self.transform(img)         # 处理图像
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)   # 输出数据集的大小
    
if __name__=='__main__':

    # /data0/liangjie/Implicit-Semantic-Response-Alignment-main/data/office_home/modify_test_list.txt
    # /data1/liangjie/New_ISRA/data/ads-b/X_15_100_-5dB_list.txt
    dset = 'ads-b'
    names = [ 'X_15_100_-5dB', 'X_15_100_0dB' ]      # 不同风格列表/ 不同干扰模式列表
    data_folder = '/data1/liangjie/New_ISRA/data/'   # 数据集路径
    s_dset_path = data_folder + dset + '/' + names[0] + '_list.txt'   # ./data/office_home/test_list.txt

    # s_dset_path_list = open(s_dset_path).readlines()  # 是一个列表，每.txt的一行作为列表的一个元素
    # print(type(list_1)) 
    # print(list_1)
    '''
    dsets数据集字典
    '''
    dsets = {}
    dsets["test"] = ImageList(open(s_dset_path).readlines(), transform=image_train())   # open(path).readlines()返回列表，每个元素是文件中的一行内容
    # print(type(dsets['test']))    # <class '__main__.ImageList'>表示这是在主程序脚本内定义的类
    index = 2  # 选择要查看的样本索引
    img, target = dsets['test'].__getitem__(index)
    print(f"图像张量形状: {img.shape}, 标签: {target}")

    '''
    linux系统需要借助GUI来显示图像，服务器估计没有安装
    '''
    # # 反转归一化
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # 与 transform 中的 mean 一致
    # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)   # 与 transform 中的 std 一致
    # img_denorm = img * std + mean  # 反转归一化公式：(img * std) + mean
    # img_denorm = img_denorm.clamp(0, 1)  # 确保像素值在 [0, 1] 范围内

    # # 转换为 PIL 图像
    # to_pil = transforms.ToPILImage()
    # img_pil = to_pil(img_denorm)

    # # 显示图像
    # plt.imshow(img_pil)
    # plt.title(f"Label: {target}")  # 显示标签
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()


    # '''
    # dset_loaders数据集加载器
    # '''
    # dset_loaders = {}         # 数据会被打乱顺序（shuffle=True）,workers个工作线程进行数据加载, 并且丢弃最后一个不完整的批次（drop_last=True）
    # dset_loaders["test"] = DataLoader(dsets["test"], batch_size=36, shuffle=True, num_workers=4, drop_last=True)






