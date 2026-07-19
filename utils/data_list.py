#from __future__ import print_function, division    全是

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
import os

from utils.GASF import (
    Mat73SignalReader,
    SignalAugmentConfig,
    add_measured_awgn,
    augment_iq,
    iq_to_pil,
)

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


class MatSourceGASFDataset(Dataset):
    """Generate source-domain GASF inputs directly from MATLAB I/Q signals.

    Records contain only ``(mat_index, label)`` pairs. In training mode the
    dataset returns one deterministic, unaugmented GASF and one independently
    augmented GASF view. In evaluation mode it returns only the deterministic
    GASF, label and MAT index.
    """

    def __init__(
        self,
        records,
        mat_path,
        transform=None,
        gasf_size=256,
        seed=2026,
        samples_per_class=1000,
        augment_config=None,
        contrastive=True,
        snr_db=None,
    ):
        self.records = [(int(index), int(label)) for index, label in records]
        self.transform = transform
        self.gasf_size = int(gasf_size)
        self.seed = int(seed)
        self.augment_config = augment_config or SignalAugmentConfig()
        self.contrastive = bool(contrastive)
        self.snr_db = None if snr_db is None else float(snr_db)
        if self.snr_db is not None and not np.isfinite(self.snr_db):
            raise ValueError(f"snr_db must be finite or None, got {snr_db!r}")
        self.signal_reader = Mat73SignalReader(
            mat_path, samples_per_class=samples_per_class
        )
        self._rng = None
        self._rng_owner = None

    def _get_rng(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        owner = (os.getpid(), worker_id)
        if self._rng is None or self._rng_owner != owner:
            # One independent deterministic stream per worker. Persistent
            # workers keep advancing the stream across epochs.
            self._rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, worker_id])
            )
            self._rng_owner = owner
        return self._rng

    # __getitem__()：读取并处理一个样本
    def __getitem__(self, index):
        mat_index, target = self.records[index]
        signal, actual_label = self.signal_reader.read_index(mat_index)
        if actual_label != target:
            raise ValueError(
                f"Label mismatch at MAT index {mat_index}: "
                f"record={target}, mat={actual_label}"
            )

        # Fixed domain noise is deterministic per MAT sample and independent
        # of worker scheduling, shuffling, epochs and augmentation RNG state.
        if self.snr_db is not None:
            noise_rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, mat_index])
            )
            signal = add_measured_awgn(signal, self.snr_db, noise_rng)
        
        # 生成原始GASF
        original = iq_to_pil(signal, target_size=self.gasf_size)
        if self.transform is not None:
            original = self.transform(original)

        if not self.contrastive:
            return original, target, mat_index

        rng = self._get_rng()
        signal_view_one = augment_iq(signal, rng, self.augment_config)
        view_one = iq_to_pil(signal_view_one, target_size=self.gasf_size)
        if self.transform is not None:
            view_one = self.transform(view_one)
        return original, view_one, target, mat_index

    def __len__(self):
        return len(self.records)
