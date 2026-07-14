import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

def Entropy(input_):           # 计算输入张量的熵--数据的混乱程度 论文《A Balanced and Uncertainty-aware》 公式（2）上两行
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-7)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):           # 梯度反转层的钩子函数（GRL）， coe ff是一个系数，0~1，随模型迭代次数增加而变大
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def DANN_Loss(features, ad_net, entropy=None, coeff=None, cls_weight=None):        # 对抗性领域适应的损失函数
    '''
    双加权
    TD样本: 熵感知权重加权
    SD样本: 熵感知权重加权 + 类别加权
    '''
    ad_out = ad_net(features)           # 域判别器ad_net，输出0/1
    train_bs = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * train_bs + [[0]] * train_bs )).float().cuda()    # 创建标签，源域1， 目标域0

    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy) # 论文《A Balanced and Uncertainty-aware》 公式（2）上一行的熵感知权重，信息熵越大（分类越不准确），熵感知权重越小                                     
    else:
        entropy = torch.ones(ad_out.size(0)).cuda()  # 初始化一个全为1的一维张量，等价于不加权

    # 源域 权重系数 
    source_mask = torch.ones_like(entropy) # 创建一个和entropy张量具有相同形状和数据类型的新张量，并且所有元素的值都是1。
    source_mask[train_bs ::] = 0  # 令目标域位置全为0, 为的是后面的类级权重系数
    source_weight = entropy * source_mask     # 熵感知权重 
    source_weight = source_weight * cls_weight  # 再 * 类级权重-- 只有源域 有这个类级权重系数 
    # 目标域 权重系数 
    target_mask = torch.ones_like(entropy) 
    target_mask[0 : train_bs] = 0 
    target_weight = entropy * target_mask    # 熵感知权重--目标域样本的分类结果  

    weight = source_weight / (torch.sum(source_weight).detach().item()) + \
             target_weight / torch.sum(target_weight).detach().item()    
    
    weight = weight.view(-1, 1)
    return torch.sum(weight * nn.BCELoss(reduction='none')(ad_out, dc_target)) / (1e-8 + torch.sum(weight).detach().item())     # 计算加权的二元交叉熵损失






