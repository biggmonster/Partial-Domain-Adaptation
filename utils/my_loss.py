import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb


def LabelGuidedContrastiveLoss(
    original_features,
    augmented_features,
    labels,
    temperature=0.05,
    eps=1e-12,
):
    """Label-guided symmetric contrastive loss for source-domain views.

    The original and augmented views of the same source instance form the
    only positive pair. Views belonging to a different class are negatives,
    while other instances of the same class are excluded from the contrastive
    denominator. Similarities are calculated in float32 for numerical
    stability when automatic mixed precision is enabled.
    """

    if original_features.ndim != 2 or augmented_features.ndim != 2:
        raise ValueError(
            "contrastive inputs must be two-dimensional feature tensors"
        )
    if original_features.shape != augmented_features.shape:
        raise ValueError(
            "original and augmented feature shapes must match, got "
            f"{original_features.shape} and {augmented_features.shape}"
        )
    if original_features.size(0) == 0:
        raise ValueError("contrastive inputs must contain at least one instance")
    if labels.ndim != 1:
        raise ValueError(f"labels must be one-dimensional, got {labels.shape}")
    if labels.size(0) != original_features.size(0):
        raise ValueError(
            f"labels contain {labels.size(0)} entries for "
            f"{original_features.size(0)} feature pairs"
        )
    if labels.device != original_features.device:
        raise ValueError("labels and features must be on the same device")
    if augmented_features.device != original_features.device:
        raise ValueError("original and augmented features must be on the same device")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    # Disable CUDA autocast locally so normalization, matrix multiplication,
    # and logsumexp are all truly evaluated in float32.
    with torch.cuda.amp.autocast(enabled=False):
        original = F.normalize(
            original_features.float(), p=2, dim=1, eps=eps
        )
        augmented = F.normalize(
            augmented_features.float(), p=2, dim=1, eps=eps
        )

        # m_ij = 1 only when sample j belongs to a different class. The
        # diagonal and all other instances of the anchor's class are neutral.
        negative_mask = labels[:, None].ne(labels[None, :])
        valid_anchors = negative_mask.any(dim=1)
        if not bool(valid_anchors.any()):
            # Preserve a differentiable connection to both inputs so that the
            # normal backward pass also works for a single-class batch.
            return (
                original_features.float().sum()
                + augmented_features.float().sum()
            ) * 0.0

        inverse_temperature = 1.0 / float(temperature)
        similarity_oo = (
            torch.matmul(original, original.t()) * inverse_temperature
        )
        similarity_oa = (
            torch.matmul(original, augmented.t()) * inverse_temperature
        )
        similarity_ao = (
            torch.matmul(augmented, original.t()) * inverse_temperature
        )
        similarity_aa = (
            torch.matmul(augmented, augmented.t()) * inverse_temperature
        )

        positive_oa = torch.diagonal(similarity_oa)
        positive_ao = torch.diagonal(similarity_ao)
        negative_fill = float("-inf")

        # Each candidate set contains exactly one positive logit followed by
        # the two different-class negative groups specified in the design.
        original_candidates = torch.cat(
            (
                positive_oa.unsqueeze(1),
                similarity_oo.masked_fill(~negative_mask, negative_fill),
                similarity_oa.masked_fill(~negative_mask, negative_fill),
            ),
            dim=1,
        )
        augmented_candidates = torch.cat(
            (
                positive_ao.unsqueeze(1),
                similarity_ao.masked_fill(~negative_mask, negative_fill),
                similarity_aa.masked_fill(~negative_mask, negative_fill),
            ),
            dim=1,
        )

        loss_original = (
            torch.logsumexp(original_candidates, dim=1) - positive_oa
        )
        loss_augmented = (
            torch.logsumexp(augmented_candidates, dim=1) - positive_ao
        )
        symmetric_loss = 0.5 * (loss_original + loss_augmented)
        return symmetric_loss[valid_anchors].mean().clamp_min(0.0)

def Entropy(input_):           # 计算输入张量的熵--数据的混乱程度 论文《A Balanced and Uncertainty-aware》 公式（2）上两行
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-7)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):           # 梯度反转层的钩子函数（GRL）， coe ff是一个系数，0~1，随模型迭代次数增加而变大
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def MLP_Reg_Loss(MLPS, f_dims, e_dims, l1_weight=0.5):
    '''
    用 MLP_Reg_Loss 计算每个语义维度e_dims 对 feature 的梯度敏感性: f_dim[j].grad
    '''
    mlp_losses = 0.0
    for j in range(len(MLPS)):
        l1_penalty = l1_weight * sum([p.abs().sum() for p in MLPS[j].hidden_1.parameters()])
        prediction = MLPS[j](f_dims[j])
        mlp_loss = F.mse_loss(prediction.view(-1), e_dims[j]) + l1_penalty
        mlp_losses += mlp_loss

    return mlp_losses / len(MLPS)

def Semantic_Alignment_Loss(features, f_dims, source_size, target_size):
    '''
    用f_dim中已有的特征梯度敏感性作为 mask, 去计算 source/target 的语义对齐
    '''
    align_losses = 0.0
    for j in range(len(f_dims)):
        absolute_grad = torch.abs(f_dims[j].grad.data)
        absolute_grad = absolute_grad.div(absolute_grad.norm(p=2, keepdim=True)/(source_size + target_size))
        att_features = torch.mul(features, absolute_grad)

        att_s = att_features[:source_size]
        att_t = att_features[source_size:]

        source_mean = torch.mean(att_s, 0)
        target_mean = torch.mean(att_t, 0)

        align_loss = F.mse_loss(source_mean, target_mean)
        align_losses += align_loss

    return align_losses

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
