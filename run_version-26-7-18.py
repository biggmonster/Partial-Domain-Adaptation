import argparse
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import copy
import network.en_network as en_network
import utils.data_list as data_list
import utils.lr_schedule as lr_schedule
import utils.my_loss as my_loss
from utils.GASF import (
    Mat73SignalReader,
    SignalAugmentConfig,
)
import torch.nn.functional as F


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def image_train_transform(resize_size=256, crop_size=224):
    """Deterministic image conversion; augmentation is performed on I/Q only.
    GASF RGB图片
        ↓ 调整尺寸
    256*256
        ↓ 随机裁剪
    224*224
        ↓ 随机水平翻转
        ↓ 转成PyTorch张量
    [3,224,224]
        ↓ ImageNet标准化
    模型输入
    """

    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def image_test_transform(resize_size=256, crop_size=224):
    """Deterministic image conversion; augmentation is performed on I/Q only.
    GASF RGB图片
        ↓ 调整尺寸
    256*256
        ↓ 中心裁剪
    224*224
        ↓ 转成PyTorch张量
    [3,224,224]
        ↓ ImageNet标准化
    模型输入
    """

    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_source_target_records(
    mat_path,
    class_num,
    samples_per_class,
    source_samples_per_class,
    target_samples_per_class,
    target_class_num,
    seed,
):
    """
    打开MAT并获得形状
        ↓
    根据每类连续存放1,000条的规则,计算每类[0,359]条Source记录, [500,859]条Target记录
        ↓
    保存(mat_index, label)
    """

    with Mat73SignalReader(
        mat_path, samples_per_class=samples_per_class
    ) as reader:
        shape = reader.validate_layout()    # X (2, 4800, 50000)

    if not 1 <= target_class_num <= class_num:
        raise ValueError(
            "target_class_num must be within [1, class_num], "
            f"got target_class_num={target_class_num}, class_num={class_num}"
        )

    source_records = []
    for label in range(class_num):
        source_class_start = label * samples_per_class
        source_class_end = source_class_start + source_samples_per_class
        source_records.extend(
            (mat_index, label)
            for mat_index in range(source_class_start, source_class_end)
        )

    target_rng = random.Random(seed)
    target_class_ids = sorted(
        target_rng.sample(range(class_num), target_class_num)
    )
    target_records = []
    for label in target_class_ids:
        target_class_start = label * samples_per_class + 500
        target_class_end = target_class_start + target_samples_per_class
        target_records.extend(
            (mat_index, label)
            for mat_index in range(target_class_start, target_class_end)
        )

    return source_records, target_records, shape, target_class_ids


def split_train_valid(records, validation_ratio, seed):
    """Split every source class independently and reproducibly."""

    groups = defaultdict(list)  
    for record in records:
        groups[record[1]].append(record)      #按类别分组

    rng = random.Random(seed)
    train_records = []
    validation_records = []
    
    for label in sorted(groups):
        class_records = groups[label][:]    #复制当前类别的记录
        rng.shuffle(class_records)
        validation_count = int(len(class_records) * validation_ratio)  #验证集数量
        validation_records.extend(class_records[:validation_count])
        train_records.extend(class_records[validation_count:])

    return train_records, validation_records


def make_augment_config(args):
    '''
    集中保存+检查参数是否合法
    '''
    return SignalAugmentConfig(
        snr_db_min=args.snr_min,
        snr_db_max=args.snr_max,
        gain_drift_max=args.gain_drift_max,
        gain_drift_probability=args.gain_drift_probability,
        phase_degrees_max=args.phase_max_degrees,
        phase_probability=args.phase_probability,
        crop_keep_min=args.crop_keep_min,
        crop_probability=args.crop_probability,
        time_shift_max_ratio=args.time_shift_max_ratio,
        time_shift_probability=args.time_shift_probability,
    )


def read_lines_and_validate(path, args):
    lines = path.read_text(encoding="utf-8").splitlines()
    labels = [int(line.rsplit(maxsplit=1)[1]) for line in lines if line.strip()]
    if not labels:
        raise ValueError(f"No samples found in: {path}")

    return [line + "\n" for line in lines if line.strip()], labels

def make_loader_22222(lines, transform, args, shuffle, drop_last):
    dataset = data_list.ImageList(lines, transform=transform)
    args.target_class_ids = [0, 3, 5, 6, 7, 9, 10, 12, 14, 15, 20, 21, 25, 26, 28, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 43, 44, 45, 47, 49]
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.worker,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=args.worker > 0,
    )


def make_loaders(args, augment_config):
    '''
    读取MAT标签和结构
        ↓
    构建每类前360条源域记录
        ↓
    按类别划分训练集和验证集
        ↓
    创建在线GASF数据集
        ↓
    创建PyTorch DataLoader    
    '''
    (
        source_records,
        target_records,
        mat_shape,
        target_class_ids,
    ) = build_source_target_records(
        args.raw_mat_path,
        args.SD_class_num,
        args.samples_per_class,
        args.source_samples_per_class,
        args.target_samples_per_class,
        args.TD_class_num,
        args.seed,
    )
    args.target_class_ids = target_class_ids

    train_transform = image_train_transform(args.gasf_size, args.crop_size)
    test_transform = image_test_transform(args.gasf_size, args.crop_size)
    SD_dataset = data_list.MatSourceGASFDataset(
        source_records,
        mat_path=args.raw_mat_path,
        transform=train_transform,
        gasf_size=args.gasf_size,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
        augment_config=augment_config,
        contrastive=False,         # 在训练阶段不添加数据增强
        snr_db=args.s,
    )
    TD_dataset = data_list.MatSourceGASFDataset(
        target_records,
        mat_path=args.raw_mat_path,
        transform=train_transform,
        gasf_size=args.gasf_size,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
        augment_config=augment_config,
        contrastive=False,         # 在训练阶段不添加数据增强
        snr_db=args.t,
    ) 
    test_dataset = data_list.MatSourceGASFDataset(
        target_records,
        mat_path=args.raw_mat_path,
        transform=test_transform,
        gasf_size=args.gasf_size,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
        augment_config=augment_config,
        contrastive=False,         # 在训练阶段不添加数据增强
        snr_db=args.t,
    ) 
    common_loader_args = {
        "num_workers": args.worker,
        "pin_memory": True,
        "persistent_workers": args.worker > 0,
    }
    if args.worker > 0:
        common_loader_args["prefetch_factor"] = args.prefetch_factor

    SD_loader = DataLoader(
        SD_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **common_loader_args,
    )

    TD_loader = DataLoader(
        TD_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **common_loader_args,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2* args.batch_size,
        shuffle=True,
        drop_last=True,
        **common_loader_args,
    )
    print(
        f"MAT X shape: {mat_shape}; source uses the first "
        f"{args.source_samples_per_class}/{args.samples_per_class} samples per class; "
        f"SD_dataset={len(SD_dataset)}, TD_dataset={len(TD_dataset)}; "
        f"SD SNR={args.s} dB, TD SNR={args.t} dB"
    )
    print(
        f"Target classes ({args.TD_class_num}/{args.SD_class_num}): "
        f"{target_class_ids}"
    )
    return SD_loader, TD_loader, test_loader


def _reset_encoder_decoder(base_network):
    """Initialize the encoder/decoder independently of the checkpoint."""

    for module in (base_network.encoder, base_network.decoder):
        for layer in module.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()


def load_pretrained_checkpoint(
    base_network,
    classifier,
    checkpoint_path,
    expected_net,
):
    """Load trained source parameters while reinitializing encoder/decoder.
    读取.pt检查点
        ↓
    检查网络类型是否一致
        ↓
    排除encoder/decoder参数
        ↓
    加载RepVGG和bottleneck
        ↓
    加载随机分类器
        ↓
    重新初始化encoder/decoder
    """

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if checkpoint["net"] != expected_net:
        raise ValueError(
            f"Checkpoint network mismatch: expected={expected_net}, "
            f"actual={checkpoint['net']}"
        )

    excluded_prefixes = ("encoder.", "decoder.")
    pretrained_base_state = checkpoint["base_network"]
    trained_base_state = {
        key: value
        for key, value in pretrained_base_state.items()
        if not key.startswith(excluded_prefixes)
    }
    base_network.load_state_dict(trained_base_state, strict=False)
    classifier.load_state_dict(checkpoint["classifier"], strict=True)
    _reset_encoder_decoder(base_network)

    print(f"Loaded pretrained checkpoint: {checkpoint_path}")
    print("Reinitialized modules: encoder, decoder")


def build_models(args, device):
    if "ResNet" in args.net:
        parameters = {
            "resnet_name": args.net,
            "use_bottleneck": True,
            "bottleneck_dim": args.fdim,
            "embedding_dim": args.edim,
        }
        base_network = en_network.ResNetFc(**parameters)
    elif args.net == "RepVGG_B1g2":
        parameters = {
            "use_bottleneck": True,
            "bottleneck_dim": args.fdim,
            "embedding_dim": args.edim,
        }
        base_network = en_network.RepVGG_B1g2(**parameters)
    else:
        raise ValueError(f"Unsupported network: {args.net}")

    # classifier = en_network.StochasticClassifier(
    #     base_network.output_num(), args.SD_class_num
    # )
    classifier = en_network.common_fc(
        base_network.output_num(), args.SD_class_num
    )
    if args.pretrain_model_path:
        load_pretrained_checkpoint(
            base_network,
            classifier,
            args.pretrain_model_path,
            expected_net=args.net,
        )

    return (
        base_network.to(device),
        classifier.to(device),
    )


def get_Classifier_logits_probs(Classifiers, feat):
    logits = Classifiers(feat)
    probs = torch.nn.Softmax(dim=1)(logits)
    return logits, probs


def classifier_loss(classifier, features, labels, samples):
    loss = 0.0
    for _ in range(samples):
        logits = classifier(features)
        loss = loss + torch.nn.functional.cross_entropy(logits, labels)
    return loss / samples


def image_test_stochastic_classification(loader, base_model, classifiers):
    start_test = True
    base_model.eval()
    classifiers.eval()
    with torch.no_grad():  # 下述代码不进行梯度计算

        iter_test = iter(loader)       # 测试集中batch_size*2 
        total_samples_iter = 0
        for i in range(len(loader)):   # 会遍历一个epoch全部目标域都测试一遍   
            data = next(iter_test)             # iter_test.next()返回一个batch的数据
            inputs = data[0]
            labels = data[1]
            
            batch_size_flag = inputs.shape[0]   # 用于查验test的样本数目
            total_samples_iter += batch_size_flag

            inputs = inputs.cuda()  # 
            out_f,  _, _ = base_model(inputs) # 模型输出特征
            outputs = classifiers(out_f, mode='test')

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()   #
                start_test = False   
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0) 
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)    # .max()选择了其中最大概率输出的类别标签  
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])   # 准确率
    mean_ent = torch.mean(my_loss.Entropy(torch.nn.Softmax(dim=1)(all_output))).cpu().data.item()           # 平均熵 维度[类别数量]，每一个元素值表示这个测试集样本的预测的混乱程度（置信度）

    hist_tar = torch.nn.Softmax(dim=1)(all_output).sum(dim=0)   
    hist_tar = hist_tar / hist_tar.sum()    # 每个类别占总类别的输出概率 （算是归一化了），用于表示一个类别在测试集中的比例
    base_model.train()
    classifiers.train()

    return accuracy, hist_tar, mean_ent


def image_test_classification(loader, base_model, classifiers):
    start_test = True
    base_model.eval()
    classifiers.eval()
    with torch.no_grad():  # 下述代码不进行梯度计算

        iter_test = iter(loader)       # 测试集中batch_size*2 
        total_samples_iter = 0
        for i in range(len(loader)):   # 会遍历一个epoch全部目标域都测试一遍   
            data = next(iter_test)             # iter_test.next()返回一个batch的数据
            inputs = data[0]
            labels = data[1]
            
            batch_size_flag = inputs.shape[0]   # 用于查验test的样本数目
            total_samples_iter += batch_size_flag

            inputs = inputs.cuda()  # 
            out_f,  _, _ = base_model(inputs) # 模型输出特征
            outputs = classifiers(out_f)

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()   #
                start_test = False   
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0) 
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)    # .max()选择了其中最大概率输出的类别标签  
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])   # 准确率
    mean_ent = torch.mean(my_loss.Entropy(torch.nn.Softmax(dim=1)(all_output))).cpu().data.item()           # 平均熵 维度[类别数量]，每一个元素值表示这个测试集样本的预测的混乱程度（置信度）

    hist_tar = torch.nn.Softmax(dim=1)(all_output).sum(dim=0)   
    hist_tar = hist_tar / hist_tar.sum()    # 每个类别占总类别的输出概率 （算是归一化了），用于表示一个类别在测试集中的比例
    base_model.train()
    classifiers.train()

    return accuracy, hist_tar, mean_ent


def save_checkpoint(
    checkpoint_path,
    args,
    augment_config,
    epoch,
    cur_accuracy,
    min_entropy,
    base_network,
    classifier,
):
    checkpoint = {
        "base_network": copy.deepcopy(base_network.state_dict()),
        "classifier": copy.deepcopy(classifier.state_dict()),
        "cur_acc": cur_accuracy,
        "min_entropy": min_entropy,
        "epoch": epoch,
        "seed": args.seed,
        "net": args.net,
        "SD_class_num": args.SD_class_num,
        "TD_class_num": args.TD_class_num,
        "classifier_num": args.classifiers_num,
        "domain_snr_db": {
            "source": float(args.s),
            "target": float(args.t),
        },
        "target_class_ids": list(args.target_class_ids),
        "training_args": vars(args).copy(),
        "augmentation": augment_config.to_dict(),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def next_batch_or_restart(loader, iterator):
    """Read one batch and restart the loader after exhaustion."""

    try:
        inputs, labels, indices = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        inputs, labels, indices = next(iterator)

    return inputs, labels, indices, iterator


def update_epoch_metrics(
    progress,
    loss_totals,
    losses,
    batches,
    global_step,
):
    """Accumulate batch losses and refresh the training progress display."""

    for name in ("src", "rc", "align", "transfer", "tar"):
        loss_totals[name] += losses[name].detach().item()

    batches += 1
    global_step += 1
    progress.set_postfix(
        src=f"{loss_totals['src'] / batches:.4f}",
        rc=f"{loss_totals['rc'] / batches:.4f}",
        align=f"{loss_totals['align'] / batches:.4f}",
        transfer=f"{loss_totals['transfer'] / batches:.4f}",
        tar=f"{loss_totals['tar'] / batches:.4f}",
    )
    return loss_totals, batches, global_step


def print_epoch_summary(
    epoch,
    loss_totals,
    batches,
):
    """Print average training losses and validation accuracy for one epoch."""

    denominator = max(1, batches)
    print(
        f"Epoch {epoch:03d}: CE={loss_totals['src'] / denominator:.6f}, "
        f"Transfer={loss_totals['transfer'] / denominator:.6f}, "
        f"Tar={loss_totals['tar'] / denominator:.6f}, "
        f"RC={loss_totals['rc'] / denominator:.6f}, "
        f"Align={loss_totals['align'] / denominator:.6f}, "
    )




def train(args):

    device = torch.device("cuda")
    print("Training precision: FP32 (mixed precision disabled)")
    augment_config = make_augment_config(args)

    # SD_loader, TD_loader = make_loaders(args, augment_config)
    SD_lines, _ = read_lines_and_validate(args.SD_list.resolve(), args)
    TD_lines, _ = read_lines_and_validate(args.TD_list.resolve(), args)

    train_transform = image_train_transform(args.gasf_size, args.crop_size)
    test_transform = image_test_transform(args.gasf_size, args.crop_size)
    SD_loader = make_loader_22222(SD_lines, train_transform, args, shuffle=True, drop_last=True)
    TD_loader = make_loader_22222(TD_lines, train_transform, args, shuffle=True, drop_last=True)
    test_loader = make_loader_22222(TD_lines, test_transform, args, shuffle=False, drop_last=False)


    base_network, classifier = build_models(args, device)
    MLPS = en_network.MLPS_regressores(args.edim, base_network.bottleneck_dim, 128).to(device)
    ''' 
        1024这里要改: 512 -> 1024 -> 2
    '''
    max_iter = args.max_epoch * max(len(SD_loader), len(TD_loader))
    AD_NET = en_network.AdversarialNetwork(base_network.output_num(), 1024, max_iter).to(device)   # 对抗网络 

    parameter_list = (
        base_network.get_parameters()
        + AD_NET.get_parameters()
        + MLPS.get_parameters()
        + classifier.get_parameters()
    )

    optimizer = torch.optim.SGD(
        parameter_list,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=5e-4,
        nesterov=True,
    )
    schedule_parameters = {
        "lr": args.lr,
        "gamma": args.lr_gamma,
        "power": args.lr_power,
    }
    scheduler = lr_schedule.schedule_dict["inv"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = (
        Path(args.save_dir)
        / args.net
        / f"SD_{args.SD_class_num}class_SNR_{args.s}_TD_{args.TD_class_num}class_SNR_{args.t}_{timestamp}.pt"
    )
    epochs_without_improvement = 0
    global_step = 0

    class_weight = None
    min_entropy = float("inf")

    for epoch in range(1, args.max_epoch + 1):
        
        base_network.train()
        classifier.train()
        AD_NET.train(True)
        MLPS.train(True)
        loss_totals = {
            "src": 0.0,
            "rc": 0.0,
            "align": 0.0,
            "transfer": 0.0,
            "tar": 0.0,
        }
        batches = 0
        num_batch_per_epoch = max(len(SD_loader),len(TD_loader))

        SD_iterator = iter(SD_loader)
        TD_iterator = iter(TD_loader)
        progress = tqdm(
            range(num_batch_per_epoch),
            desc=f"Epoch {epoch:03d}/{args.max_epoch:03d}",
            dynamic_ncols=True,
            unit="batch",
        )

        test_interval_epoch = int(args.max_epoch / 20)
        if epoch > 2 and (epoch % test_interval_epoch == 0 or epoch == args.max_epoch):
            cur_accuracy, class_weight, cur_mean_entropy  = image_test_classification(
                base_model=base_network, 
                classifiers=classifier, 
                loader=test_loader
                )
            class_weight = class_weight.cuda()
            if cur_mean_entropy < min_entropy:
                min_entropy = cur_mean_entropy
                corresponding_accuracy = cur_accuracy
                epochs_without_improvement = 0
                save_checkpoint(
                    checkpoint_path,
                    args,
                    augment_config,
                    epoch,
                    cur_accuracy,
                    min_entropy,
                    base_network,
                    classifier,
                )
                print(f"TD-entropy improved to {min_entropy:.6f} at epoch {epoch-1}, accuracy={cur_accuracy:.6f}")
                print(f"Saved best model checkpoint: {checkpoint_path}")
            else:
                epochs_without_improvement += test_interval_epoch
                if epochs_without_improvement >= args.early_stop:
                    print(
                        f"Early stopping after {args.early_stop} epochs without "
                        "TD-entropy improvement."
                    )
                    break
            
        # 开始对loader进行迭代,每次迭代获取一个batch的数据
        for _ in progress:   

            SD_batch, SD_label, _, SD_iterator  = next_batch_or_restart(SD_loader, SD_iterator)
            TD_batch,        _, _, TD_iterator = next_batch_or_restart(TD_loader, TD_iterator)
            SD_batch, TD_batch, SD_label = SD_batch.cuda(), TD_batch.cuda(), SD_label.cuda()        
            
            optimizer = scheduler(optimizer, global_step, **schedule_parameters)
            optimizer.zero_grad(set_to_none=True)

            fea_s, e_s, r_s = base_network(SD_batch)  # 源域数据输入base网络后的输出
            fea_t, e_t, r_t = base_network(TD_batch)
            features = torch.cat((fea_s, fea_t), dim=0)

            '''
            计算自编码器encoder-decoder的误差loss
            '''
            embeddings = torch.cat((e_s, e_t), dim=0)
            recons = torch.cat((r_s, r_t), dim=0)
            f_dims = []
            e_dims = []
            for j in range(args.edim):   # 遍历每个embedding维度
                f_dims.append(features.detach().clone().requires_grad_(True).cuda())   # .requires_grad_(True) 是计算注意力权重的关键设置
                e_dims.append(embeddings[:, j].detach().clone().cuda())   # 取所有样本在第 j 维上的 embedding 值 放到embeddings[j]上

            rc_loss = F.mse_loss(features, recons)   # 计算均方误差，backbone网络提取的特征X 与 编码解码后输出的X之间的重构误差

            '''
            多层感知机gj的损失函数Loss_reg
            '''
            l1_weight = 0.5         # L1正则化权重系数
            mlp_losses = my_loss.MLP_Reg_Loss(MLPS, f_dims, e_dims, l1_weight)
            optimizer.zero_grad()
            mlp_losses.backward()

            '''
            语义topic对齐
            '''
            align_losses = my_loss.Semantic_Alignment_Loss(
                features,
                f_dims,
                fea_s.size(0),
                fea_t.size(0)
            )

            cls_weight = torch.ones(features.size(0)).cuda()  # 一个全为1的张量 cls_weight
            if class_weight is not None:
                cls_weight[0:fea_s.size(0)] = class_weight[SD_label]

            '''
            Stochastic classifier avg prediction

            源域交叉熵损失*类级权重
            '''
            src_loss, probs_avg = my_loss.StochasticClassifierLoss(
                    classifier,
                    features,
                    SD_label,
                    args.classifiers_num,
                    class_weight
            )

            '''
            DANN损失函数transfer_loss
            '''
            entropy = my_loss.Entropy(probs_avg)           # 计算信息熵（预测概率向量所包含信息的混乱程度）

            transfer_loss = my_loss.DANN_Loss(features, AD_NET, entropy, en_network.calc_coeff(global_step, 1, 0, 10, max_iter), cls_weight)

            '''
            目标域样本的预测信息熵tar_loss
            '''
            probs_t = probs_avg[fea_s.size(0):]
            tar_loss = torch.mean(my_loss.Entropy(probs_t))

            total_loss = src_loss \
                + tar_loss * args.tar_loss_weight \
                + transfer_loss \
                + rc_loss * args.rc \
                + align_losses * args.align

            total_loss.backward()
            optimizer.step()


            loss_totals, batches, global_step = update_epoch_metrics(
                progress,
                loss_totals,
                {
                    "src": src_loss,
                    "rc": rc_loss,
                    "align": align_losses,
                    "transfer": transfer_loss,
                    "tar": tar_loss,
                },
                batches,
                global_step,
            )

        print_epoch_summary(
            epoch,
            loss_totals,
            batches,
        )

    print(f"min TD-entropy: {min_entropy} and corresponding accuracy: {corresponding_accuracy:.6f}")
    print(f"Pretraining checkpoint: {checkpoint_path}")
    return corresponding_accuracy 


def build_parser():
    parser = argparse.ArgumentParser(
        description="Source-only CE + label-guided signal contrastive pretraining"
    )
    # Keep the former command line accepted so existing launch scripts do not
    # fail. Target-domain compatibility options are intentionally ignored.
    parser.add_argument(
        "--s",
        type=float,
        default=20.0,
        help="Source-domain signal-to-noise ratio in dB",
    )
    parser.add_argument(
        "--t",
        type=float,
        default=0.0,
        help="Target-domain signal-to-noise ratio in dB",
    )
    parser.add_argument("--output", type=str, default="run", help=argparse.SUPPRESS)
    parser.add_argument("--dset", type=str, default="LongSig_50", help=argparse.SUPPRESS)
    parser.add_argument("--cls_type", type=str, default="fixed_30class", help=argparse.SUPPRESS)
    parser.add_argument("--classifiers_num", type=int, default=3)
    parser.add_argument("--SD_class_num", type=int, default=50)
    parser.add_argument(
        "--TD_class_num",
        type=int,
        default=30,
        help="Number of randomly selected target-domain classes",
    )
    parser.add_argument("--K", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--net", type=str, default="RepVGG_B1g2")
    
    
    parser.add_argument("--early_stop", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_gamma", type=float, default=0.001)
    parser.add_argument("--lr_power", type=float, default=0.75)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--fdim", type=int, default=512)
    parser.add_argument("--edim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--tar_loss_weight", type=float, default=0.1)
    parser.add_argument('--rc', type=float, default=1.0) 
    parser.add_argument('--align', type=float, default=1.0) 

    parser.add_argument("--source_val_ratio", type=float, default=0.0)
    parser.add_argument("--samples_per_class", type=int, default=1000)
    parser.add_argument("--source_samples_per_class", type=int, default=360)
    parser.add_argument("--target_samples_per_class", type=int, default=360)

    parser.add_argument("--raw_mat_path", type=str,
        default=
        r"D:\LJ\workstation\Matlab\ads-b_data_gen"
        r"\data_LongSig_50"
        r"\extracted_50classes_1000samples.mat",
        help='raw I/Q signal'
    )
    parser.add_argument(
        "--SD_list",
        type=Path,
        default=
        r"D:\LJ\workstation\Vscode\New_ISRA"
        r"\data\LongSig_50"
        r"\Source\GASF_old\SNR_20_list.txt",
        help='MATLAB-GASF SD list file, each line: "gasf-image_path label"',
    )
    parser.add_argument(
        "--TD_list",
        type=Path,
        default=
        r"D:\LJ\workstation\Vscode\New_ISRA"
        r"\data\LongSig_50"
        r"\Target\Multi_class\SNR_0_30_list-test.txt",
        help='MATLAB-GASF TD list file, each line: "gasf-image_path label"',
    )

    
    parser.add_argument("--pretrain_model_path", type=Path,
                        default=None,
                        # default=Path(
                        #     r"D:\LJ\workstation\Vscode\26-7-14"
                        #     r"\model\RepVGG_B1g2"
                        #     r"\SD_50class_SNR_20.0_pretrain_20260719_111232.pt"
                        #     ),
                        help="Path to the pretrained model"
    )
    
    parser.add_argument("--gasf_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--save_dir", type=str, default="model")
    parser.add_argument("--snr_min", type=float, default=20.0)
    parser.add_argument("--snr_max", type=float, default=30.0)
    parser.add_argument("--gain_drift_max", type=float, default=0.10)
    parser.add_argument("--gain_drift_probability", type=float, default=0.50)
    parser.add_argument("--phase_max_degrees", type=float, default=10.0)
    parser.add_argument("--phase_probability", type=float, default=0.50)
    parser.add_argument("--crop_keep_min", type=float, default=0.95)
    parser.add_argument("--crop_probability", type=float, default=0.50)
    parser.add_argument("--time_shift_max_ratio", type=float, default=0.01)
    parser.add_argument("--time_shift_probability", type=float, default=0.30)

    return parser


def print_args(args):
    print("=" * 42)
    print("Source-only contrastive pretraining config")
    print("=" * 42)
    for name, value in vars(args).items():
        print(f"{name}: {value}")
    print("=" * 42)


def validate_args(args):
    if args.crop_size > args.gasf_size:
        raise ValueError("crop_size must not exceed gasf_size")
    if not 1 <= args.TD_class_num <= args.SD_class_num:
        raise ValueError(
            "TD_class_num must be within [1, SD_class_num], "
            f"got TD_class_num={args.TD_class_num}, SD_class_num={args.SD_class_num}"
        )
    # pretrain_model_path = args.pretrain_model_path.expanduser().resolve()
    # if not pretrain_model_path.is_file():
    #     raise FileNotFoundError(
    #         f"Pretrained checkpoint not found: {pretrain_model_path}"
    #     )


def main():
    args = build_parser().parse_args()
    validate_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print_args(args)
    train(args)


if __name__ == "__main__":
    main()
