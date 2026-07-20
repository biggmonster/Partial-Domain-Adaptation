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


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def image_transform(resize_size=256, crop_size=224):
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


def build_source_records(
    mat_path,
    class_num,
    samples_per_class,
    source_samples_per_class,
):
    """
    打开MAT并获得形状
        ↓
    根据每类连续存放1,000条的规则,计算每类前360条的索引
        ↓
    保存(mat_index, label)
    """

    with Mat73SignalReader(
        mat_path, samples_per_class=samples_per_class
    ) as reader:
        shape = reader.validate_layout()    # X (2, 4800, 50000)

    records = []
    for label in range(class_num):
        class_start = label * samples_per_class
        class_end = class_start + source_samples_per_class
        
        records.extend((mat_index, label) for mat_index in range(class_start, class_end))
    return records, shape


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
    SD_records, mat_shape = build_source_records(
        args.raw_mat_path,
        args.class_num,
        args.samples_per_class,
        args.source_samples_per_class,
    )
    train_records, validation_records = split_train_valid(
        SD_records, args.source_val_ratio, args.seed
    )
    transform = image_transform(args.gasf_size, args.crop_size)

    SD_dataset = data_list.MatSourceGASFDataset(
        SD_records,
        mat_path=args.raw_mat_path,
        transform=transform,
        gasf_size=args.gasf_size,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
        augment_config=augment_config,
        contrastive=True,         # 在预阶段数据增强
        snr_db=args.s,
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

    print(
        f"MAT X shape: {mat_shape}; source uses the first "
        f"{args.source_samples_per_class}/{args.samples_per_class} samples per class; "
        f"train={len(SD_dataset)}, validation={0}"
    )
    return SD_loader


def build_models(args, device):
    if "ResNet" in args.net:
        parameters = {
            "resnet_name": args.net,
            "use_bottleneck": True,
            "bottleneck_dim": args.fdim,
            "new_cls": True,
            "embedding_dim": args.edim,
        }
        base_network = en_network.ResNetFc(**parameters)
    elif args.net == "RepVGG_B1g2":
        parameters = {
            "use_bottleneck": True,
            "bottleneck_dim": args.fdim,
            "new_cls": True,
            "embedding_dim": args.edim,
        }
        base_network = en_network.RepVGG_B1g2(**parameters)
    else:
        raise ValueError(f"Unsupported network: {args.net}")

    classifier = en_network.StochasticClassifier(
        base_network.output_num(), args.class_num
    )

    return (
        base_network.to(device),
        classifier.to(device),
    )


def classifier_loss(classifier, features, labels, samples):
    loss = 0.0
    for _ in range(samples):
        logits = classifier(features)
        loss = loss + torch.nn.functional.cross_entropy(logits, labels)
    return loss / samples


def evaluate_source(
    loader,
    base_network,
    classifier,
    device,
    amp_enabled,
    class_num,
):
    base_network.eval()
    classifier.eval()

    correct = 0
    total = 0

    # 每个类别预测正确的数量
    class_correct = torch.zeros(
        class_num,
        dtype=torch.long,
        device=device,
    )

    # 每个类别的验证样本总数
    class_total = torch.zeros(
        class_num,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        for images, _, labels, _ in loader:
            images = images.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            ).long()

            with torch.cuda.amp.autocast(
                enabled=amp_enabled
            ):
                features, _, _ = base_network(images)
                logits = classifier(
                    features,
                    mode="test",
                )

            predictions = logits.argmax(dim=1)
            matches = predictions.eq(labels)

            # 计算总体正确数
            correct += matches.sum().item()
            total += labels.numel()

            # 统计每个类别的样本总数
            class_total += torch.bincount(
                labels,
                minlength=class_num,
            )

            # 统计每个类别预测正确的样本数
            class_correct += torch.bincount(
                labels[matches],
                minlength=class_num,
            )

    accuracy = correct / total if total else 0.0

    class_correct_cpu = class_correct.cpu()
    class_total_cpu = class_total.cpu()

    print("\nSource validation accuracy by class:")
    print("-" * 55)

    # for class_index in range(class_num):
    #     category_correct = class_correct_cpu[
    #         class_index
    #     ].item()
    #     category_total = class_total_cpu[
    #         class_index
    #     ].item()

    #     if category_total > 0:
    #         category_accuracy = (
    #             category_correct / category_total
    #         )

    #         print(
    #             f"C{class_index + 1:02d} "
    #             f"(label={class_index:02d}): "
    #             f"{category_accuracy * 100:6.2f}% "
    #             f"({category_correct:2d}/"
    #             f"{category_total:2d})"
    #         )
    #     else:
    #         print(
    #             f"C{class_index + 1:02d} "
    #             f"(label={class_index:02d}): "
    #             "N/A (0 samples)"
    #         )

    print("-" * 55)
    print(
        f"Overall: {accuracy * 100:.2f}% "
        f"({correct}/{total})"
    )

    return accuracy


def save_checkpoint(
    checkpoint_path,
    args,
    augment_config,
    epoch,
    validation_accuracy,
    base_network,
    classifier,
):
    checkpoint = {
        "base_network": copy.deepcopy(base_network.state_dict()),
        "classifier": copy.deepcopy(classifier.state_dict()),
        "source_val_acc": validation_accuracy,
        "epoch": epoch,
        "seed": args.seed,
        "net": args.net,
        "class_num": args.class_num,
        "classifier_num": args.classifiers_num,
        "training_args": vars(args).copy(),
        "augmentation": augment_config.to_dict(),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def train(args):

    device = torch.device("cuda")
    amp_enabled = bool(args.amp)
    augment_config = make_augment_config(args)
    SD_loader = make_loaders(args, augment_config)
    base_network, classifier = build_models(args, device)

    parameter_list = (
        base_network.get_parameters()
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
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = (
        Path(args.save_dir)
        / args.net
        / f"SD_{args.class_num}class_SNR_{args.s}_pretrain_{timestamp}.pt"
    )
    best_accuracy = -1.0
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(1, args.max_epoch + 1):
        base_network.train()
        classifier.train()
        ce_total = 0.0
        contrast_total = 0.0
        batches = 0
        progress = tqdm(
            SD_loader,
            desc=f"Epoch {epoch:03d}/{args.max_epoch:03d}",
            dynamic_ncols=True,
            unit="batch",
        )
        
        for original, view_one, labels, _ in progress:
            optimizer = scheduler(optimizer, global_step, **schedule_parameters)
            optimizer.zero_grad(set_to_none=True)
            labels = labels.to(device, non_blocking=True).long()

            original = original.to(device, non_blocking=True)
            augmented = view_one.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):       # 使用混合精度，自动选择计算精度
                combined = torch.cat((original, augmented), dim=0)
                combined_features, _, _ = base_network(combined)
                original_features, augmented_features = combined_features.chunk(
                    2, dim=0
                )
                ce_loss = classifier_loss(
                    classifier, original_features, labels, args.classifiers_num
                )
                contrast_loss = my_loss.LabelGuidedContrastiveLoss(
                    original_features,
                    augmented_features,
                    labels,
                    temperature=args.temperature,
                )
                weighted_contrast_loss = args.contrast_weight * contrast_loss
                total_loss = ce_loss + weighted_contrast_loss

            scaler.scale(total_loss).backward()
            # scaler.scale(ce_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ce_total += ce_loss.detach().item()
            contrast_total += contrast_loss.detach().item()
            batches += 1
            global_step += 1
            progress.set_postfix(
                ce=f"{ce_total / batches:.4f}",
                label_con=f"{contrast_total / batches:.4f}",
            )

        test_interval_epoch = int(args.max_epoch / 20)
        if epoch % test_interval_epoch == 0 or epoch == args.max_epoch:
            validation_accuracy = evaluate_source(
                SD_loader,
                base_network,
                classifier,
                device,
                amp_enabled,
                args.class_num
            )
            print(
                f"Epoch {epoch:03d}: CE={ce_total / max(1, batches):.6f}, "
                f"Label-CON={contrast_total / max(1, batches):.6f}, "
                f"source_val_acc={validation_accuracy:.6f}"
            )

            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                epochs_without_improvement = 0
                save_checkpoint(
                    checkpoint_path,
                    args,
                    augment_config,
                    epoch,
                    validation_accuracy,
                    base_network,
                    classifier,
                )
                print(f"Saved best source-only pretraining checkpoint: {checkpoint_path}")
            else:
                epochs_without_improvement += test_interval_epoch
                if epochs_without_improvement >= args.early_stop:
                    print(
                        f"Early stopping after {args.early_stop} epochs without "
                        "source-validation improvement."
                    )
                    break

    print(f"Best source validation accuracy: {best_accuracy:.6f}")
    print(f"Pretraining checkpoint: {checkpoint_path}")
    return best_accuracy


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
    parser.add_argument("--t", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=str, default="run", help=argparse.SUPPRESS)
    parser.add_argument("--dset", type=str, default="LongSig_50", help=argparse.SUPPRESS)
    parser.add_argument("--cls_type", type=str, default="fixed_30class", help=argparse.SUPPRESS)
    parser.add_argument("--multi_class", type=str, default="30", help=argparse.SUPPRESS)
    parser.add_argument("--K", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--net", type=str, default="RepVGG_B1g2")
    parser.add_argument("--class_num", type=int, default=50)
    parser.add_argument("--classifiers_num", type=int, default=3)
    parser.add_argument("--early_stop", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_gamma", type=float, default=0.001)
    parser.add_argument("--lr_power", type=float, default=0.75)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--fdim", type=int, default=512)
    parser.add_argument("--edim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--contrast_weight", type=float, default=0.1)
    parser.add_argument("--source_val_ratio", type=float, default=0.0)
    parser.add_argument("--samples_per_class", type=int, default=1000)
    parser.add_argument("--source_samples_per_class", type=int, default=360)

    parser.add_argument("--raw_mat_path", type=str,
        default=
        r"D:\LJ\workstation\Matlab\ads-b_data_gen"
        r"\data_LongSig_50"
        r"\extracted_50classes_1000samples.mat",
        help='raw I/Q signal'
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

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
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
