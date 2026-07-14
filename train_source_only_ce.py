import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import data_list
import en_network


CLASS_NUM = 50
BOTTLENECK_DIM = 512
EMBEDDING_DIM = 256


def image_train(resize_size=256, crop_size=224):
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def parse_args():
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train a strict source-domain-only RepVGG classifier."
    )
    parser.add_argument(
        "--source-list",
        type=Path,
        default=project_root
        / "data"
        / "LongSig_50"
        / "Source"
        / "GASF_old"
        / "SNR_20_list.txt",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root
        / "model"
        / "RepVGG_B1g2"
        / "source_only_ce_SNR20_seed2025.pt",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=project_root
        / "results"
        / "RepVGG"
        / "only_SD"
        / "source_only_ce_train_metrics.json",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_lines_and_validate(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    labels = [int(line.rsplit(maxsplit=1)[1]) for line in lines if line.strip()]
    if not labels:
        raise ValueError(f"No samples found in: {path}")
    unique_labels = sorted(set(labels))
    if min(labels) < 0 or max(labels) >= CLASS_NUM:
        raise ValueError(
            f"Labels must be in [0, {CLASS_NUM - 1}], "
            f"but found [{min(labels)}, {max(labels)}]."
        )
    if unique_labels != list(range(CLASS_NUM)):
        raise ValueError(f"Expected all {CLASS_NUM} source labels, found: {unique_labels}")
    return [line + "\n" for line in lines if line.strip()], labels


def make_loader(lines, transform, args, shuffle):
    dataset = data_list.ImageList(lines, transform=transform)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )


def create_model():
    return en_network.RepVGG_B1g2(
        use_bottleneck=True,
        bottleneck_dim=BOTTLENECK_DIM,
        new_cls=True,
        class_num=CLASS_NUM,
        embedding_dim=EMBEDDING_DIM,
    )


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            _, outputs, _, _ = model(inputs)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)
    lines, labels = read_lines_and_validate(args.source_list.resolve())
    train_loader = make_loader(lines, image_train(), args, shuffle=True)
    eval_loader = make_loader(lines, image_test(), args, shuffle=False)

    model = create_model().to(device)
    if tuple(model.fc.weight.shape) != (CLASS_NUM, BOTTLENECK_DIM):
        raise ValueError(f"Unexpected classifier shape: {tuple(model.fc.weight.shape)}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )

    print("Training strict Source Only CE baseline")
    print(f"Source list: {args.source_list.resolve()}")
    print(f"Samples: {len(labels)}, labels: [{min(labels)}, {max(labels)}]")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}", unit="batch")
        for inputs, batch_labels, _ in progress:
            inputs = inputs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            _, outputs, _, _ = model(inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_labels.size(0)
            sample_count += batch_labels.size(0)
            progress.set_postfix(loss=f"{running_loss / sample_count:.4f}")
        print(f"epoch={epoch:02d}, train_loss={running_loss / sample_count:.6f}")

    source_accuracy = evaluate(model, eval_loader, device)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    metrics = {
        "model": "RepVGG_B1g2",
        "training_mode": "source_only_cross_entropy",
        "source_list": str(args.source_list.resolve()),
        "source_samples": len(labels),
        "class_num": CLASS_NUM,
        "label_min": min(labels),
        "label_max": max(labels),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "final_source_accuracy": source_accuracy,
        "checkpoint": str(args.checkpoint.resolve()),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Final source accuracy: {source_accuracy:.4%}")
    print(f"Checkpoint saved to: {args.checkpoint.resolve()}")
    print(f"Metrics saved to: {args.metrics.resolve()}")


if __name__ == "__main__":
    main()
