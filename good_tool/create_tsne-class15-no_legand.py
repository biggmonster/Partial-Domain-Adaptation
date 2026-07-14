import argparse
import json
import random
from pathlib import Path
import sys
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

import data_list
import en_network

from dotenv import load_dotenv
import os

# Load the shared class selection from the project-level .env file.
load_dotenv(PROJECT_ROOT / ".env")


def load_class_ids():
    raw_class_ids = os.getenv("CLASS_IDS_15")
    class_ids = json.loads(raw_class_ids)
    return class_ids


CLASS_IDS = load_class_ids() 
CLASS_NAMES = {class_id: f"C{index}" for index, class_id in enumerate(CLASS_IDS, 1)} 
SOURCE_SNR = "SNR_20"
TARGET_SNR = "SNR_0"
SAMPLES_PER_CLASS = 200 
SAMPLES_PER_DOMAIN = len(CLASS_IDS) * SAMPLES_PER_CLASS 
TOTAL_SAMPLES = SAMPLES_PER_DOMAIN * 2
COLOR_VALUES = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
]
MARKER_VALUES = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">", "h", "H", "p", "8", "d"]
if len(CLASS_IDS) > len(COLOR_VALUES):
    raise ValueError(f"At most {len(COLOR_VALUES)} CLASS_IDS are supported.")
COLORS = dict(zip(CLASS_IDS, COLOR_VALUES))
MARKERS = dict(zip(CLASS_IDS, MARKER_VALUES))


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
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Export trained model features and draw shared-coordinate t-SNE figures."
    )
    parser.add_argument(
        "--source-list",
        type=Path,
        default=project_root/ "data"/ "LongSig_50"/ "t-SNE"/ "19-27-47-5-38"/ "class_5"/ "source"/ "SNR_20_200_list.txt",
    )

    parser.add_argument(
        "--target-list",
        type=Path,
        default=project_root/ "data"/ "LongSig_50"/ "t-SNE"/ "19-27-47-5-38"/ "class_5"/ "target"/ "SNR_0_200_list.txt",
    )

    parser.add_argument(
        "--feature-extractor",
        choices=["pretrained_resnet50", "trained_repvgg", "trained_resnet50"],
        default="pretrained_resnet50",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=project_root/'results'/ "t-SNE_results"/ "pretrained_resnet50"/ "class_5",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
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


def read_and_validate_list(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.strip()]
    labels = [int(line.rsplit(maxsplit=1)[1]) for line in lines]
    counts = {class_id: labels.count(class_id) for class_id in CLASS_IDS}
    if len(lines) != SAMPLES_PER_DOMAIN or counts != {
        class_id: SAMPLES_PER_CLASS for class_id in CLASS_IDS
    }:
        raise ValueError(
            f"{path} must contain {SAMPLES_PER_DOMAIN} samples with "
            f"{SAMPLES_PER_CLASS} samples for each "
            f"label in {CLASS_IDS}; found total={len(lines)}, counts={counts}."
        )
    if sorted(set(labels)) != sorted(CLASS_IDS):
        raise ValueError(f"Unexpected labels in {path}: {sorted(set(labels))}")
    return [line + "\n" for line in lines]


def make_loader(list_path, args):
    dataset = data_list.ImageList(read_and_validate_list(list_path), transform=image_test())
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )


class RepVGGFeatureExtractor(nn.Module):
    def __init__(self, checkpoint, device):
        super().__init__()
        model = en_network.RepVGG_B1g2(
            use_bottleneck=True,
            bottleneck_dim=512,
            new_cls=True,
            class_num=50,
            embedding_dim=256,
        )
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model = model

    '''
    这个feature是512维度的, 属于encoder之前的
    '''
    def forward(self, inputs):
        features, logits, _, _ = self.model(inputs)
        return features, logits

    # '''
    # 这个feature是256维度的, 属于encoder之后的 
    # '''
    # def forward(self, inputs):
    #     _, logits, feature, _ = self.model(inputs)
    #     return feature, logits
    
    
    

class ResNet50FeatureExtractor(nn.Module):
    '''
    在 CDPDA框架下的resnet50训练
    '''
    def __init__(self, checkpoint, device):
        super().__init__()
        state_dict = torch.load(checkpoint, map_location=device)
        class_num = state_dict["fc.weight"].shape[0]
        model = en_network.ResNetFc(
            resnet_name="ResNet50",
            use_bottleneck=True,
            bottleneck_dim=512,
            new_cls=True,
            class_num=class_num,
            embedding_dim=256,
        )
        model.load_state_dict(state_dict)
        self.model = model

    '''
    这个feature是512维度的, 属于encoder之前的
    '''
    def forward(self, inputs):
        features, logits, _, _ = self.model(inputs)
        return features, logits



class PretrainedResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(model.children())[:-1])

    def forward(self, inputs):
        return torch.flatten(self.backbone(inputs), 1)


def create_model(args, device):
    """Create a feature extractor whose forward() returns [batch_size, feature_dim]."""
    if args.feature_extractor == "pretrained_resnet50" : 
        raise ValueError(
            "pretrained_resnet50 has no project classification head, so accuracy "
            "cannot be computed. Please use trained_repvgg or trained_resnet50."
        )
    if args.feature_extractor == "trained_resnet50":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for trained_resnet50.")
        return ResNet50FeatureExtractor(args.checkpoint, device), 512
    if args.feature_extractor == "trained_repvgg":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for trained_repvgg.")
        return RepVGGFeatureExtractor(args.checkpoint, device), 512
    
    raise ValueError(f"Unsupported feature_extractor: {args.feature_extractor}")


def extract_features(model, loader, device, feature_dim):
    feature_batches = []
    label_batches = []
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels_on_device = labels.to(device, non_blocking=True)
            features, logits = model(inputs)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels_on_device).sum().item()
            total += labels_on_device.size(0)
            feature_batches.append(features.cpu().numpy())
            label_batches.append(labels.cpu().numpy())
    features = np.vstack(feature_batches)
    labels = np.concatenate(label_batches)
    if features.shape != (SAMPLES_PER_DOMAIN, feature_dim):
        raise ValueError(f"Unexpected feature shape: {features.shape}")
    accuracy = correct / total if total else 0.0
    return features, labels, accuracy



def bounds_with_padding(values):
    minimum = values.min()
    maximum = values.max()
    padding = max((maximum - minimum) * 0.05, 1.0)
    return minimum - padding, maximum + padding


def add_points(ax, source_points, source_labels, target_points, target_labels):
    for class_id in CLASS_IDS:
        source_mask = source_labels == class_id
        target_mask = target_labels == class_id
        ax.scatter(
            source_points[source_mask, 0],
            source_points[source_mask, 1],
            color=COLORS[class_id],
            marker=MARKERS[class_id],
            s=28,
            alpha=0.35,
            linewidths=0,
        )
        ax.scatter(
            target_points[target_mask, 0],
            target_points[target_mask, 1],
            color=COLORS[class_id],
            marker=MARKERS[class_id],
            s=28,
            alpha=1,
            linewidths=0,
        )


def legend_handles():
    source_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[class_id],
            linestyle="",
            markerfacecolor=COLORS[class_id],
            markeredgecolor="none",
            markersize=6,
            alpha=0.35,
            label=f"SD {CLASS_NAMES[class_id]}",
        )
        for class_id in CLASS_IDS
    ]
    target_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[class_id],
            linestyle="",
            markerfacecolor=COLORS[class_id],
            markeredgecolor="none",
            markersize=6,
            alpha=1,
            label=f"TD {CLASS_NAMES[class_id]}",
        )
        for class_id in CLASS_IDS
    ]
    return [
        handle
        for pair in zip(source_handles, target_handles)
        for handle in pair
    ]


def figure_stem():
    return "SNR20_to_SNR0"


def draw_figures(
    plots_dir, source_points, source_labels, target_points, target_labels
):
    plots_dir.mkdir(parents=True, exist_ok=True)
    all_points = np.vstack([source_points, target_points])
    xlim = bounds_with_padding(all_points[:, 0])
    ylim = bounds_with_padding(all_points[:, 1])

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    add_points(ax, source_points, source_labels, target_points, target_labels)
    # ax.set_xlabel("t-SNE dimension 1")
    # ax.set_ylabel("t-SNE dimension 2")
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    
    stem = figure_stem()
    fig.savefig(plots_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(plots_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return xlim, ylim


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)
    model, feature_dim = create_model(args, device)
    model = model.to(device)

    source_features, source_labels, source_accuracy = extract_features(
        model, make_loader(args.source_list, args), device, feature_dim
    )

    target_features, target_labels, target_accuracy = extract_features(
        model, make_loader(args.target_list, args), device, feature_dim
    )
    overall_accuracy = (
        (source_accuracy * len(source_labels)) + (target_accuracy * len(target_labels))
    ) / (len(source_labels) + len(target_labels))

    print(f"Source accuracy: {source_accuracy * 100:.2f}%")
    print(f"Target accuracy: {target_accuracy * 100:.2f}%")
    print(f"Overall accuracy: {overall_accuracy * 100:.2f}%")

    all_features = np.vstack([source_features, target_features])
    if all_features.shape != (TOTAL_SAMPLES, feature_dim):
        raise ValueError(f"Unexpected combined feature shape: {all_features.shape}")
    print(f"Running t-SNE for feature matrix: {all_features.shape}")
    embedding = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    ).fit_transform(all_features)
    if embedding.shape != (TOTAL_SAMPLES, 2):
        raise ValueError(f"Unexpected t-SNE embedding shape: {embedding.shape}")

    source_points = embedding[:SAMPLES_PER_DOMAIN]
    target_points = embedding[SAMPLES_PER_DOMAIN : SAMPLES_PER_DOMAIN * 2]
    embedding_rows = []
    for index, (point, label) in enumerate(zip(source_points, source_labels)):
        embedding_rows.append(
            {
                "domain": "source",
                "snr": SOURCE_SNR,
                "sample_index": index,
                "label": int(label),
                "class_name": CLASS_NAMES[int(label)],
                "tsne_x": point[0],
                "tsne_y": point[1],
            }
        )
    for index, (point, label) in enumerate(zip(target_points, target_labels)):
        embedding_rows.append(
            {
                "domain": "target",
                "snr": TARGET_SNR,
                "sample_index": index,
                "label": int(label),
                "class_name": CLASS_NAMES[int(label)],
                "tsne_x": point[0],
                "tsne_y": point[1],
            }
        )


    # figure_title = (
    #     f"{args.feature_extractor}: Source SNR 20 vs Target SNR 0"
    # )
    xlim, ylim = draw_figures(
        args.plots_dir,
        source_points,
        source_labels,
        target_points,
        target_labels,
        # figure_title,
    )
    metrics = {
        "feature_extractor": args.feature_extractor,
        "weights": (
            str(ResNet50_Weights.DEFAULT)
            if args.feature_extractor == "pretrained_resnet50"
            else str(args.checkpoint.resolve())
        ),
        "seed": args.seed,
        "class_mapping": {CLASS_NAMES[class_id]: class_id for class_id in CLASS_IDS},
        "samples_per_class": SAMPLES_PER_CLASS,
        "source_samples": len(source_labels),
        "target_samples": len(target_labels),
        "source_accuracy": source_accuracy,
        "target_accuracy": target_accuracy,
        "overall_accuracy": overall_accuracy,
        "feature_shape": list(all_features.shape),
        "embedding_shape": list(embedding.shape),
        "shared_axis_limits": {"x": list(xlim), "y": list(ylim)},
    }
    (args.plots_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Figures saved to: {args.plots_dir.resolve()}")


if __name__ == "__main__":
    main()
