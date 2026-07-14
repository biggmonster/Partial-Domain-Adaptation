import argparse
import ast
import csv
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import data_list
import en_network


SNR_VALUES = ["-5", "0", "5", "10", "15", "20"]
CLASS_ENV_NAME = "CLASS_IDS_30"
EXPECTED_CLASS_COUNT = 30
EXPECTED_SAMPLES_PER_CLASS = 360
BOTTLENECK_DIM = 512
EMBEDDING_DIM = 256


@dataclass(frozen=True)
class SnrPair:
    snr: str
    model_path: Path
    list_path: Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create confusion matrices for six matched TD 30-class SNR models."
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device. Default: auto.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers. Default 0 is stable on Windows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / 'results'
        / "confusion_matrix_results"
        / "RepVGG_B1g2"
        / "TD_30class",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Do not write PDF files.")
    return parser.parse_args()


def make_run_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def load_class_ids(env_path):
    if not env_path.exists():
        raise FileNotFoundError(f"Missing .env file: {env_path}")

    raw_value = None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == CLASS_ENV_NAME:
            raw_value = value.strip()
            break

    if raw_value is None:
        raise ValueError(f"{CLASS_ENV_NAME} was not found in {env_path}")

    try:
        class_ids = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse {CLASS_ENV_NAME}: {raw_value}") from exc

    if not isinstance(class_ids, list) or not all(
        isinstance(class_id, int) for class_id in class_ids
    ):
        raise ValueError(f"{CLASS_ENV_NAME} must be a list of integers.")
    if len(class_ids) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"{CLASS_ENV_NAME} must contain {EXPECTED_CLASS_COUNT} classes, "
            f"but found {len(class_ids)}."
        )
    if len(set(class_ids)) != len(class_ids):
        raise ValueError(f"{CLASS_ENV_NAME} contains duplicate class ids: {class_ids}")
    return class_ids


def display_names(class_ids):
    return [f"C{index:02d}" for index in range(1, len(class_ids) + 1)]


def extract_model_snr(path):
    match = re.fullmatch(r"TD_30class_SNR_(-?\d+)_best_model_en\.pt", path.name)
    if not match:
        raise ValueError(f"Unexpected model file name: {path.name}")
    return match.group(1)


def extract_list_snr(path):
    match = re.fullmatch(r"SNR_(-?\d+)_30_list\.txt", path.name)
    if not match:
        raise ValueError(f"Unexpected test list file name: {path.name}")
    return match.group(1)


def build_pairs():
    model_dir = PROJECT_ROOT / "model" / "RepVGG_B1g2"
    list_dir = PROJECT_ROOT / "data" / "LongSig_50" / "Target" / "Multi_class"
    pairs = []

    for snr in SNR_VALUES:
        model_path = model_dir / f"TD_30class_SNR_{snr}_best_model_en.pt"
        list_path = list_dir / f"SNR_{snr}_30_list.txt"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model for SNR {snr}: {model_path}")
        if not list_path.exists():
            raise FileNotFoundError(f"Missing test list for SNR {snr}: {list_path}")

        model_snr = extract_model_snr(model_path)
        list_snr = extract_list_snr(list_path)
        if model_snr != snr or list_snr != snr or model_snr != list_snr:
            raise ValueError(
                "SNR mismatch: "
                f"expected={snr}, model={model_snr}, test_list={list_snr}"
            )
        pairs.append(SnrPair(snr=snr, model_path=model_path, list_path=list_path))

    return pairs


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def read_and_validate_list(path, class_ids):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No samples found in: {path}")

    labels = []
    for line_number, line in enumerate(lines, start=1):
        parts = line.rsplit(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid sample line {line_number} in {path}: {line}")
        try:
            labels.append(int(parts[1]))
        except ValueError as exc:
            raise ValueError(
                f"Invalid label on line {line_number} in {path}: {parts[1]}"
            ) from exc

    label_counts = Counter(labels)
    expected_counts = {class_id: EXPECTED_SAMPLES_PER_CLASS for class_id in class_ids}
    if set(label_counts) != set(class_ids):
        raise ValueError(
            f"Unexpected labels in {path}. Expected {class_ids}, "
            f"found {sorted(label_counts)}."
        )
    if dict(label_counts) != expected_counts:
        counts_in_order = {class_id: label_counts[class_id] for class_id in class_ids}
        raise ValueError(
            f"{path} must contain {EXPECTED_SAMPLES_PER_CLASS} samples per class. "
            f"Found: {counts_in_order}"
        )

    expected_total = EXPECTED_SAMPLES_PER_CLASS * len(class_ids)
    if len(lines) != expected_total:
        raise ValueError(
            f"{path} must contain {expected_total} samples, but found {len(lines)}."
        )

    return [line + "\n" for line in lines], labels, label_counts


def make_loader(list_path, class_ids, args, device):
    lines, labels, label_counts = read_and_validate_list(list_path, class_ids)
    dataset = data_list.ImageList(lines, transform=image_test())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    return loader, labels, label_counts


def normalize_state_dict(raw_state):
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        raw_state = raw_state["state_dict"]
    if not isinstance(raw_state, dict):
        raise ValueError("Checkpoint did not contain a PyTorch state_dict.")

    if any(key.startswith("module.") for key in raw_state):
        return {
            key.removeprefix("module."): value
            for key, value in raw_state.items()
        }
    return raw_state


def checkpoint_class_num(state_dict, checkpoint_path):
    if "fc.weight" not in state_dict:
        raise ValueError(f"{checkpoint_path} has no fc.weight entry.")
    return int(state_dict["fc.weight"].shape[0])


def create_model(checkpoint_path, device):
    state_dict = normalize_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    class_num = checkpoint_class_num(state_dict, checkpoint_path)
    model = en_network.RepVGG_B1g2(
        use_bottleneck=True,
        bottleneck_dim=BOTTLENECK_DIM,
        new_cls=True,
        class_num=class_num,
        embedding_dim=EMBEDDING_DIM,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_num


def prediction_indices_to_class_ids(prediction_indices, class_ids, class_num):
    prediction_indices = np.asarray(prediction_indices, dtype=np.int64)
    if class_num == len(class_ids):
        class_ids_array = np.asarray(class_ids, dtype=np.int64)
        return class_ids_array[prediction_indices]
    if class_num <= max(class_ids):
        raise ValueError(
            f"Cannot map classifier outputs to raw class ids: class_num={class_num}, "
            f"max CLASS_IDS_30={max(class_ids)}."
        )
    return prediction_indices


def collect_predictions(model, loader, device, class_ids, class_num):
    true_batches = []
    pred_batches = []
    total_samples = 0
    inference_seconds = 0.0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_start = time.perf_counter()
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_start = time.perf_counter()
            _, logits, _, _ = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - inference_start
            prediction_indices = logits.argmax(dim=1).cpu().numpy()
            prediction_class_ids = prediction_indices_to_class_ids(
                prediction_indices, class_ids, class_num
            )
            true_batches.append(labels.cpu().numpy().astype(np.int64))
            pred_batches.append(prediction_class_ids.astype(np.int64))
            total_samples += int(labels.shape[0])

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - total_start
    timing = {
        "total_samples": total_samples,
        "total_seconds": total_seconds,
        "seconds_per_sample": total_seconds / total_samples if total_samples else 0.0,
        "inference_seconds": inference_seconds,
        "inference_seconds_per_sample": (
            inference_seconds / total_samples if total_samples else 0.0
        ),
    }

    return np.concatenate(true_batches), np.concatenate(pred_batches), timing


def build_confusion_data(y_true, y_pred, class_ids):
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    counts = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    true_totals = Counter(int(label) for label in y_true)
    outside_by_true = Counter()
    outside_predicted_labels = Counter()
    correct = 0

    for true_label, pred_label in zip(y_true, y_pred):
        true_label = int(true_label)
        pred_label = int(pred_label)
        if true_label not in class_to_index:
            raise ValueError(f"Unexpected true label after validation: {true_label}")
        if pred_label == true_label:
            correct += 1
        if pred_label in class_to_index:
            counts[class_to_index[true_label], class_to_index[pred_label]] += 1
        else:
            outside_by_true[true_label] += 1
            outside_predicted_labels[pred_label] += 1

    denominators = np.array([true_totals[class_id] for class_id in class_ids])
    normalized = np.divide(
        counts,
        denominators[:, None],
        out=np.zeros_like(counts, dtype=np.float64),
        where=denominators[:, None] != 0,
    )
    total = len(y_true)
    outside_total = sum(outside_by_true.values())
    accuracy = correct / total if total else 0.0

    return {
        "counts": counts,
        "normalized": normalized,
        "true_totals": true_totals,
        "outside_by_true": outside_by_true,
        "outside_predicted_labels": outside_predicted_labels,
        "outside_total": outside_total,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }


def write_class_mapping(output_dir, class_ids):
    output_dir.mkdir(parents=True, exist_ok=True)
    names = display_names(class_ids)
    with (output_dir / "class_mapping.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["display_name", "class_id"])
        writer.writerows(zip(names, class_ids))


def write_matrix_csv(path, matrix, class_ids, is_float):
    names = display_names(class_ids)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class", "true_class_id", *names])
        for row_index, class_id in enumerate(class_ids):
            if is_float:
                row_values = [f"{value:.8f}" for value in matrix[row_index]]
            else:
                row_values = [int(value) for value in matrix[row_index]]
            writer.writerow([names[row_index], class_id, *row_values])


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_confusion_figure(
    path_png,
    path_pdf,
    normalized,
    snr,
    accuracy,
    outside_total,
    class_ids,
    no_pdf,
):
    names = display_names(class_ids)
    fig, ax = plt.subplots(figsize=(12.5, 10.5))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    # ax.set_title(
    #     f"{snr} SNR"
    # )
    ax.set_xlabel("Predicted Classes", fontsize=20)
    ax.set_ylabel("Actual Classes", fontsize=20)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=70, fontsize=15)
    ax.set_yticklabels(names, fontsize=15)
    ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    annotate_cells(ax, normalized, fontsize=10)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=20)
    # cbar.set_label("Recall-normalized count")
    fig.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    if not no_pdf:
        fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def annotate_cells(ax, matrix, fontsize):
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            text_color = "white" if value >= 0.5 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value * 100:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=fontsize,
            )


def save_overview(output_dir, results, class_ids, no_pdf):
    names = display_names(class_ids)
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.5), constrained_layout=True)
    axes = axes.ravel()
    last_image = None

    for ax, result in zip(axes, results):
        last_image = ax.imshow(
            result["normalized"],
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(
            f"SNR {result['snr']} dB\n"
            f"Acc {result['accuracy']:.2%}, Outside {result['outside_total']}"
        )
        ax.set_xlabel("Predicted", fontsize=15)
        ax.set_ylabel("True", fontsize=15)
        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        annotate_cells(ax, result["normalized"], fontsize=5)

    if last_image is not None:
        fig.colorbar(last_image, ax=axes.tolist(), shrink=0.75, label="Recall-normalized count")
    fig.savefig(output_dir / "overview_2x3.png", dpi=300, bbox_inches="tight")
    if not no_pdf:
        fig.savefig(output_dir / "overview_2x3.pdf", bbox_inches="tight")
    plt.close(fig)


def metrics_payload(
    pair,
    class_ids,
    class_num,
    label_counts,
    confusion,
    timing,
    run_timestamp,
):
    names = display_names(class_ids)
    outside_by_true = {
        names[index]: {
            "class_id": class_id,
            "outside_target_predictions": int(confusion["outside_by_true"][class_id]),
        }
        for index, class_id in enumerate(class_ids)
    }
    return {
        "run_timestamp": run_timestamp,
        "snr_db": pair.snr,
        "model_path": str(pair.model_path.resolve()),
        "test_list_path": str(pair.list_path.resolve()),
        "class_num": class_num,
        "class_mapping": {
            name: class_id for name, class_id in zip(names, class_ids)
        },
        "samples": int(confusion["total"]),
        "timing": {
            "total_seconds": timing["total_seconds"],
            "seconds_per_sample": timing["seconds_per_sample"],
            "milliseconds_per_sample": timing["seconds_per_sample"] * 1000,
            "inference_seconds": timing["inference_seconds"],
            "inference_seconds_per_sample": timing["inference_seconds_per_sample"],
            "inference_milliseconds_per_sample": (
                timing["inference_seconds_per_sample"] * 1000
            ),
        },
        "samples_per_class": {
            str(class_id): int(label_counts[class_id]) for class_id in class_ids
        },
        "correct": int(confusion["correct"]),
        "accuracy": confusion["accuracy"],
        "outside_target_predictions": int(confusion["outside_total"]),
        "outside_target_rate": (
            confusion["outside_total"] / confusion["total"]
            if confusion["total"]
            else 0.0
        ),
        "outside_by_true_class": outside_by_true,
        "outside_predicted_labels": {
            str(label): int(count)
            for label, count in sorted(confusion["outside_predicted_labels"].items())
        },
    }


def write_summary(output_dir, rows):
    fieldnames = [
        "run_timestamp",
        "snr_db",
        "accuracy",
        "correct",
        "samples",
        "total_seconds",
        "milliseconds_per_sample",
        "inference_seconds",
        "inference_milliseconds_per_sample",
        "outside_target_predictions",
        "outside_target_rate",
        "class_num",
        "samples_per_class_min",
        "samples_per_class_max",
        "model_path",
        "test_list_path",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "summary.json", rows)


def process_pair(pair, class_ids, args, device):
    print(f"\nRunning SNR {pair.snr} dB")
    print(f"  model: {pair.model_path}")
    print(f"  test : {pair.list_path}")

    loader, _, label_counts = make_loader(pair.list_path, class_ids, args, device)
    model, class_num = create_model(pair.model_path, device)
    y_true, y_pred, timing = collect_predictions(
        model, loader, device, class_ids, class_num
    )
    confusion = build_confusion_data(y_true, y_pred, class_ids)

    stem = f"SNR_{pair.snr}"
    write_matrix_csv(
        args.output_dir / f"{stem}_confusion_matrix_counts.csv",
        confusion["counts"],
        class_ids,
        is_float=False,
    )
    write_matrix_csv(
        args.output_dir / f"{stem}_confusion_matrix_normalized.csv",
        confusion["normalized"],
        class_ids,
        is_float=True,
    )

    metrics = metrics_payload(
        pair,
        class_ids,
        class_num,
        label_counts,
        confusion,
        timing,
        args.run_timestamp,
    )
    write_json(
        args.output_dir / f"{stem}_metrics.json",
        metrics,
    )
    save_confusion_figure(
        args.output_dir / f"{stem}_confusion_matrix.png",
        args.output_dir / f"{stem}_confusion_matrix.pdf",
        confusion["normalized"],
        pair.snr,
        confusion["accuracy"],
        confusion["outside_total"],
        class_ids,
        args.no_pdf,
    )

    print(
        f"  samples={confusion['total']}, class_num={class_num}, "
        f"accuracy={confusion['accuracy']:.4%}, "
        f"outside={confusion['outside_total']}"
    )
    print(
        "  timing: "
        f"total={timing['total_seconds']:.4f}s, "
        f"per_sample={timing['seconds_per_sample'] * 1000:.4f} ms, "
        f"forward_per_sample={timing['inference_seconds_per_sample'] * 1000:.4f} ms"
    )

    summary_counts = list(label_counts.values())
    summary_row = {
        "run_timestamp": args.run_timestamp,
        "snr_db": pair.snr,
        "accuracy": f"{confusion['accuracy']:.8f}",
        "correct": int(confusion["correct"]),
        "samples": int(confusion["total"]),
        "total_seconds": f"{timing['total_seconds']:.8f}",
        "milliseconds_per_sample": f"{timing['seconds_per_sample'] * 1000:.8f}",
        "inference_seconds": f"{timing['inference_seconds']:.8f}",
        "inference_milliseconds_per_sample": (
            f"{timing['inference_seconds_per_sample'] * 1000:.8f}"
        ),
        "outside_target_predictions": int(confusion["outside_total"]),
        "outside_target_rate": f"{metrics['outside_target_rate']:.8f}",
        "class_num": class_num,
        "samples_per_class_min": int(min(summary_counts)),
        "samples_per_class_max": int(max(summary_counts)),
        "model_path": str(pair.model_path.resolve()),
        "test_list_path": str(pair.list_path.resolve()),
    }
    overview_result = {
        "snr": pair.snr,
        "normalized": confusion["normalized"],
        "accuracy": confusion["accuracy"],
        "outside_total": confusion["outside_total"],
    }
    return summary_row, overview_result


def main():
    args = parse_args()
    args.run_timestamp = make_run_timestamp()
    base_output_dir = args.output_dir.resolve()
    args.output_dir = base_output_dir / args.run_timestamp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    class_ids = load_class_ids(PROJECT_ROOT / ".env")
    write_class_mapping(args.output_dir, class_ids)
    pairs = build_pairs()
    device = resolve_device(args.device)

    print("Creating confusion matrices with strict SNR pairing.")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output base : {base_output_dir}")
    print(f"Run dir     : {args.output_dir}")
    print(f"Timestamp   : {args.run_timestamp}")
    print(f"Device      : {device}")
    print(f"{CLASS_ENV_NAME}: {class_ids}")
    print("Matched pairs:")
    for pair in pairs:
        print(f"  SNR {pair.snr:>2} dB -> {pair.model_path.name} | {pair.list_path.name}")

    summary_rows = []
    overview_results = []
    for pair in pairs:
        summary_row, overview_result = process_pair(pair, class_ids, args, device)
        summary_rows.append(summary_row)
        overview_results.append(overview_result)

    save_overview(args.output_dir, overview_results, class_ids, args.no_pdf)
    write_summary(args.output_dir, summary_rows)
    print(f"\nDone. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
