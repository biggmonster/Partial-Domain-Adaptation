import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

import create_confusion_matrix as cm


BOX_COLORS = [
    "#f5c28b",
    "#9ed19f",
    "#c8b9dd",
    "#f0e870",
    "#8fb9e6",
    "#d944d2",
]
OUTSIDE_OUTLIER_RATIO = 0.05 / 1.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create class-level accuracy statistics plot for matched SNR models."
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
        / "statistics_plot_results"
        / "RepVGG_B1g2"
        / "TD_30class",
    )
    parser.add_argument(
        "--outside-outlier-ratio",
        type=float,
        default=OUTSIDE_OUTLIER_RATIO,
        help="Orange point threshold: outside predictions / class samples. Default: 1/3.",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Do not write PDF files.")
    return parser.parse_args()


def compute_per_class_accuracy(
    pair, class_ids, class_num, y_true, y_pred, run_timestamp, outside_outlier_ratio
):
    target_class_set = set(class_ids)
    names = cm.display_names(class_ids)
    rows = []

    for display_name, class_id in zip(names, class_ids):
        mask = y_true == class_id
        class_true = y_true[mask]
        class_pred = y_pred[mask]
        total = int(class_true.size)
        correct = int(np.sum(class_pred == class_true))
        inside = int(np.isin(class_pred, list(target_class_set)).sum())
        outside = int(total - inside)
        outside_ratio = (outside / total) if total else 0.0
        is_outlier = outside_ratio >= outside_outlier_ratio
        accuracy = (correct / total * 100.0) if total else 0.0

        rows.append(
            {
                "run_timestamp": run_timestamp,
                "snr_db": pair.snr,
                "display_name": display_name,
                "class_id": class_id,
                "accuracy_percent": accuracy,
                "samples": total,
                "correct": correct,
                "inside_target_predictions": inside,
                "outside_target_predictions": outside,
                "outside_target_ratio": outside_ratio,
                "outside_outlier_ratio": outside_outlier_ratio,
                "point_type": "outlier" if is_outlier else "data",
                "class_num": class_num,
                "model_path": str(pair.model_path.resolve()),
                "test_list_path": str(pair.list_path.resolve()),
            }
        )

    return rows


def summarize_snr(pair, rows, run_timestamp):
    accuracies = np.array([row["accuracy_percent"] for row in rows], dtype=np.float64)
    samples = sum(row["samples"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    outside = sum(row["outside_target_predictions"] for row in rows)
    outside_classes = sum(1 for row in rows if row["outside_target_predictions"] > 0)
    outlier_classes = sum(1 for row in rows if row["point_type"] == "outlier")
    return {
        "run_timestamp": run_timestamp,
        "snr_db": pair.snr,
        "class_count": len(rows),
        "samples": samples,
        "correct": correct,
        "overall_accuracy_percent": correct / samples * 100.0 if samples else 0.0,
        "class_accuracy_mean": float(np.mean(accuracies)),
        "class_accuracy_q1": float(np.percentile(accuracies, 25)),
        "class_accuracy_q3": float(np.percentile(accuracies, 75)),
        "class_accuracy_min": float(np.min(accuracies)),
        "class_accuracy_max": float(np.max(accuracies)),
        "outside_target_predictions": outside,
        "outside_target_classes": outside_classes,
        "outlier_classes": outlier_classes,
        "outside_outlier_ratio": rows[0]["outside_outlier_ratio"] if rows else None,
        "class_num": rows[0]["class_num"] if rows else None,
        "model_path": str(pair.model_path.resolve()),
        "test_list_path": str(pair.list_path.resolve()),
    }


def format_float(value):
    if isinstance(value, float):
        return f"{value:.8f}"
    return value


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_float(row.get(key, "")) for key in fieldnames})


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def draw_box(ax, position, stats, color):
    box_width = 0.16
    q1 = stats["class_accuracy_q1"]
    q3 = stats["class_accuracy_q3"]
    mean = stats["class_accuracy_mean"]
    minimum = stats["class_accuracy_min"]
    maximum = stats["class_accuracy_max"]

    ax.vlines(position, minimum, maximum, color="#777777", linewidth=1.0, zorder=1)
    ax.hlines([minimum, maximum], position - box_width * 0.35, position + box_width * 0.35,
              color="#777777", linewidth=1.0, zorder=1)
    rect = Rectangle(
        (position - box_width / 2.0, q1),
        box_width,
        q3 - q1,
        facecolor=color,
        edgecolor="#777777",
        linewidth=1.0,
        alpha=0.75,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.hlines(mean, position - box_width * 0.65, position + box_width * 0.65,
              color="#222222", linewidth=1.8, zorder=3)


def draw_accuracy_plot(output_dir, per_class_rows, summary_rows, no_pdf):
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    positions = np.arange(1, len(cm.SNR_VALUES) + 1, dtype=np.float64)
    all_accuracies = []

    rows_by_snr = {
        snr: [row for row in per_class_rows if row["snr_db"] == snr]
        for snr in cm.SNR_VALUES
    }
    summary_by_snr = {row["snr_db"]: row for row in summary_rows}

    for index, snr in enumerate(cm.SNR_VALUES):
        position = positions[index]
        rows = rows_by_snr[snr]
        stats = summary_by_snr[snr]
        draw_box(ax, position, stats, BOX_COLORS[index])

        point_x_base = position + 0.36
        for row in rows:
            accuracy = row["accuracy_percent"]
            all_accuracies.append(accuracy)
            is_outlier = row["point_type"] == "outlier"
            ax.scatter(
                point_x_base,
                accuracy,
                marker="D",
                s=24,
                color="#ff7f0e" if is_outlier else "#3d3d3d",
                edgecolors="none",
                zorder=4,
            )

    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_xlim(0.5, len(cm.SNR_VALUES) + 0.75)
    ax.set_xticks(positions)
    ax.set_xticklabels(cm.SNR_VALUES, fontsize=14, fontweight="bold")
    ax.set_xlabel("SNR(dB)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Accuracy(%)", fontsize=18, fontweight="bold")
    ax.tick_params(axis="y", labelsize=13, width=1.0)
    ax.tick_params(axis="x", width=1.0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            color="#3d3d3d",
            markersize=6,
            label="Data points",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            color="#ff7f0e",
            markersize=6,
            label="Outliers",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fancybox=False,
              edgecolor="#222222", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_statistics_plot.png", dpi=300, bbox_inches="tight")
    if not no_pdf:
        fig.savefig(output_dir / "accuracy_statistics_plot.pdf", bbox_inches="tight")
    plt.close(fig)


def run_pair(pair, class_ids, args, device):
    print(f"\nRunning SNR {pair.snr} dB")
    print(f"  model: {pair.model_path}")
    print(f"  test : {pair.list_path}")
    loader, _, label_counts = cm.make_loader(pair.list_path, class_ids, args, device)
    model, class_num = cm.create_model(pair.model_path, device)
    y_true, y_pred = cm.collect_predictions(model, loader, device, class_ids, class_num)
    rows = compute_per_class_accuracy(
        pair,
        class_ids,
        class_num,
        y_true,
        y_pred,
        args.run_timestamp,
        args.outside_outlier_ratio,
    )
    summary = summarize_snr(pair, rows, args.run_timestamp)
    print(
        f"  samples={summary['samples']}, class_num={class_num}, "
        f"overall_accuracy={summary['overall_accuracy_percent']:.4f}%, "
        f"class_mean={summary['class_accuracy_mean']:.4f}%, "
        f"outside_classes={summary['outside_target_classes']}, "
        f"outlier_classes={summary['outlier_classes']}"
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, summary


def main():
    args = parse_args()
    args.run_timestamp = cm.make_run_timestamp()
    base_output_dir = args.output_dir.resolve()
    args.output_dir = base_output_dir / args.run_timestamp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    class_ids = cm.load_class_ids(PROJECT_ROOT / ".env")
    cm.write_class_mapping(args.output_dir, class_ids)
    pairs = cm.build_pairs()
    device = cm.resolve_device(args.device)

    print("Creating class-level accuracy statistics plot with strict SNR pairing.")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output base : {base_output_dir}")
    print(f"Run dir     : {args.output_dir}")
    print(f"Timestamp   : {args.run_timestamp}")
    print(f"Device      : {device}")
    print(f"{cm.CLASS_ENV_NAME}: {class_ids}")
    print("Matched pairs:")
    for pair in pairs:
        print(f"  SNR {pair.snr:>2} dB -> {pair.model_path.name} | {pair.list_path.name}")

    per_class_rows = []
    summary_rows = []
    for pair in pairs:
        rows, summary = run_pair(pair, class_ids, args, device)
        per_class_rows.extend(rows)
        summary_rows.append(summary)

    per_class_fields = [
        "run_timestamp",
        "snr_db",
        "display_name",
        "class_id",
        "accuracy_percent",
        "samples",
        "correct",
        "inside_target_predictions",
        "outside_target_predictions",
        "outside_target_ratio",
        "outside_outlier_ratio",
        "point_type",
        "class_num",
        "model_path",
        "test_list_path",
    ]
    summary_fields = [
        "run_timestamp",
        "snr_db",
        "class_count",
        "samples",
        "correct",
        "overall_accuracy_percent",
        "class_accuracy_mean",
        "class_accuracy_q1",
        "class_accuracy_q3",
        "class_accuracy_min",
        "class_accuracy_max",
        "outside_target_predictions",
        "outside_target_classes",
        "outlier_classes",
        "outside_outlier_ratio",
        "class_num",
        "model_path",
        "test_list_path",
    ]

    write_csv(args.output_dir / "per_class_accuracy.csv", per_class_rows, per_class_fields)
    write_csv(args.output_dir / "summary.csv", summary_rows, summary_fields)
    write_json(args.output_dir / "summary.json", summary_rows)
    draw_accuracy_plot(args.output_dir, per_class_rows, summary_rows, args.no_pdf)

    expected_rows = len(cm.SNR_VALUES) * len(class_ids)
    if len(per_class_rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} per-class rows, found {len(per_class_rows)}."
        )
    print(f"\nDone. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
