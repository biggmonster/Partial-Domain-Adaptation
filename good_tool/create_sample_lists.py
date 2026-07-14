import argparse
import json
import random
from pathlib import Path
from dotenv import load_dotenv
import os

# Load the shared class selection from the project-level .env file.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def load_class_ids():
    raw_class_ids = os.getenv("CLASS_IDS_15")
    if raw_class_ids is None:
        raise ValueError("CLASS_IDS is missing from the project-level .env file.")
    try:
        class_ids = json.loads(raw_class_ids)
    except json.JSONDecodeError as exc:
        raise ValueError("CLASS_IDS must be a JSON list such as [19, 27, 38].") from exc
    if (
        not isinstance(class_ids, list)
        or not class_ids
        or any(not isinstance(class_id, int) for class_id in class_ids)
        or len(set(class_ids)) != len(class_ids)
        or any(class_id < 0 or class_id >= 50 for class_id in class_ids)
    ):
        raise ValueError("CLASS_IDS must contain unique integer labels in [0, 49].")
    return class_ids


CLASS_IDS_15 = load_class_ids()
CLASS_NAMES = {class_id: f"C{index}" for index, class_id in enumerate(CLASS_IDS_15, 1)}
TARGET_SNRS = ["SNR_0"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


'''
为 t-SNE 可视化固定抽取一批原始图片样本，并生成图片路径列表。
'''


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create reproducible t-SNE sample lists."
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            r"D:\LJ\workstation\Matlab\ads-b_data_gen\data_LongSig_50\source\gasf_old"
        ),
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path(
            r"D:\LJ\workstation\Matlab\ads-b_data_gen\data_LongSig_50\target\gasf_old"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "LongSig_50" / "t-SNE" / "class_15",
    )
    return parser.parse_args()


def image_files(class_dir):
    if not class_dir.is_dir():
        raise FileNotFoundError(f"Class directory does not exist: {class_dir}")
    return sorted(
        (path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.lower(),
    )


def select_samples(root_dir, snr, samples_per_class, rng):
    records = []
    counts = {}
    for class_id in CLASS_IDS_15:
        class_dir = root_dir / snr / f"class_{(class_id + 1):02d}"
        candidates = image_files(class_dir)
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"{class_dir} contains {len(candidates)} images; "
                f"{samples_per_class} are required."
            )
        selected = sorted(
            rng.sample(candidates, samples_per_class),
            key=lambda path: path.name.lower(),
        )
        records.extend({"path": str(path), "label": class_id} for path in selected)
        counts[str(class_id)] = len(selected)
    return records, counts


def write_list(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{record['path']} {record['label']}\n" for record in records]
    path.write_text("".join(lines), encoding="utf-8")


def add_manifest_entry(manifest, key, list_path, records, counts):
    manifest[key] = {
        "list_path": str(list_path),
        "sample_count": len(records),
        "counts_by_label": counts,
        "samples": records,
    }


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = args.output_dir.resolve()
    manifest = {
        "seed": args.seed,
        "samples_per_class": args.samples_per_class,
        "class_ids": CLASS_IDS_15,
        "class_mapping": {CLASS_NAMES[class_id]: class_id for class_id in CLASS_IDS_15},
        "source_snr": "SNR_20",
        "target_snrs": TARGET_SNRS,
        "lists": {},
    }

    source_records, source_counts = select_samples(
        args.source_root.resolve(), "SNR_20", args.samples_per_class, rng
    )
    source_path = output_dir / "source" / "SNR_20_200_list.txt"
    write_list(source_path, source_records)
    add_manifest_entry(
        manifest["lists"], "source/SNR_20", source_path, source_records, source_counts
    )

    for snr in TARGET_SNRS:
        target_records, target_counts = select_samples(
            args.target_root.resolve(), snr, args.samples_per_class, rng
        )
        target_path = output_dir / "target" / f"{snr}_200_list.txt"
        write_list(target_path, target_records)
        add_manifest_entry(
            manifest["lists"], f"target/{snr}", target_path, target_records, target_counts
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Created reproducible t-SNE lists in: {output_dir}")
    print(f"Class mapping: {manifest['class_mapping']}")
    print(f"Each list contains {len(source_records)} samples.")


if __name__ == "__main__":
    main()
