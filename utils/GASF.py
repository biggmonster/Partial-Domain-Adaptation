"""Signal augmentation and Gramian Angular Summation Field generation.

This module is the Python counterpart of ``GASF_gen.py`` (the retained
MATLAB batch script) and ``GASF_gene.m``.  It serves two use cases:

* random-access conversion used by the source-domain training dataset;
* an optional command-line batch converter that writes MATLAB-compatible
  ``class_XX/CXX_SXX.png`` files.

MATLAB v7.3 files are HDF5 containers.  MATLAB reverses the displayed array
dimensions in HDF5, so an ``X`` saved as ``[N, L, 2]`` is exposed by h5py as
``[2, L, N]``.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import h5py
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from tqdm import tqdm


PathLike = Union[str, os.PathLike]
_SAMPLE_PATTERN = re.compile(r"C(?P<class_id>\d+)_S(?P<sample_id>\d+)\.[^.]+$", re.IGNORECASE)


def parse_sample_path(path: PathLike) -> Tuple[int, int]:
    """Return the one-based ``(class_id, sample_id)`` encoded in a filename."""

    match = _SAMPLE_PATTERN.search(Path(path).name)
    if match is None:
        raise ValueError(f"Cannot parse class/sample id from path: {path}")
    return int(match.group("class_id")), int(match.group("sample_id"))


def matlab_sample_index(class_id: int, sample_id: int, samples_per_class: int = 1000) -> int:
    """Map MATLAB's one-based class/sample ids to a zero-based flat index."""

    if class_id < 1 or sample_id < 1:
        raise ValueError("class_id and sample_id must be one-based positive integers")
    if sample_id > samples_per_class:
        raise ValueError(
            f"sample_id={sample_id} exceeds samples_per_class={samples_per_class}"
        )
    return (class_id - 1) * samples_per_class + (sample_id - 1)


def smooth_downsample(raw_signal: np.ndarray, target_length: int) -> np.ndarray:
    """Reproduce ``smoothDownsample`` from ``GASF_gene.m``.

    The first ``target_length - 1`` bins contain ``floor(N/target_length)``
    samples.  The final bin receives every remaining sample, exactly matching
    the current MATLAB implementation.
    """

    signal = np.asarray(raw_signal)
    if signal.ndim != 1:
        raise ValueError(f"raw_signal must be one-dimensional, got {signal.shape}")
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if signal.size < target_length:
        raise ValueError(
            f"signal length {signal.size} is smaller than target_length {target_length}"
        )

    step = signal.size // target_length
    output_dtype = np.result_type(signal.dtype, np.float64)
    downsampled = np.empty(target_length, dtype=output_dtype)
    for index in range(target_length - 1):
        start = index * step
        downsampled[index] = np.mean(signal[start : start + step])
    downsampled[-1] = np.mean(signal[(target_length - 1) * step :])
    return downsampled


def normalize_data(signal: np.ndarray) -> np.ndarray:
    """Normalize a one-dimensional signal to ``[-1, 1]`` like MATLAB."""

    values = np.asarray(signal)
    if values.ndim != 1:
        raise ValueError(f"signal must be one-dimensional, got {values.shape}")
    minimum = np.min(values)
    maximum = np.max(values)
    if maximum == minimum:
        return np.zeros(values.shape, dtype=np.result_type(values.dtype, np.float64))
    return 2.0 * (values - minimum) / (maximum - minimum) - 1.0


def generate_gasf(normalized_signal: np.ndarray) -> np.ndarray:
    """Build a Gramian Angular Summation Field and map it to ``[0, 1]``."""

    values = np.asarray(normalized_signal)
    if values.ndim != 1:
        raise ValueError(f"normalized_signal must be one-dimensional, got {values.shape}")
    values = np.clip(values, -1.0, 1.0)
    theta = np.arccos(values)
    gasf = np.cos(theta[:, None] + theta[None, :])
    return (gasf + 1.0) / 2.0


def _split_iq(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signal)
    if np.iscomplexobj(values):
        if values.ndim != 1:
            raise ValueError(f"complex signal must be one-dimensional, got {values.shape}")
        return values.real, values.imag
    if values.ndim != 2:
        raise ValueError(
            "real-valued signal must have shape (2, length) or (length, 2), "
            f"got {values.shape}"
        )
    if values.shape[0] == 2:
        return values[0], values[1]
    if values.shape[1] == 2:
        return values[:, 0], values[:, 1]
    raise ValueError(f"cannot identify I/Q axes in shape {values.shape}")


def iq_to_gasf(signal: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Convert a complex or two-channel I/Q signal to a ``N x 2N`` GASF."""

    i_values, q_values = _split_iq(signal)
    i_smooth = smooth_downsample(i_values, target_size)
    q_smooth = smooth_downsample(q_values, target_size)
    i_gasf = generate_gasf(normalize_data(i_smooth))
    q_gasf = generate_gasf(normalize_data(q_smooth))
    return np.concatenate((i_gasf, q_gasf), axis=1)


def _turbo_lut(size: int = 256) -> np.ndarray:
    # MATLAB imwrite rounds RGB doubles to the closest uint8 value.  All
    # values are non-negative, so floor(x + 0.5) gives MATLAB round semantics.
    colors = cm.get_cmap("turbo", size)(np.arange(size))[:, :3]
    return np.floor(colors * 255.0 + 0.5).astype(np.uint8)


TURBO_256 = _turbo_lut(256)


def gasf_to_rgb(gasf: np.ndarray, color_map: np.ndarray = TURBO_256) -> np.ndarray:
    """Apply MATLAB-style indexed Turbo coloring to a GASF matrix."""

    values = np.asarray(gasf)
    if values.ndim != 2:
        raise ValueError(f"gasf must be two-dimensional, got {values.shape}")
    lut = np.asarray(color_map)
    if lut.ndim != 2 or lut.shape[1] != 3:
        raise ValueError(f"color_map must have shape (N, 3), got {lut.shape}")
    if lut.dtype != np.uint8:
        lut = np.floor(np.clip(lut, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    indices = np.floor(np.clip(values, 0.0, 1.0) * (len(lut) - 1) + 0.5).astype(np.int64)
    return lut[indices]


def iq_to_rgb(signal: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Convert an I/Q signal directly to an RGB uint8 GASF image."""

    return gasf_to_rgb(iq_to_gasf(signal, target_size=target_size))


def iq_to_pil(signal: np.ndarray, target_size: int = 256) -> Image.Image:
    """Convert an I/Q signal to a PIL RGB image."""

    return Image.fromarray(iq_to_rgb(signal, target_size=target_size), mode="RGB")


def save_gasf_png(signal: np.ndarray, output_path: PathLike, target_size: int = 256) -> Path:
    """Generate a GASF image and save it as PNG."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    iq_to_pil(signal, target_size=target_size).save(destination, format="PNG")
    return destination


def add_measured_awgn(
    signal: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Add circular complex AWGN using the measured signal power."""

    values = np.asarray(signal, dtype=np.complex128)
    signal_power = float(np.mean(np.abs(values) ** 2))
    if not np.isfinite(signal_power) or signal_power <= 0.0:
        return values.copy()
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    scale = math.sqrt(noise_power / 2.0)
    noise = scale * (
        rng.standard_normal(values.shape) + 1j * rng.standard_normal(values.shape)
    )
    return values + noise


@dataclass(frozen=True)
class SignalAugmentConfig:
    """
    Configuration for identity-preserving source-signal augmentation.
    保存并检查所有信号增强参数

    snr_db: 高斯噪声
    gain_drift: 增益漂移±10%
    gain_drift_probability:  概率0.5
    phase_degrees_max: 相位旋转最大±10°
    phase_probability: 概率0.5
    crop_keep_min: 
    """

    snr_db_min: float = 20.0
    snr_db_max: float = 30.0
    gain_drift_max: float = 0.10
    gain_drift_probability: float = 0.50
    phase_degrees_max: float = 10.0
    phase_probability: float = 0.50
    crop_keep_min: float = 0.95
    crop_probability: float = 0.50
    time_shift_max_ratio: float = 0.01
    time_shift_probability: float = 0.30

    def __post_init__(self) -> None:
        if self.snr_db_min > self.snr_db_max:
            raise ValueError("snr_db_min must not exceed snr_db_max")
        if not 0.0 < self.crop_keep_min <= 1.0:
            raise ValueError("crop_keep_min must be in (0, 1]")
        for name in (
            "gain_drift_probability",
            "phase_probability",
            "crop_probability",
            "time_shift_probability",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.gain_drift_max < 0.0 or self.phase_degrees_max < 0.0:
            raise ValueError("augmentation magnitudes must be non-negative")
        if not 0.0 <= self.time_shift_max_ratio < 1.0:
            raise ValueError("time_shift_max_ratio must be in [0, 1)")

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def augment_iq(
    signal: np.ndarray,
    rng: np.random.Generator,
    config: SignalAugmentConfig = SignalAugmentConfig(),
) -> np.ndarray:
    """Create one random source-domain signal view."""

    values = np.asarray(signal, dtype=np.complex128).copy()
    if values.ndim != 1:
        raise ValueError(f"signal must be one-dimensional, got {values.shape}")

    if rng.random() < config.gain_drift_probability and config.gain_drift_max > 0.0:
        drift = rng.uniform(-config.gain_drift_max, config.gain_drift_max)
        envelope = 1.0 + drift * np.linspace(-1.0, 1.0, values.size)
        values *= envelope

    if rng.random() < config.phase_probability and config.phase_degrees_max > 0.0:
        phase = math.radians(
            rng.uniform(-config.phase_degrees_max, config.phase_degrees_max)
        )
        values *= np.exp(1j * phase)

    if rng.random() < config.time_shift_probability and config.time_shift_max_ratio > 0.0:
        maximum_shift = int(round(values.size * config.time_shift_max_ratio))
        if maximum_shift > 0:
            shift = int(rng.integers(-maximum_shift, maximum_shift + 1))
            shifted = np.zeros_like(values)
            if shift > 0:
                shifted[shift:] = values[:-shift]
            elif shift < 0:
                shifted[:shift] = values[-shift:]
            else:
                shifted = values
            values = shifted

    if rng.random() < config.crop_probability and config.crop_keep_min < 1.0:
        keep_ratio = rng.uniform(config.crop_keep_min, 1.0)
        keep_length = max(1, int(round(values.size * keep_ratio)))
        start_max = values.size - keep_length
        start = int(rng.integers(0, start_max + 1)) if start_max > 0 else 0
        values = values[start : start + keep_length]

    snr_db = rng.uniform(config.snr_db_min, config.snr_db_max)
    return add_measured_awgn(values, snr_db, rng)


class Mat73SignalReader:
    """Lazy, process-safe random reader for the MATLAB v7.3 signal file."""

    def __init__(
        self,
        mat_path: PathLike,
        samples_per_class: int = 1000,
        x_key: str = "X",
        y_key: str = "Y",
    ) -> None:
        self.mat_path = str(Path(mat_path))
        self.samples_per_class = int(samples_per_class)
        self.x_key = x_key
        self.y_key = y_key
        self._file: Optional[h5py.File] = None
        self._owner_pid: Optional[int] = None

    def _ensure_open(self) -> h5py.File:
        current_pid = os.getpid()
        if self._file is not None and self._owner_pid != current_pid:
            self.close()
        if self._file is None:
            self._file = h5py.File(self.mat_path, "r")
            self._owner_pid = current_pid
        return self._file

    @property
    def sample_count(self) -> int:
        return int(self._ensure_open()[self.x_key].shape[2])

    @property
    def signal_length(self) -> int:
        return int(self._ensure_open()[self.x_key].shape[1])

    @property
    def class_count(self) -> int:
        labels = self.labels_array()
        return int(labels.max()) + 1

    def labels_array(self) -> np.ndarray:
        """Return all labels as a small one-dimensional integer array."""

        return np.asarray(self._ensure_open()[self.y_key]).reshape(-1).astype(np.int64)

    def validate_layout(self) -> Tuple[int, int, int]:
        dataset = self._ensure_open()[self.x_key]
        if dataset.ndim != 3 or dataset.shape[0] != 2:
            raise ValueError(
                f"expected {self.x_key} shape (2, length, samples), got {dataset.shape}"
            )
        labels = self._ensure_open()[self.y_key]
        if labels.size != dataset.shape[2]:
            raise ValueError(
                f"{self.y_key} contains {labels.size} labels for {dataset.shape[2]} signals"
            )
        return tuple(int(value) for value in dataset.shape)

    def read_index(self, index: int) -> Tuple[np.ndarray, int]:
        file = self._ensure_open()
        dataset = file[self.x_key]
        if not 0 <= index < dataset.shape[2]:
            raise IndexError(f"signal index {index} is outside [0, {dataset.shape[2]})")
        iq = np.asarray(dataset[:, :, index], dtype=np.float64)
        label_dataset = file[self.y_key]
        if label_dataset.ndim == 2 and label_dataset.shape[0] == 1:
            label = int(label_dataset[0, index])
        else:
            label = int(np.asarray(label_dataset).reshape(-1)[index])
        return iq[0] + 1j * iq[1], label

    def read_sample(
        self,
        class_id: int,
        sample_id: int,
        expected_label: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        index = matlab_sample_index(class_id, sample_id, self.samples_per_class)
        signal, label = self.read_index(index)
        if expected_label is not None and label != int(expected_label):
            raise ValueError(
                f"label mismatch for C{class_id:02d}_S{sample_id:02d}: "
                f"list={expected_label}, mat={label}"
            )
        return signal, label

    def read_path(
        self, path: PathLike, expected_label: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        class_id, sample_id = parse_sample_path(path)
        return self.read_sample(class_id, sample_id, expected_label=expected_label)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._owner_pid = None

    def __enter__(self) -> "Mat73SignalReader":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __getstate__(self) -> Dict[str, object]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_owner_pid"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


_BATCH_READER: Optional[Mat73SignalReader] = None
_BATCH_CONFIG: Optional[Dict[str, object]] = None


def _init_batch_worker(mat_path: str, samples_per_class: int, config: Dict[str, object]) -> None:
    global _BATCH_READER, _BATCH_CONFIG
    _BATCH_READER = Mat73SignalReader(mat_path, samples_per_class=samples_per_class)
    _BATCH_CONFIG = config


def _batch_task(task: Tuple[int, int, str]) -> Tuple[str, str]:
    class_id, sample_id, output_path = task
    assert _BATCH_READER is not None and _BATCH_CONFIG is not None
    destination = Path(output_path)
    if destination.exists() and not bool(_BATCH_CONFIG["overwrite"]):
        return "skipped", str(destination)
    try:
        signal, _ = _BATCH_READER.read_sample(class_id, sample_id)
        if bool(_BATCH_CONFIG["add_noise"]):
            seed = int(_BATCH_CONFIG["seed"])
            rng = np.random.default_rng(np.random.SeedSequence([seed, class_id, sample_id]))
            signal = add_measured_awgn(signal, float(_BATCH_CONFIG["snr_db"]), rng)
        save_gasf_png(signal, destination, int(_BATCH_CONFIG["target_size"]))
        return "written", str(destination)
    except Exception as error:
        return "failed", f"{destination}: {error}"


def generate_gasf_dataset(
    mat_path: PathLike,
    output_dir: PathLike,
    class_ids: Iterable[int],
    sample_ids: Iterable[int],
    target_size: int = 256,
    add_noise: bool = True,
    snr_db: float = 0.0,
    workers: int = 8,
    seed: int = 2026,
    samples_per_class: int = 1000,
    overwrite: bool = False,
) -> Dict[str, object]:
    """Generate a directory tree of GASF PNGs from a MATLAB v7.3 file."""

    destination_root = Path(output_dir)
    classes = [int(value) for value in class_ids]
    samples = [int(value) for value in sample_ids]
    if not classes or not samples:
        raise ValueError("class_ids and sample_ids must not be empty")

    tasks = []
    for class_id in classes:
        class_dir = destination_root / f"class_{class_id:02d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in samples:
            output_path = class_dir / f"C{class_id:02d}_S{sample_id:02d}.png"
            tasks.append((class_id, sample_id, str(output_path)))

    config: Dict[str, object] = {
        "target_size": int(target_size),
        "add_noise": bool(add_noise),
        "snr_db": float(snr_db),
        "seed": int(seed),
        "overwrite": bool(overwrite),
    }
    counts = {"written": 0, "skipped": 0, "failed": 0}
    failures = []
    started = time.perf_counter()

    if workers <= 1:
        _init_batch_worker(str(mat_path), samples_per_class, config)
        results: Sequence[Tuple[str, str]] = [
            _batch_task(task)
            for task in tqdm(tasks, desc="Generating GASF", dynamic_ncols=True)
        ]
    else:
        chunksize = max(1, min(32, len(tasks) // (workers * 8) or 1))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_batch_worker,
            initargs=(str(mat_path), samples_per_class, config),
        ) as executor:
            mapped = executor.map(_batch_task, tasks, chunksize=chunksize)
            results = list(
                tqdm(mapped, total=len(tasks), desc="Generating GASF", dynamic_ncols=True)
            )

    for status, message in results:
        counts[status] += 1
        if status == "failed":
            failures.append(message)
    elapsed = time.perf_counter() - started
    return {
        **counts,
        "elapsed_seconds": elapsed,
        "failures": failures,
        "output_dir": str(destination_root),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate GASF images from MATLAB v7.3 I/Q signals"
    )
    parser.add_argument("--mat-path", required=True, help="MATLAB v7.3 input file")
    parser.add_argument("--output-dir", required=True, help="PNG output root")
    parser.add_argument("--class-start", type=int, default=1)
    parser.add_argument("--class-end", type=int, default=None)
    parser.add_argument("--sample-start", type=int, default=1)
    parser.add_argument("--sample-end", type=int, default=500)
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument("--add-noise", dest="add_noise", action="store_true")
    noise_group.add_argument("--no-add-noise", dest="add_noise", action="store_false")
    parser.set_defaults(add_noise=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    with Mat73SignalReader(
        args.mat_path, samples_per_class=args.samples_per_class
    ) as reader:
        reader.validate_layout()
        class_end = args.class_end if args.class_end is not None else reader.class_count

    summary = generate_gasf_dataset(
        mat_path=args.mat_path,
        output_dir=args.output_dir,
        class_ids=range(args.class_start, class_end + 1),
        sample_ids=range(args.sample_start, args.sample_end + 1),
        target_size=args.target_size,
        add_noise=args.add_noise,
        snr_db=args.snr_db,
        workers=args.workers,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
        overwrite=args.overwrite,
    )
    print(
        "GASF generation finished: "
        f"written={summary['written']}, skipped={summary['skipped']}, "
        f"failed={summary['failed']}, seconds={summary['elapsed_seconds']:.3f}"
    )
    for failure in summary["failures"]:
        print(f"ERROR: {failure}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
