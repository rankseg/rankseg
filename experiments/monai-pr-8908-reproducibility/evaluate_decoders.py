"""Evaluate argmax, RankDice, and RankIoU on identical cached probabilities."""

from __future__ import annotations

import argparse
import csv
import inspect
import io
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
from common import (
    DEFAULT_CONFIG,
    RESULT_COLUMNS,
    atomic_write_text,
    config_sha256,
    load_case,
    load_config,
    output_dir,
    resolve_device,
    seed_everything,
    software_info,
    validate_probability_payload,
)
from monai.transforms import AsDiscrete

METHODS = ("argmax", "rankdice", "rankiou")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", help="Override the configured decoding device")
    parser.add_argument("--case-limit", type=int, help="Only evaluate the first N configured cases")
    parser.add_argument("--warmup", type=int, help="Override timing warmup repetitions")
    parser.add_argument("--repetitions", type=int, help="Override timed repetitions")
    parser.add_argument("--overwrite", action="store_true", help="Discard existing per_case.csv")
    return parser.parse_args()


def validate_pr_transform() -> None:
    signature = inspect.signature(AsDiscrete)
    if "rankseg" not in signature.parameters:
        raise RuntimeError(
            "This MONAI does not implement PR #8908: AsDiscrete has no explicit rankseg parameter. "
            "Put the PR checkout first on PYTHONPATH."
        )
    probe = torch.tensor([[[0.6, 0.4]], [[0.4, 0.6]]], dtype=torch.float32)
    output = AsDiscrete(rankseg=True, metric="dice")(probe)
    if tuple(output.shape) != (1, 1, 2):
        raise RuntimeError(f"Unexpected RankSEG probe shape {tuple(output.shape)}")


def build_decoders() -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    return {
        "argmax": AsDiscrete(argmax=True, keepdim=True),
        "rankdice": AsDiscrete(rankseg=True, metric="dice", keepdim=True),
        "rankiou": AsDiscrete(rankseg=True, metric="iou", keepdim=True),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def decode_once(decoder: Callable[[torch.Tensor], torch.Tensor], probabilities: torch.Tensor) -> torch.Tensor:
    prediction = decoder(probabilities)
    if prediction.shape[0] != 1:
        raise ValueError(f"Decoder did not preserve a singleton class dimension: {tuple(prediction.shape)}")
    return prediction.squeeze(0).to(torch.uint8)


@torch.inference_mode()
def time_decoder(
    decoder: Callable[[torch.Tensor], torch.Tensor],
    probabilities: torch.Tensor,
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> tuple[float, list[float]]:
    for _ in range(warmup):
        prediction = decoder(probabilities)
        synchronize(device)
        del prediction

    timings = []
    for _ in range(repetitions):
        synchronize(device)
        started = time.perf_counter()
        prediction = decoder(probabilities)
        synchronize(device)
        timings.append((time.perf_counter() - started) * 1000.0)
        del prediction
    return statistics.median(timings), timings


def binary_overlap(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    prediction = prediction.bool()
    target = target.bool()
    intersection = int(torch.count_nonzero(prediction & target))
    prediction_size = int(torch.count_nonzero(prediction))
    target_size = int(torch.count_nonzero(target))
    dice_denominator = prediction_size + target_size
    union = prediction_size + target_size - intersection
    # Explicit policy: an absent class predicted as absent is a perfect result;
    # an absent class predicted as present gets zero through the formulas below.
    dice = 1.0 if dice_denominator == 0 else 2.0 * intersection / dice_denominator
    iou = 1.0 if union == 0 else intersection / union
    return dice, iou


def case_metrics(
    prediction: torch.Tensor, label: torch.Tensor, foreground_classes: list[int]
) -> tuple[float, float, dict[int, tuple[float, float]]]:
    per_class = {class_id: binary_overlap(prediction == class_id, label == class_id) for class_id in foreground_classes}
    dice = sum(values[0] for values in per_class.values()) / len(per_class)
    iou = sum(values[1] for values in per_class.values()) / len(per_class)
    return dice, iou, per_class


def load_existing(path: Path, overwrite: bool) -> dict[str, dict[str, str]]:
    if overwrite or not path.exists():
        return {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != RESULT_COLUMNS:
            raise ValueError(f"Existing {path} has an incompatible schema; use --overwrite")
        return {row["case_id"]: row for row in reader}


def save_rows(rows: dict[str, dict[str, object]], path: Path, configured_order: list[str]) -> None:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for case_id in configured_order:
        if case_id in rows:
            writer.writerow(rows[case_id])
    atomic_write_text(stream.getvalue(), path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    validate_pr_transform()
    device = resolve_device(args.device, config["decoding"]["device"])
    warmup = args.warmup if args.warmup is not None else int(config["decoding"]["warmup"])
    repetitions = args.repetitions if args.repetitions is not None else int(config["decoding"]["repetitions"])
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be >= 0 and repetitions must be >= 1")

    case_ids = list(config["dataset"]["case_ids"])
    if args.case_limit is not None:
        case_ids = case_ids[: args.case_limit]
    class_names = list(config["dataset"]["class_names"])
    foreground_classes = list(range(1, len(class_names)))
    decoders = build_decoders()

    destination = output_dir(config) / "per_case.csv"
    rows: dict[str, dict[str, object]] = load_existing(destination, args.overwrite)
    for index, case_id in enumerate(case_ids, start=1):
        if case_id in rows:
            print(f"[{index}/{len(case_ids)}] Using existing metrics for {case_id}", flush=True)
            continue
        print(f"[{index}/{len(case_ids)}] Evaluating {case_id} on {device}", flush=True)
        payload = load_case(config, case_id)
        validate_probability_payload(payload, num_classes=len(class_names))
        probabilities = payload["probabilities"].to(device=device, dtype=torch.float32)
        label = payload["label"].to(torch.uint8)

        predictions: dict[str, torch.Tensor] = {}
        timings: dict[str, float] = {}
        raw_timings: dict[str, list[float]] = {}
        metrics: dict[str, tuple[float, float, dict[int, tuple[float, float]]]] = {}
        for method in METHODS:
            prediction = decode_once(decoders[method], probabilities)
            if tuple(prediction.shape) != tuple(label.shape):
                raise ValueError(
                    f"{method} shape mismatch for {case_id}: prediction={prediction.shape}, label={label.shape}"
                )
            labels = set(int(value) for value in torch.unique(prediction))
            if not labels.issubset(set(range(len(class_names)))):
                raise ValueError(f"{method} produced illegal labels {sorted(labels)} for {case_id}")
            predictions[method] = prediction.cpu()
            metrics[method] = case_metrics(predictions[method], label, foreground_classes)
            median_ms, samples_ms = time_decoder(
                decoders[method], probabilities, device, warmup=warmup, repetitions=repetitions
            )
            timings[method] = median_ms
            raw_timings[method] = samples_ms
            if device.type == "cuda":
                torch.cuda.empty_cache()

        shape = tuple(int(value) for value in label.shape)
        row: dict[str, object] = {
            "case_id": case_id,
            "shape": "x".join(str(value) for value in shape),
            "num_voxels": label.numel(),
        }
        for method in METHODS:
            row[f"{method}_dice"] = metrics[method][0]
            row[f"{method}_iou"] = metrics[method][1]
            row[f"{method}_time_ms"] = timings[method]
            for class_id, class_name in enumerate(class_names[1:], start=1):
                row[f"{method}_{class_name}_dice"] = metrics[method][2][class_id][0]
                row[f"{method}_{class_name}_iou"] = metrics[method][2][class_id][1]
        rows[case_id] = row
        save_rows(rows, destination, list(config["dataset"]["case_ids"]))

        timing_path = output_dir(config) / "timing_samples" / f"{case_id}.json"
        atomic_write_text(json.dumps(raw_timings, indent=2) + "\n", timing_path)
        del probabilities, predictions
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "config_sha256": config_sha256(config),
        "cases_evaluated": [case_id for case_id in config["dataset"]["case_ids"] if case_id in rows],
        "device": str(device),
        "timing": {
            "warmup": warmup,
            "repetitions": repetitions,
            "statistic": "median",
            "excludes": ["disk_read", "host_to_device_transfer", "model_inference", "device_to_host_transfer"],
            "cuda_synchronize_before_and_after": device.type == "cuda",
        },
        "software": software_info(),
    }
    atomic_write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", output_dir(config) / "evaluation_metadata.json"
    )
    print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
