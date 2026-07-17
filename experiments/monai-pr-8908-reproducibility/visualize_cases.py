"""Create the pre-specified quantile-based 3x5 qualitative comparison."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-monai-pr8908")

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import (
    DEFAULT_CONFIG,
    atomic_write_text,
    load_case,
    load_config,
    output_dir,
    resolve_device,
    validate_probability_payload,
)
from monai.transforms import AsDiscrete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", help="Override the configured decoding device")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def choose_cases(rows: list[dict[str, str]]) -> list[tuple[float, dict[str, str]]]:
    if len(rows) < 3:
        raise ValueError("At least three evaluated cases are required for the 25/50/75% visualization")
    deltas = np.asarray([float(row["rankdice_dice"]) - float(row["argmax_dice"]) for row in rows], dtype=np.float64)
    targets = np.quantile(deltas, [0.25, 0.50, 0.75])
    available = set(range(len(rows)))
    selected: list[tuple[float, dict[str, str]]] = []
    for target in targets:
        index = min(available, key=lambda item: (abs(deltas[item] - target), rows[item]["case_id"]))
        available.remove(index)
        selected.append((float(target), rows[index]))
    return selected


def largest_foreground_slice(label: torch.Tensor) -> int:
    areas = (label > 0).sum(dim=(0, 1))
    return int(torch.argmax(areas))


def ground_truth_crop(mask: torch.Tensor, margin: int = 24, minimum_size: int = 128) -> tuple[slice, slice]:
    coordinates = torch.nonzero(mask, as_tuple=False)
    height, width = mask.shape
    if coordinates.numel() == 0:
        center_y, center_x = height // 2, width // 2
        y_radius = x_radius = minimum_size // 2
    else:
        y_min, x_min = coordinates.min(dim=0).values.tolist()
        y_max, x_max = coordinates.max(dim=0).values.tolist()
        center_y, center_x = (y_min + y_max) // 2, (x_min + x_max) // 2
        y_radius = max(minimum_size // 2, (y_max - y_min + 1) // 2 + margin)
        x_radius = max(minimum_size // 2, (x_max - x_min + 1) // 2 + margin)
    y_radius = min(y_radius, height // 2)
    x_radius = min(x_radius, width // 2)
    y0 = max(0, center_y - y_radius)
    y1 = min(height, center_y + y_radius)
    x0 = max(0, center_x - x_radius)
    x1 = min(width, center_x + x_radius)
    if y1 - y0 < min(minimum_size, height):
        y0 = max(0, min(y0, height - minimum_size))
        y1 = min(height, y0 + minimum_size)
    if x1 - x0 < min(minimum_size, width):
        x0 = max(0, min(x0, width - minimum_size))
        x1 = min(width, x0 + minimum_size)
    return slice(y0, y1), slice(x0, x1)


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    low = torch.quantile(image.float(), 0.01)
    high = torch.quantile(image.float(), 0.99)
    return ((image.float() - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)


def overlay(image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    base = normalize_image(image)
    rgb = torch.stack((base, base, base), dim=-1)
    colors = {1: torch.tensor([0.00, 0.78, 0.95]), 2: torch.tensor([1.00, 0.68, 0.05])}
    for class_id, color in colors.items():
        mask = labels == class_id
        rgb[mask] = 0.35 * rgb[mask] + 0.65 * color
    return rgb


def error_overlay(image: torch.Tensor, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    base = normalize_image(image) * 0.20
    rgb = torch.stack((base, base, base), dim=-1)
    pred_fg = prediction > 0
    label_fg = label > 0
    correct_fg = pred_fg & (prediction == label)
    false_positive = pred_fg & ~label_fg
    false_negative = ~pred_fg & label_fg
    wrong_class = pred_fg & label_fg & (prediction != label)
    rgb[correct_fg] = torch.tensor([1.0, 1.0, 1.0])
    rgb[false_positive] = torch.tensor([0.95, 0.15, 0.15])
    rgb[false_negative] = torch.tensor([0.00, 0.78, 0.95])
    rgb[wrong_class] = torch.tensor([1.00, 0.70, 0.05])
    return rgb


def display_array(tensor: torch.Tensor) -> np.ndarray:
    return np.rot90(tensor.detach().cpu().numpy())


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if "rankseg" not in inspect.signature(AsDiscrete).parameters:
        raise RuntimeError("Put the MONAI PR #8908 checkout first on PYTHONPATH")
    device = resolve_device(args.device, config["decoding"]["device"])
    rows = read_rows(output_dir(config) / "per_case.csv")
    selected = choose_cases(rows)
    argmax_decoder = AsDiscrete(argmax=True)
    rankdice_decoder = AsDiscrete(rankseg=True, metric="dice")

    fig, axes = plt.subplots(3, 5, figsize=(15.5, 9.2), constrained_layout=True)
    selection_metadata = []
    for row_index, (target_quantile, row) in enumerate(selected):
        case_id = row["case_id"]
        payload = load_case(config, case_id)
        validate_probability_payload(payload, len(config["dataset"]["class_names"]))
        probabilities = payload["probabilities"].to(device=device, dtype=torch.float32)
        label = payload["label"].to(torch.uint8)
        argmax = argmax_decoder(probabilities).squeeze(0).to(torch.uint8).cpu()
        rankdice = rankdice_decoder(probabilities).squeeze(0).to(torch.uint8).cpu()
        if argmax.shape != label.shape or rankdice.shape != label.shape:
            raise ValueError(f"Decoder shape mismatch for {case_id}")

        z_index = largest_foreground_slice(label)
        image_slice = payload["image"][:, :, z_index]
        label_slice = label[:, :, z_index]
        argmax_slice = argmax[:, :, z_index]
        rankdice_slice = rankdice[:, :, z_index]
        crop = ground_truth_crop(label_slice > 0)
        image_slice = image_slice[crop]
        label_slice = label_slice[crop]
        argmax_slice = argmax_slice[crop]
        rankdice_slice = rankdice_slice[crop]
        delta = float(row["rankdice_dice"]) - float(row["argmax_dice"])

        panels = [
            (
                f"q{[25, 50, 75][row_index]} · {case_id} · Image\nΔDice {delta:+.3f} · z={z_index}",
                normalize_image(image_slice),
                "gray",
            ),
            ("Ground truth", overlay(image_slice, label_slice), None),
            (f"Argmax\nDice {float(row['argmax_dice']):.3f}", overlay(image_slice, argmax_slice), None),
            (
                f"RankDice\nDice {float(row['rankdice_dice']):.3f}",
                overlay(image_slice, rankdice_slice),
                None,
            ),
            (
                "RankDice errors\nwhite TP · red FP · cyan FN · yellow class",
                error_overlay(image_slice, rankdice_slice, label_slice),
                None,
            ),
        ]
        for column, (title, panel, color_map) in enumerate(panels):
            axes[row_index, column].imshow(display_array(panel), cmap=color_map, interpolation="nearest")
            axes[row_index, column].set_title(title, fontsize=10)
            axes[row_index, column].axis("off")
        selection_metadata.append(
            {
                "case_id": case_id,
                "quantile": [0.25, 0.50, 0.75][row_index],
                "quantile_target_delta": target_quantile,
                "actual_delta": delta,
                "slice_index": z_index,
                "slice_rule": "ground_truth_foreground_area_maximum",
                "crop_rule": "ground_truth_foreground_bounds_only",
            }
        )
        del probabilities
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fig.suptitle(
        "MSD Pancreas · fixed pretrained DiNTS · no retraining or fine-tuning",
        fontsize=14,
    )
    figure_path = output_dir(config) / "qualitative_comparison.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    atomic_write_text(json.dumps(selection_metadata, indent=2) + "\n", output_dir(config) / "qualitative_cases.json")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
