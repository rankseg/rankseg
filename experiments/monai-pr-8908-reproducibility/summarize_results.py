"""Summarize samplewise decoder metrics with paired bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from common import DEFAULT_CONFIG, atomic_write_text, config_sha256, load_config, output_dir

METHODS = ("argmax", "rankdice", "rankiou")
DISPLAY_NAMES = {"argmax": "Argmax", "rankdice": "RankDice", "rankiou": "RankIoU"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--allow-partial", action="store_true", help="Allow fewer than all configured cases")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation output: {path}")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def values(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.asarray([float(row[column]) for row in rows], dtype=np.float64)


def describe(array: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def bootstrap_mean_ci(
    array: np.ndarray, repetitions: int, confidence_level: float, rng: np.random.Generator
) -> list[float]:
    sample_indices = rng.integers(0, len(array), size=(repetitions, len(array)))
    bootstrap_means = array[sample_indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return [float(value) for value in np.quantile(bootstrap_means, [alpha, 1.0 - alpha])]


def paired_comparison(
    candidate: np.ndarray,
    baseline: np.ndarray,
    repetitions: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    delta = candidate - baseline
    tolerance = 1e-12
    return {
        **describe(delta),
        "mean_95ci": bootstrap_mean_ci(delta, repetitions, confidence_level, rng),
        "win_rate": float(np.mean(delta > tolerance)),
        "tie_rate": float(np.mean(np.abs(delta) <= tolerance)),
        "loss_rate": float(np.mean(delta < -tolerance)),
        "wins": int(np.count_nonzero(delta > tolerance)),
        "ties": int(np.count_nonzero(np.abs(delta) <= tolerance)),
        "losses": int(np.count_nonzero(delta < -tolerance)),
    }


def format_score(summary: dict[str, float]) -> str:
    return f"{summary['mean']:.4f} ± {summary['std']:.4f}"


def make_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# MONAI PR #8908 decoder experiment ({summary['n_cases']} cases)",
        "",
        "Samplewise scores are foreground-class macro averages computed per volume; background is excluded.",
        "Timing is the mean across cases of each case's median repeated decoder time.",
        "",
        "| Method | Samplewise Dice | Samplewise IoU | Target-metric win rate vs Argmax | Time/case |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        method_summary = summary["methods"][method]
        if method == "argmax":
            win_rate = "—"
        else:
            target_metric = "dice" if method == "rankdice" else "iou"
            win_rate = f"{summary['paired_vs_argmax'][method][target_metric]['win_rate'] * 100:.1f}%"
        lines.append(
            f"| {DISPLAY_NAMES[method]} | {format_score(method_summary['dice'])} | "
            f"{format_score(method_summary['iou'])} | {win_rate} | "
            f"{method_summary['time_ms']['mean']:.2f} ms |"
        )

    lines.extend(
        [
            "",
            "| Comparison | Metric | Paired mean improvement | 95% bootstrap CI | Wins / ties / losses |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for method in ("rankdice", "rankiou"):
        for metric in ("dice", "iou"):
            comparison = summary["paired_vs_argmax"][method][metric]
            low, high = comparison["mean_95ci"]
            lines.append(
                f"| {DISPLAY_NAMES[method]} vs Argmax | {metric.upper()} | {comparison['mean']:+.4f} | "
                f"[{low:+.4f}, {high:+.4f}] | "
                f"{comparison['wins']} / {comparison['ties']} / {comparison['losses']} |"
            )

    lines.extend(["", "## Per-class mean scores", ""])
    lines.extend(
        [
            "| Method | Pancreas Dice | Tumor Dice | Pancreas IoU | Tumor IoU |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in METHODS:
        per_class = summary["methods"][method]["per_class"]
        lines.append(
            f"| {DISPLAY_NAMES[method]} | {per_class['pancreas']['dice']['mean']:.4f} | "
            f"{per_class['tumor']['dice']['mean']:.4f} | {per_class['pancreas']['iou']['mean']:.4f} | "
            f"{per_class['tumor']['iou']['mean']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows = read_rows(output_dir(config) / "per_case.csv")
    configured_cases = list(config["dataset"]["case_ids"])
    actual_cases = [row["case_id"] for row in rows]
    unexpected = sorted(set(actual_cases) - set(configured_cases))
    missing = [case_id for case_id in configured_cases if case_id not in actual_cases]
    if unexpected:
        raise ValueError(f"per_case.csv contains unexpected cases: {unexpected}")
    if missing and not args.allow_partial:
        raise ValueError(
            f"Missing {len(missing)} configured cases; use --allow-partial only for a smoke test: {missing}"
        )
    order = {case_id: index for index, case_id in enumerate(configured_cases)}
    rows.sort(key=lambda row: order[row["case_id"]])

    repetitions = int(config["metrics"]["bootstrap_repetitions"])
    confidence_level = float(config["metrics"]["confidence_level"])
    rng = np.random.default_rng(int(config["experiment"]["seed"]))
    summary: dict[str, Any] = {
        "experiment": config["experiment"]["name"],
        "config_sha256": config_sha256(config),
        "n_cases": len(rows),
        "case_ids": [row["case_id"] for row in rows],
        "missing_configured_cases": missing,
        "include_background": bool(config["dataset"]["include_background"]),
        "empty_class_policy": config["dataset"]["empty_class_policy"],
        "bootstrap": {
            "repetitions": repetitions,
            "confidence_level": confidence_level,
            "seed": int(config["experiment"]["seed"]),
            "unit": "case",
            "paired": True,
        },
        "methods": {},
        "paired_vs_argmax": {},
    }
    for method in METHODS:
        method_summary: dict[str, Any] = {
            "dice": describe(values(rows, f"{method}_dice")),
            "iou": describe(values(rows, f"{method}_iou")),
            "time_ms": describe(values(rows, f"{method}_time_ms")),
            "per_class": {},
        }
        for class_name in config["dataset"]["class_names"][1:]:
            method_summary["per_class"][class_name] = {
                metric: describe(values(rows, f"{method}_{class_name}_{metric}")) for metric in ("dice", "iou")
            }
        summary["methods"][method] = method_summary

    for method in ("rankdice", "rankiou"):
        summary["paired_vs_argmax"][method] = {}
        for metric in ("dice", "iou"):
            summary["paired_vs_argmax"][method][metric] = paired_comparison(
                values(rows, f"{method}_{metric}"),
                values(rows, f"argmax_{metric}"),
                repetitions,
                confidence_level,
                rng,
            )

    metadata_path = output_dir(config) / "evaluation_metadata.json"
    if metadata_path.exists():
        summary["evaluation_metadata"] = json.loads(metadata_path.read_text())
    json_path = output_dir(config) / "summary.json"
    markdown_path = output_dir(config) / "summary.md"
    atomic_write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", json_path)
    markdown = make_markdown(summary)
    atomic_write_text(markdown, markdown_path)
    print(markdown)
    print(f"Saved {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
