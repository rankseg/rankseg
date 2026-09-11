"""Render theme-aware transparent benchmark charts for the project READMEs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class BenchmarkResult:
    dataset: str
    argmax_iou: float
    rankseg_iou: float
    argmax_dice: float
    rankseg_dice: float


# This promotional chart intentionally shows the five positive-gain cohorts.
# The linked rankseg-benchmark report remains the source of complete results,
# including the external MSD Spleen cohort.
RESULTS = (
    BenchmarkResult("PASCAL VOC", 87.80, 88.23, 91.02, 91.46),
    BenchmarkResult("Cityscapes", 75.67, 76.19, 82.62, 83.23),
    BenchmarkResult("ADE20K", 56.95, 57.67, 63.98, 64.92),
    BenchmarkResult("KiTS", 54.19, 56.21, 61.16, 63.53),
    BenchmarkResult("MSD\nPancreas", 35.2915010536, 39.1872318675, 50.2952507760, 54.8695724574),
)

THEMES = {
    "light": {
        "text": "#111827",
        "muted": "#334155",
        "grid": "#dfe3eb",
        "axis": "#979fac",
        "rankseg": "#2446c4",
        "argmax": "#b9bec8",
    },
    "dark": {
        "text": "#f1f4fa",
        "muted": "#a9b2c3",
        "grid": "#303744",
        "axis": "#707a8c",
        "rankseg": "#7184ff",
        "argmax": "#7d879a",
    },
}


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size=size)
    except OSError:  # pragma: no cover - depends on fonts installed by the OS
        return ImageFont.load_default()


def _validate_results() -> None:
    if len(RESULTS) != 5:
        raise ValueError("The promotional README chart must contain the five selected benchmark datasets")
    for result in RESULTS:
        values = (result.argmax_iou, result.rankseg_iou, result.argmax_dice, result.rankseg_dice)
        if any(not 0.0 <= value <= 100.0 for value in values):
            raise ValueError(f"Metric outside [0, 100] for {result}")
        if result.rankseg_iou <= result.argmax_iou or result.rankseg_dice <= result.argmax_dice:
            raise ValueError(f"Promotional chart contains a non-positive gain for {result}")


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bounds = draw.textbbox((0, 0), text, font=font)
    return bounds[2] - bounds[0]


def _center_text(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    draw.text((x - _text_width(draw, text, font) / 2, y), text, font=font, fill=fill)


def _draw_legend(draw: ImageDraw.ImageDraw, colors: dict[str, str]) -> None:
    y = 62
    draw.rectangle((1430, y, 1480, y + 25), fill=colors["rankseg"])
    draw.text((1498, y - 9), "RankSEG", font=_font(25, bold=True), fill=colors["rankseg"])
    draw.rectangle((1696, y, 1746, y + 25), fill=colors["argmax"])
    draw.text((1764, y - 9), "Argmax", font=_font(25), fill=colors["text"])


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    *,
    panel_top: int,
    title: str,
    metric: str,
    colors: dict[str, str],
) -> None:
    chart_left = 104
    chart_right = 1950
    plot_top = panel_top + 78
    plot_bottom = panel_top + 444
    plot_height = plot_bottom - plot_top
    label_y = plot_bottom + 20
    gain_y = plot_bottom + 74
    tick_font = _font(24)
    value_font = _font(27)
    value_bold_font = _font(28, bold=True)
    dataset_font = _font(27)
    gain_font = _font(25, bold=True)

    draw.text((chart_left, panel_top), title, font=_font(29, bold=True), fill=colors["text"])
    draw.text((1730, panel_top + 5), "Higher is better", font=_font(24), fill=colors["muted"])

    for tick in (0, 25, 50, 75, 100):
        y = plot_bottom - plot_height * tick / 100
        draw.line((chart_left, y, chart_right, y), fill=colors["grid"], width=2)
        tick_text = str(tick)
        draw.text(
            (chart_left - _text_width(draw, tick_text, tick_font) - 15, y - 12),
            tick_text,
            font=tick_font,
            fill=colors["muted"],
        )
    draw.line((chart_left, plot_bottom, chart_right, plot_bottom), fill=colors["axis"], width=2)

    group_width = (chart_right - chart_left) / len(RESULTS)
    bar_width = 110
    bar_gap = 18
    for index, result in enumerate(RESULTS):
        center = chart_left + group_width * (index + 0.5)
        if metric == "dice":
            baseline, optimized = result.argmax_dice, result.rankseg_dice
        else:
            baseline, optimized = result.argmax_iou, result.rankseg_iou
        gain = optimized - baseline

        baseline_left = center - bar_gap / 2 - bar_width
        rankseg_left = center + bar_gap / 2
        baseline_top = plot_bottom - plot_height * baseline / 100
        rankseg_top = plot_bottom - plot_height * optimized / 100
        draw.rectangle((baseline_left, baseline_top, baseline_left + bar_width, plot_bottom), fill=colors["argmax"])
        draw.rectangle((rankseg_left, rankseg_top, rankseg_left + bar_width, plot_bottom), fill=colors["rankseg"])

        _center_text(
            draw,
            baseline_left + bar_width / 2,
            baseline_top - 39,
            f"{baseline:.2f}",
            font=value_font,
            fill=colors["text"],
        )
        _center_text(
            draw,
            rankseg_left + bar_width / 2,
            rankseg_top - 41,
            f"{optimized:.2f}",
            font=value_bold_font,
            fill=colors["rankseg"],
        )

        lines = result.dataset.splitlines()
        for line_index, line in enumerate(lines):
            _center_text(
                draw,
                center,
                label_y + line_index * 29,
                line,
                font=dataset_font,
                fill=colors["text"],
            )
        _center_text(
            draw,
            center,
            gain_y + (18 if len(lines) > 1 else 0),
            f"{gain:+.2f} pp",
            font=gain_font,
            fill=colors["rankseg"],
        )


def render_chart(output: Path, *, theme: str) -> None:
    _validate_results()
    colors = THEMES[theme]
    canvas = Image.new("RGBA", (2000, 1380), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    draw.text((60, 34), "Selected RankSEG benchmark results", font=_font(48, bold=True), fill=colors["text"])
    draw.text(
        (62, 98),
        "RMA with Dice objective  ·  frozen probabilities/checkpoints  ·  no retraining",
        font=_font(26),
        fill=colors["muted"],
    )
    _draw_legend(draw, colors)
    _draw_panel(draw, panel_top=176, title="mDice (%)", metric="dice", colors=colors)
    _draw_panel(draw, panel_top=772, title="mIoU (%)", metric="iou", colors=colors)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-light", type=Path, default=Path("fig/benchmark_results.png"))
    parser.add_argument("--output-dark", type=Path, default=Path("fig/benchmark_results_dark.png"))
    args = parser.parse_args()
    light_output = args.output_light.expanduser().resolve()
    dark_output = args.output_dark.expanduser().resolve()
    render_chart(light_output, theme="light")
    render_chart(dark_output, theme="dark")
    print(f"Wrote {light_output}")
    print(f"Wrote {dark_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
