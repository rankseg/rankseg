# MONAI PR #8908: fixed-model decoder experiment

This experiment asks one question: with a frozen public MONAI Bundle, does
replacing `AsDiscrete(argmax=True)` with RankSEG improve samplewise Dice/IoU,
and what is the post-processing cost?

It compares exactly three decoder calls from MONAI PR #8908:

```python
AsDiscrete(argmax=True)
AsDiscrete(rankseg=True, metric="dice")
AsDiscrete(rankseg=True, metric="iou")
```

The model is the unmodified `pancreas_ct_dints_segmentation` Bundle v0.5.2 on
MSD Task07 Pancreas. There is no retraining, fine-tuning, checkpoint change, or
RankSEG-specific preprocessing.

## Fixed protocol

- Split: reproduce the Bundle's `prepare_datalist.py` (`seed=123`, 196 of 281
  sorted labeled cases for training), then take the first 20 validation case IDs
  in numeric order. The IDs are frozen in `config.yaml`.
- Metrics: per-volume, foreground-class macro Dice and IoU. Background is
  excluded. For an empty class, both masks empty scores 1; otherwise the usual
  formula gives 0 when only the reference is empty.
- Timing: CUDA synchronization before/after every call, 5 warmups, 10 measured
  repetitions, median per case. Disk I/O, host/device transfer, inference, and
  device/host transfer are excluded.
- Statistics: mean ± sample SD, median, paired mean improvement, case-level
  paired 95% bootstrap CI (10,000 resamples), strict win rate, and per-class
  scores.
- Figure selection: nearest cases to the pre-specified 25%, 50%, and 75%
  quantiles of RankDice-minus-argmax Dice. The slice maximizes ground-truth
  foreground area; the crop uses ground truth only.

Checkpoint SHA-256 values and all other settings are recorded in `config.yaml`.

## Environment

The scripts intentionally fail if `AsDiscrete` does not expose an explicit
`rankseg` parameter. This prevents MONAI 1.5.2's permissive `**kwargs` from
silently ignoring `rankseg=True`.

```bash
git clone https://github.com/rankseg/MONAI.git vendor/MONAI-pr8908
git -C vendor/MONAI-pr8908 checkout fcee761a3280cd740753331a913a65fa0d90ba89
env/bin/python -m pip install rankseg==0.0.5 scikit-learn==1.5.1
```

Run commands from the repository root with the PR checkout first on
`PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/vendor/MONAI-pr8908"
export MPLCONFIGDIR=/tmp/matplotlib-monai-pr8908
```

## Smoke test (first 3 cases)

```bash
env/bin/python experiments/monai_pr8908/cache_probabilities.py --case-limit 3
env/bin/python experiments/monai_pr8908/evaluate_decoders.py --case-limit 3
env/bin/python experiments/monai_pr8908/summarize_results.py --allow-partial
env/bin/python experiments/monai_pr8908/visualize_cases.py
```

For a CPU-only smoke test, append `--device cpu`; final timing must use the GPU
configuration.

## Full fixed experiment

```bash
env/bin/python experiments/monai_pr8908/cache_probabilities.py
env/bin/python experiments/monai_pr8908/evaluate_decoders.py
env/bin/python experiments/monai_pr8908/summarize_results.py
env/bin/python experiments/monai_pr8908/visualize_cases.py
```

Outputs are written under `outputs/`:

```text
outputs/
├── probabilities/{case_id}.pt
├── timing_samples/{case_id}.json
├── per_case.csv
├── evaluation_metadata.json
├── summary.json
├── summary.md
├── qualitative_cases.json
└── qualitative_comparison.png
```

Probability maps are cached as float32 and shared unchanged by all decoders.
The cache also stores the identically resampled label and a float16 image only
for reproducible metric calculation and visualization. Large caches and result
artifacts should remain outside the MONAI PR diff.
