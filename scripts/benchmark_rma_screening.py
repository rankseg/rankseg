"""Compare opt-in screening with full-sort RMA on paired synthetic inputs.

Run from the repository root, for example:
    python -m scripts.benchmark_rma_screening --device cuda --output /tmp/rma.json

Generation, correctness checks, and warm-up are excluded from timing. GPU
latency includes host scheduling/synchronization, and memory is incremental
peak allocated memory with the input already resident. No model is run.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from rankseg import _screening as screening_module
from rankseg._rankseg_algo import rankseg_rma


def make_probabilities(scenario, dim, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand((2, 3, dim), generator=generator, dtype=dtype)
    if scenario == "sparse_resolved":
        return torch.where(values > 0.99, values.new_tensor(0.99), values.new_tensor(1e-5))
    if scenario == "all_pruned":
        return values * 0.4
    if scenario == "softmax_sparse":
        # A dominant background and two rare foreground classes. Unlike the
        # independent sparse profile, these probabilities sum to one per pixel.
        label = (values[:, 0] > 0.98).long() + (values[:, 0] > 0.99).long()
        normalized = torch.full_like(values, 1e-5)
        return normalized.scatter_(1, label[:, None], 1 - 2e-5)
    if scenario == "ragged":
        values[:, 0] = torch.where(values[:, 0] > 0.99, 0.99, 1e-5)
        values[:, 1] = values[:, 1].pow(16)
        values[:, 2] = values[:, 2].pow(4)
    return values


def objective_diagnostics(probs, masks, pruning_prob=0.5):
    """Independent float64 full-prefix oracle; never included in timing.

    Limit oracle workspace by evaluating a bounded number of active rows at a
    time. A single large row is still evaluated as one decision set.
    """
    flat = probs.flatten(2).reshape(-1, probs.flatten(2).shape[-1])
    selected = masks.reshape_as(flat)
    active = flat.amax(-1) > pruning_prob
    inactive_rows = (~active).nonzero(as_tuple=True)[0]
    if selected.index_select(0, inactive_rows).any():
        raise AssertionError("class pruning changed")
    indices = active.nonzero(as_tuple=True)[0]
    step = max(1, 1_048_576 // flat.shape[-1])
    regrets = []
    volumes = torch.arange(1, flat.shape[-1] + 1, dtype=torch.float64, device=probs.device)
    for start in range(0, indices.numel(), step):
        rows = indices[start:start + step]
        values = flat.index_select(0, rows).double()
        means = values.sum(-1)
        chosen = selected.index_select(0, rows)
        actual = 2 * (values * chosen).sum(-1) / (means + chosen.sum(-1) + 1)
        prefixes = values.sort(descending=True).values.cumsum(-1)
        optimum = (2 * prefixes / (means[:, None] + volumes + 1)).amax(-1)
        regrets.extend((optimum - actual).clamp_min(0).cpu().tolist())
    working_dtype = torch.float64 if probs.dtype == torch.float64 else torch.float32
    eps = torch.finfo(working_dtype).eps
    # Regression budget for arithmetic, not an inference tie rule or a
    # universal analytic bound.
    limit = (16 if working_dtype == torch.float64 else 4) * eps
    maximum = max(regrets, default=0.0)
    if maximum > limit:
        raise AssertionError(f"objective regret {maximum} exceeds {limit} ({maximum / eps} eps)")
    return {"max_regret": maximum, "max_regret_eps": maximum / eps,
            "mean_regret": statistics.mean(regrets) if regrets else 0.0,
            "active_rows": len(regrets), "checked_limit": limit}


def benchmark(probs, output_mode, repeats, warmup):
    use_cuda = probs.device.type == "cuda"

    def sync():
        if use_cuda:
            torch.cuda.synchronize(probs.device)

    def predict(screening):
        return rankseg_rma(probs, output_mode=output_mode, safe_screening=screening)

    expected, actual = predict(False), predict(True)
    differing_pixels = int((expected != actual).sum().item())
    del expected, actual
    binary = rankseg_rma(probs, output_mode="multilabel", safe_screening=True)
    objective = objective_diagnostics(probs, binary)
    del binary
    for _ in range(warmup):
        for screening in (False, True):
            prediction = predict(screening)
            del prediction
    sync()
    timings = {False: [], True: []}
    for repetition in range(repeats):
        for screening in ((False, True) if repetition % 2 == 0 else (True, False)):
            sync()
            start = time.perf_counter()
            prediction = predict(screening)
            sync()
            timings[screening].append((time.perf_counter() - start) * 1000)
            del prediction
    full_ms = statistics.median(timings[False])
    screened_ms = statistics.median(timings[True])
    result = {
        "full_ms": full_ms,
        "screened_ms": screened_ms,
        "speedup": full_ms / screened_ms,
        "fallback_rows": 0,  # Schema compatibility; candidate retries were removed.
        "rows": probs.shape[0] * probs.shape[1],
        "outputs_equal": differing_pixels == 0,
        "differing_pixels": differing_pixels,
        "objective": objective,
        "full_samples_ms": timings[False],
        "screened_samples_ms": timings[True],
    }
    if use_cuda:
        for screening, name in ((False, "full"), (True, "screened")):
            sync()
            baseline = torch.cuda.memory_allocated(probs.device)
            torch.cuda.reset_peak_memory_stats(probs.device)
            prediction = predict(screening)
            sync()
            result[f"{name}_peak_mib"] = (torch.cuda.max_memory_allocated(probs.device) - baseline) / 2**20
            del prediction
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtypes", choices=["float32", "float64"], nargs="+", default=["float32", "float64"])
    parser.add_argument("--dims", type=int, nargs="+", default=[65_536, 524_288, 1_048_576])
    parser.add_argument("--scenarios", nargs="+",
                        choices=["sparse_resolved", "softmax_sparse", "all_pruned", "uniform", "ragged"],
                        default=["sparse_resolved", "softmax_sparse", "all_pruned", "uniform", "ragged"])
    parser.add_argument("--modes", nargs="+", choices=["multilabel", "multiclass"], default=["multilabel", "multiclass"])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3401)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force-screening", action="store_true", help="legacy flag; True now always forces screening")
    parser.add_argument("--torch-screening", action="store_true", help="disable optional fused CUDA kernels")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.threads < 1 or any(dim < 1 for dim in args.dims):
        parser.error("dimensions, repeats, and threads must be positive; warmup must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; do not silently benchmark CPU instead")
    torch.set_num_threads(args.threads)
    # This CLI runs in its own process; do not change production dispatch rules.
    if args.torch_screening:
        screening_module._cuda_backend = lambda: None
    rows = []
    for dtype in args.dtypes:
        for dim in args.dims:
            for scenario in args.scenarios:
                probs = make_probabilities(scenario, dim, getattr(torch, dtype), args.seed).to(args.device)
                for mode in args.modes:
                    row = {"dtype": dtype, "dim": dim, "scenario": scenario, "output_mode": mode}
                    row.update(benchmark(probs, mode, args.repeats, args.warmup))
                    rows.append(row)
                    print(f"{dtype:7} D={dim:8} {scenario:15} {mode:10} "
                          f"{row['full_ms']:8.3f} -> {row['screened_ms']:8.3f} ms "
                          f"{row['speedup']:5.2f}x fallback={row['fallback_rows']}/{row['rows']}", flush=True)
    report = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "threads": args.threads,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "shape": "(2, 3, D)",
        "force_screening": args.force_screening,
        "fused_cuda": args.device == "cuda" and screening_module._cuda_backend() is not None,
        "timing": "median wall-clock latency including host overhead; synchronized on CUDA",
        "results": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
