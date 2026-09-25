"""Paired before/after CUDA benchmark using a preserved source snapshot.

The snapshot must contain rankseg/{_screening,_screening_cuda,_rankseg_algo}.py.
Private module controls affect only this benchmark process, never saved caches.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch

from rankseg import _rankseg_algo as current
from scripts.benchmark_rma_screening import objective_diagnostics


def load_baseline(root):
    modules = {}
    names = ("_screening_cuda", "_screening", "_rankseg_algo")
    if (root / "rankseg/_validation.py").is_file():
        names = ("_validation", *names)
    for name in names:
        spec = importlib.util.spec_from_file_location("screening_baseline" + name, root / "rankseg" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    modules["_screening"]._cuda_backend = lambda: modules["_screening_cuda"]
    baseline = modules["_rankseg_algo"]
    if "_validation" in modules:
        for name in ("validate_probability_tensor", "validate_finite_real", "validate_integral",
                     "_validate_probability_values"):
            if hasattr(modules["_validation"], name):
                setattr(baseline, name, getattr(modules["_validation"], name))
    baseline._rma_dice_screened_masks = modules["_screening"]._rma_dice_screened_masks
    if hasattr(modules["_screening"], "_rma_dice_screening_statistics"):
        baseline._rma_dice_screening_statistics = modules["_screening"]._rma_dice_screening_statistics
    if hasattr(modules["_screening"], "_rma_dice_validated_statistics"):
        baseline._rma_dice_validated_statistics = modules["_screening"]._rma_dice_validated_statistics
    if hasattr(modules["_screening"], "_rma_dice_nonoverlap"):
        baseline._rma_dice_nonoverlap = modules["_screening"]._rma_dice_nonoverlap
    return baseline


def measure(probs, baseline, mode, repeats=11, warmup=3, check_objective=True):
    modules = {"before": baseline, "after": current}
    timings = {name: [] for name in modules}
    def call(name, output_mode=mode):
        return modules[name].rankseg_rma(probs, output_mode=output_mode, safe_screening=True)
    for _ in range(warmup):
        for name in modules:
            result = call(name)
            del result
    for repeat in range(repeats):
        for name in (list(modules) if repeat % 2 == 0 else list(reversed(modules))):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = call(name)
            torch.cuda.synchronize()
            timings[name].append((time.perf_counter() - start) * 1000)
            del result
    predictions, report = {}, {"shape": list(probs.shape), "mode": mode, "methods": {}}
    for name in modules:
        torch.cuda.synchronize()
        resident = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        prediction = call(name)
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - resident) / 2**20
        predictions[name] = prediction.cpu()
        del prediction
        objective = None
        if check_objective:
            binary = call(name, "multilabel")
            objective = objective_diagnostics(probs, binary)
            del binary
        report["methods"][name] = {"median_ms": statistics.median(timings[name]),
                                   "samples_ms": timings[name], "peak_mib": peak, "objective": objective}
    report["different_pixels"] = int((predictions["before"] != predictions["after"]).sum())
    report["speedup"] = report["methods"]["before"]["median_ms"] / report["methods"]["after"]["median_ms"]
    return report, predictions


def source_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "rankseg").glob("*.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--force-screening", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(4)
    baseline = load_baseline(args.baseline)
    if args.force_screening:
        for module in (baseline, current):
            if hasattr(module, "_rma_dice_use_screening"):
                module._rma_dice_use_screening = lambda probs, mode: True
            else:
                module._RMA_CUDA_SCREENING_MIN_SINGLE_CHANNEL_DIM = 0
                module._RMA_CUDA_SCREENING_MIN_FEW_ROWS_DIM = 0
    report = {"torch": torch.__version__, "device": torch.cuda.get_device_name(),
              "baseline": source_hashes(args.baseline),
              "current": source_hashes(Path(current.__file__).resolve().parents[1]), "rows": []}
    for shape in ((1, 1, 147456), (4, 1, 147456), (1, 3, 1048576),
                  (1, 3, 16777216), (1, 21, 65536)):
        for scenario in ("sparse", "ragged", "pruned"):
            p = torch.rand(shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(798))
            p = p.pow(16) if scenario == "ragged" else (p * 0.4 if scenario == "pruned" else
                                                       torch.where(p > .99, .99, 1e-5))
            for mode in (["multilabel"] if shape[1] == 1 else ["multilabel", "multiclass"]):
                row, _ = measure(p, baseline, mode, repeats=args.repeats)
                row["scenario"] = scenario
                report["rows"].append(row)
            del p
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps([{k: row[k] for k in ("shape", "scenario", "mode", "speedup", "different_pixels")}
                      for row in report["rows"]], indent=2))


if __name__ == "__main__":
    main()
