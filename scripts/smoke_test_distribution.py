"""Test an installed wheel/sdist, outside the source tree, without pytest.

Run with the target environment's Python and -I, for example:
    python -I /checkout/scripts/smoke_test_distribution.py \
        --artifact /dist/rankseg-VERSION.whl --source-root /checkout \
        --device cpu --require-no-triton

Use --device cuda for local release acceptance; missing CUDA/Triton is an
error, not a skip. This is a correctness smoke test, not a performance test.
"""

import argparse
import importlib.util
import inspect
import json
import tarfile
from contextlib import ExitStack
from email.parser import Parser
from importlib.metadata import distribution
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile


def artifact_sources(artifact):
    """Read the version and intended package bytes without importing the source."""
    if artifact.suffix == ".whl":
        with ZipFile(artifact) as archive:
            metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            assert len(metadata) == 1, "Expected exactly one wheel metadata file"
            version = Parser().parsestr(archive.read(metadata[0]).decode("utf-8"))["Version"]
            sources = {name: archive.read(name) for name in archive.namelist()
                       if name.startswith("rankseg/") and name.endswith(".py")}
    else:
        with tarfile.open(artifact, "r:gz") as archive:
            metadata = [name for name in archive.getnames() if name.count("/") == 1 and name.endswith("/PKG-INFO")]
            assert len(metadata) == 1, "Expected exactly one sdist metadata file"
            version = Parser().parsestr(archive.extractfile(metadata[0]).read().decode("utf-8"))["Version"]
            prefix = metadata[0].removesuffix("PKG-INFO")
            sources = {name.removeprefix(prefix): archive.extractfile(name).read()
                       for name in archive.getnames()
                       if name.startswith(prefix + "rankseg/") and name.endswith(".py")}
    assert version and sources, "Artifact has no version or package sources"
    for name in ("rankseg/__init__.py", "rankseg/_screening.py", "rankseg/_screening_cuda.py"):
        assert name in sources, f"Missing package module: {name}"
    return version, sources


def verify_installation(module, artifact, source_root):
    version, sources = artifact_sources(artifact)
    installed = distribution("rankseg")
    package_path = Path(module.__file__).resolve()
    assert not package_path.is_relative_to(source_root.resolve()), f"Imported source checkout: {package_path}"
    assert package_path == Path(installed.locate_file("rankseg/__init__.py")).resolve(), (
        f"Imported module does not belong to the installed distribution: {package_path}"
    )
    assert module.__version__ == installed.version == version, "Artifact/imported/installed versions differ"
    for name, content in sources.items():
        path = package_path.parent.parent / name
        assert path.is_file() and path.read_bytes() == content, f"Installed module differs from artifact: {name}"
    return package_path


def golden_case(torch, device):
    # Each four-pixel block has one forced positive, one retained candidate,
    # and two excluded entries in class 0. Both classes select pixel 1;
    # class 1 wins its incremental-score assignment. These are well-separated
    # optima, not an assumption of universal bitwise screening equivalence.
    # Two channels * four pixels * 160000 repeats reaches auto's CUDA cutoff.
    repeats = 160000 if device == "cuda" else 4
    foreground = torch.tensor([0.875, 0.375, 0.25, 0.0], device=device).repeat(repeats)
    probs = torch.stack((foreground, 1 - foreground))[None]
    binary = torch.tensor([[True, True, False, False], [False, True, True, True]], device=device)
    binary = binary.repeat(1, repeats)[None]
    labels = torch.tensor([0, 1, 1, 1], device=device).repeat(repeats)[None]
    return probs, binary, labels


def check_predictions(module, torch, device):
    assert inspect.signature(module.rankseg_rma).parameters["safe_screening"].default is False
    probs, binary, labels = golden_case(torch, device)
    original = probs.clone()
    checked = 0
    for mode, expected in (("multilabel", binary), ("multiclass", labels)):
        kwargs = dict(metric="dice", smooth=0, output_mode=mode, pruning_prob=0.5)
        for enabled in (None, False, True, "auto"):
            # None means omit the option, not pass an invalid None value.
            params = dict(kwargs) if enabled is None else dict(kwargs, safe_screening=enabled)
            decoder = module.RankSEG(solver="RMA", **params)
            predictions = (
                decoder(probs),
                decoder.predict(probs),
                module.functional.rankseg(probs, solver="RMA", **params),
                module.rankseg_rma(probs, **params),
            )
            for result in predictions:
                assert result.shape == expected.shape and result.dtype == expected.dtype
                assert result.device == probs.device and not result.requires_grad
                assert torch.equal(result, expected), f"Wrong {mode} output with safe_screening={enabled}"
                checked += 1
    # Retain the original Accuracy smoke test too.
    assert torch.equal(module.RankSEG(metric="accuracy", solver="argmax")(probs), labels)
    assert torch.equal(probs, original), "Decoder modified its input probabilities"
    return checked + 1


def smoke_test(artifact, source_root, device="cpu", require_no_triton=False):
    if require_no_triton:
        assert device == "cpu", "The no-Triton installation check uses CPU"
        assert importlib.util.find_spec("triton") is None, "Expected a genuinely Triton-free environment"

    import torch

    import rankseg

    package_path = verify_installation(rankseg, artifact, source_root)
    calls = {}
    with ExitStack() as stack:
        if device == "cuda":
            assert torch.cuda.is_available(), "CUDA acceptance requested but CUDA is unavailable"
            from rankseg import _screening

            backend = _screening._cuda_backend()
            assert backend is not None, "CUDA acceptance requires the Triton backend"
            # Observe actual calls without forcing dispatch or replacing any computation.
            for name in (
                "validated_screening_statistics", "pack_candidates", "gather_candidate_group",
                "score_argmax", "scatter_candidate_group", "dice_nonoverlap_from_masks",
            ):
                calls[name] = stack.enter_context(patch.object(backend, name, wraps=getattr(backend, name)))
        checked = check_predictions(rankseg, torch, device)
        if device == "cuda":
            torch.cuda.synchronize()
            for name, observed in calls.items():
                assert observed.call_count > 0, f"CUDA smoke test did not exercise {name}"

    report = {
        "artifact": str(artifact.resolve()), "rankseg": rankseg.__version__, "package_path": str(package_path),
        "torch": torch.__version__, "device": device, "predictions_checked": checked,
        "triton_absent_verified": require_no_triton,
        "cuda_backend_calls": {name: observed.call_count for name, observed in calls.items()},
    }
    if device == "cuda":
        import triton

        report.update(triton=triton.__version__, gpu=torch.cuda.get_device_name())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--require-no-triton", action="store_true")
    args = parser.parse_args()
    print(json.dumps(smoke_test(args.artifact, args.source_root, args.device, args.require_no_triton), indent=2))


if __name__ == "__main__":
    main()
