"""Regression tests for release artifacts and their installed-package checks."""

import io
import tarfile
import types
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch

import rankseg
from scripts import smoke_test_distribution as smoke
from scripts.check_distributions import check_package_sources

SOURCES = {
    "rankseg/__init__.py": b"# init\n",
    "rankseg/_screening.py": b"# portable\n",
    "rankseg/_screening_cuda.py": b"# cuda\n",
    "rankseg/integration/sam.py": b"# nested package\n",
}


def make_artifacts(tmp_path, target=None, defect=None):
    root = tmp_path / "source"
    for name, content in SOURCES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    versions = {name: dict(SOURCES) for name in ("wheel", "sdist")}
    if defect == "missing":
        versions[target].pop("rankseg/_screening_cuda.py")
    elif defect == "stale":
        versions[target]["rankseg/_screening.py"] = b"# old module\n"
    elif defect == "unexpected":
        versions[target]["rankseg/old_module.py"] = b"# obsolete\n"
    wheel = tmp_path / "rankseg-0.0.6-py3-none-any.whl"
    sdist = tmp_path / "rankseg-0.0.6.tar.gz"
    metadata = b"Metadata-Version: 2.4\nName: rankseg\nVersion: 0.0.6\n\n"
    with ZipFile(wheel, "w") as archive:
        for name, content in versions["wheel"].items():
            archive.writestr(name, content)
        archive.writestr("rankseg-0.0.6.dist-info/METADATA", metadata)
    with tarfile.open(sdist, "w:gz") as archive:
        for name, content in {**versions["sdist"], "PKG-INFO": metadata}.items():
            info = tarfile.TarInfo("rankseg-0.0.6/" + name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return root, wheel, sdist


def test_package_sources_match_both_artifacts(tmp_path):
    root, wheel, sdist = make_artifacts(tmp_path)
    with ZipFile(wheel) as whl, tarfile.open(sdist) as src:
        check_package_sources(root, whl, src, "rankseg-0.0.6/")


@pytest.mark.parametrize("target", ["wheel", "sdist"])
@pytest.mark.parametrize("defect", ["missing", "stale", "unexpected"])
def test_package_sources_reject_bad_artifacts(tmp_path, target, defect):
    root, wheel, sdist = make_artifacts(tmp_path, target, defect)
    with ZipFile(wheel) as whl, tarfile.open(sdist) as src:
        with pytest.raises(AssertionError, match=target):
            check_package_sources(root, whl, src, "rankseg-0.0.6/")


@pytest.mark.parametrize("kind", [1, 2], ids=["wheel", "sdist"])
def test_smoke_reads_artifact_version_and_all_sources(tmp_path, kind):
    paths = make_artifacts(tmp_path)
    assert smoke.artifact_sources(paths[kind]) == ("0.0.6", SOURCES)


@pytest.mark.parametrize("kind", [1, 2], ids=["wheel", "sdist"])
def test_smoke_rejects_artifact_missing_cuda_module(tmp_path, kind):
    paths = make_artifacts(tmp_path, "wheel" if kind == 1 else "sdist", "missing")
    with pytest.raises(AssertionError, match="Missing package module"):
        smoke.artifact_sources(paths[kind])


@pytest.mark.parametrize("kind", [1, 2], ids=["wheel", "sdist"])
@pytest.mark.parametrize("defect", [None, "checkout", "shadow", "version", "stale", "missing"])
def test_smoke_verifies_installed_origin_version_and_bytes(tmp_path, monkeypatch, kind, defect):
    paths = make_artifacts(tmp_path)
    site = tmp_path / "site-packages"
    for name, content in SOURCES.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    module = types.SimpleNamespace(__file__=str(site / "rankseg/__init__.py"), __version__="0.0.6")
    metadata = types.SimpleNamespace(version="0.0.6", locate_file=lambda name: site / name)
    monkeypatch.setattr(smoke, "distribution", lambda name: metadata)
    source_root = paths[0]
    if defect == "checkout":
        source_root = site
    elif defect == "shadow":
        module.__file__ = str(tmp_path / "shadow/rankseg/__init__.py")
    elif defect == "version":
        module.__version__ = "0.0.5"
    elif defect == "stale":
        (site / "rankseg/_screening.py").write_bytes(b"# old module\n")
    elif defect == "missing":
        (site / "rankseg/_screening_cuda.py").unlink()
    if defect:
        with pytest.raises(AssertionError):
            smoke.verify_installation(module, paths[kind], source_root)
    else:
        assert smoke.verify_installation(module, paths[kind], source_root) == Path(module.__file__)


@pytest.mark.parametrize("scale", [1, 8192], ids=["cpu-size", "cuda-size"])
def test_golden_masks_match_independent_float64_formula(scale):
    probs, binary, labels = smoke.golden_case(torch, "cpu")
    probs = probs.repeat(1, 1, scale).double()
    binary = binary.repeat(1, 1, scale)
    labels = labels.repeat(1, scale)
    values, order = probs.sort(descending=True, dim=-1)
    mu = probs.sum(-1)
    k = torch.arange(1, probs.shape[-1] + 1, dtype=torch.float64)
    scores = 2 * values.cumsum(-1) / (mu[..., None] + k + 1)
    optimum = scores.argmax(-1) + 1
    expected = torch.zeros_like(binary).scatter(-1, order, k <= optimum[..., None])
    assert torch.equal(binary, expected)
    unique = binary & (binary.sum(1, keepdim=True) == 1)
    count = unique.sum(-1)
    mass = (probs * unique).sum(-1)
    incremental = 2 * (mass[..., None] + probs) / (mu + count + 2)[..., None]
    incremental -= (2 * mass / (mu + count + 1))[..., None]
    incremental.masked_fill_(~binary, -torch.inf)
    assert torch.equal(labels, incremental.argmax(1))


def test_installed_prediction_checks_on_cpu():
    assert smoke.check_predictions(rankseg, torch, "cpu") == 33


def test_cpu_screening_never_loads_cuda_backend(monkeypatch):
    from rankseg import _screening

    def forbidden():
        raise AssertionError("CPU predictions must not load the CUDA/Triton backend")

    monkeypatch.setattr(_screening, "_cuda_backend", forbidden)
    assert smoke.check_predictions(rankseg, torch, "cpu") == 33


def test_smoke_rejects_triton_in_no_triton_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(AssertionError, match="genuinely Triton-free"):
        smoke.smoke_test(tmp_path / "unused.whl", tmp_path, require_no_triton=True)


def test_cuda_acceptance_does_not_silently_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "verify_installation", lambda *args: tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(AssertionError, match="CUDA is unavailable"):
        smoke.smoke_test(tmp_path / "unused.whl", tmp_path, device="cuda")


def test_cuda_acceptance_rejects_bypassed_kernels(tmp_path, monkeypatch):
    from rankseg import _screening

    names = (
        "validated_screening_statistics", "pack_candidates", "gather_candidate_group",
        "score_argmax", "scatter_candidate_group", "dice_nonoverlap_from_masks",
    )
    backend = types.SimpleNamespace(**{name: lambda *args: None for name in names})
    monkeypatch.setattr(_screening, "_cuda_backend", lambda: backend)
    monkeypatch.setattr(smoke, "verify_installation", lambda *args: tmp_path)
    monkeypatch.setattr(smoke, "check_predictions", lambda *args: 19)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    with pytest.raises(AssertionError, match="did not exercise validated_screening_statistics"):
        smoke.smoke_test(tmp_path / "unused.whl", tmp_path, device="cuda")
