from pathlib import Path

from coverage import Coverage

ROOT = Path(__file__).resolve().parents[1]


def test_cpu_coverage_preserves_full_coverage_settings():
    full = Coverage(config_file=str(ROOT / "pyproject.toml"))
    cpu = Coverage(config_file=str(ROOT / ".coveragerc.cpu"))
    for option in (
        "run:source",
        "report:omit",
        "report:exclude_lines",
        "report:fail_under",
        "report:show_missing",
        "report:precision",
        "xml:output",
    ):
        assert cpu.get_option(option) == full.get_option(option), option


def test_full_coverage_keeps_cuda_backend_and_original_gate():
    full = Coverage(config_file=str(ROOT / "pyproject.toml"))
    assert full.get_option("run:source") == ["rankseg"]
    assert full.get_option("run:omit") == ["tests/*"]
    assert not full.get_option("report:omit")
    assert full.get_option("report:fail_under") == 85


def test_cpu_coverage_only_adds_cuda_backend_omission():
    full = Coverage(config_file=str(ROOT / "pyproject.toml"))
    cpu = Coverage(config_file=str(ROOT / ".coveragerc.cpu"))
    assert cpu.get_option("run:omit") == [*full.get_option("run:omit"), "rankseg/_screening_cuda.py"]
