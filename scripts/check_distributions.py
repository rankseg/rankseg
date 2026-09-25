"""Check packaged Python sources, release README, and sdist test dependencies.

Requires readme_renderer[md]; used by the distribution-building CI job.
"""

import argparse
import tarfile
from email.parser import Parser
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []
        self.urls = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        attribute = {"a": "href", "img": "src"}.get(tag)
        if attribute and attribute in attrs:
            self.urls.append(attrs[attribute])
            if tag == "img":
                self.images.append(attrs[attribute])


def check_package_sources(root, wheel, sdist, prefix):
    """Reject missing, stale, or unexpected package modules in either artifact."""
    sources = {
        source.relative_to(root).as_posix(): source.read_bytes()
        for source in sorted((root / "rankseg").rglob("*.py"))
    }
    assert sources, "No RankSEG Python sources found"
    for artifact, names, read in (
        ("wheel", wheel.namelist(), wheel.read),
        ("sdist", [name.removeprefix(prefix) for name in sdist.getnames()],
         lambda name: sdist.extractfile(prefix + name).read()),
    ):
        packaged = {name for name in names if name.startswith("rankseg/") and name.endswith(".py")}
        assert packaged == sources.keys(), (
            f"Package modules differ in {artifact}: "
            f"missing={sorted(sources.keys() - packaged)}, unexpected={sorted(packaged - sources.keys())}"
        )
        for name, content in sources.items():
            assert read(name) == content, f"Stale package module in {artifact}: {name}"


def check_distributions(dist_dir):
    from readme_renderer.markdown import render

    root = Path(__file__).resolve().parents[1]
    wheels = list(dist_dir.glob("*.whl"))
    sdists = list(dist_dir.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1, "Use a directory containing exactly one wheel and one sdist"

    with ZipFile(wheels[0]) as wheel:
        metadata_files = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        assert len(metadata_files) == 1
        metadata = Parser().parsestr(wheel.read(metadata_files[0]).decode("utf-8"))
    version = metadata["Version"]
    description = metadata.get_payload()
    assert "cuda" in metadata.get_all("Provides-Extra", []), "Missing CUDA extra in wheel metadata"
    assert metadata["Description-Content-Type"] == "text/markdown"
    assert "<picture" not in description.lower() and "<source" not in description.lower()

    html = render(description)
    assert html is not None, "Install readme_renderer[md] to validate Markdown rendering"
    rendered = _Links()
    rendered.feed(html)
    for url in rendered.urls:
        assert url.startswith(("#", "//")) or urlsplit(url).scheme, f"Relative URL in PyPI description: {url}"

    # Every local README image must survive rendering as a release-pinned URL.
    original = _Links()
    original.feed((root / "README.md").read_text(encoding="utf-8"))
    for url in original.images:
        if url.startswith("./"):
            expected = f"https://raw.githubusercontent.com/rankseg/rankseg/v{version}/{url[2:]}"
            assert expected in rendered.images, f"Missing release image: {expected}"

    with tarfile.open(sdists[0], "r:gz") as sdist, ZipFile(wheels[0]) as wheel:
        prefix = f"rankseg-{version}/"
        check_package_sources(root, wheel, sdist, prefix)
        required = [
            root / "pyproject.toml",
            root / ".coveragerc.cpu",
            root / "scripts/benchmark_rma_screening.py",
            root / "scripts/check_distributions.py",
            root / "scripts/smoke_test_distribution.py",
            root / "tests/conftest.py",
            *sorted((root / "tests").glob("test*.py")),
        ]
        for source in [root / "README.md", root / "setup.py", *required]:
            name = prefix + source.relative_to(root).as_posix()
            assert name in sdist.getnames(), f"Missing from sdist: {name}"
            assert sdist.extractfile(name).read() == source.read_bytes(), f"Stale file in sdist: {name}"
        source_metadata = Parser().parsestr(sdist.extractfile(prefix + "PKG-INFO").read().decode("utf-8"))
        assert source_metadata["Version"] == version
        assert source_metadata.get_payload() == description, "Wheel and sdist descriptions differ"
        for field in ("Requires-Dist", "Provides-Extra"):
            assert sorted(source_metadata.get_all(field, [])) == sorted(metadata.get_all(field, [])), (
                f"Wheel and sdist {field} differ"
            )

    print(f"RankSEG {version}: package sources, PyPI README, release image URLs, and sdist test files verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path)
    check_distributions(parser.parse_args().dist_dir)
