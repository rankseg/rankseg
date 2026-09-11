import re
import runpy
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup_metadata(monkeypatch):
    # Execute only setup.py's metadata generation, without installing anything
    # or requiring setuptools to be a runtime/test dependency.
    captured = {}
    setuptools = types.ModuleType("setuptools")
    setuptools.setup = lambda **kwargs: captured.update(kwargs)
    setuptools.find_packages = lambda **kwargs: ["rankseg", "rankseg.integration"]
    monkeypatch.setitem(sys.modules, "setuptools", setuptools)
    namespace = runpy.run_path(str(ROOT / "setup.py"))
    return namespace, captured


@pytest.mark.parametrize("version", ["0.0.6", "0.1.0"])
@pytest.mark.parametrize("quote", ['"', "'"])
def test_pypi_description_uses_release_pinned_light_fallback(setup_metadata, version, quote):
    namespace, _ = setup_metadata
    readme = (
        "<picture>\n"
        f"<source media={quote}(prefers-color-scheme: dark){quote} srcset={quote}./fig/dark.png{quote}>\n"
        f"<img src={quote}./fig/light.png{quote} alt={quote}Example{quote} width={quote}100%{quote}>\n"
        "</picture>"
    )
    actual = namespace["_pypi_description"](readme, version)
    assert actual == (
        f"<img src={quote}https://raw.githubusercontent.com/rankseg/rankseg/v{version}/fig/light.png{quote} "
        f"alt={quote}Example{quote} width={quote}100%{quote}>"
    )


def test_pypi_description_converts_inline_links_and_html_attributes(setup_metadata):
    namespace, _ = setup_metadata
    convert = namespace["_pypi_description"]
    page = "https://github.com/rankseg/rankseg/blob/v0.0.6/"
    raw = "https://raw.githubusercontent.com/rankseg/rankseg/v0.0.6/"
    readme = (
        '[Example](./examples/demo.py#L12 "Example")\n'
        '![Plot](./fig/plot.png "Plot")\n'
        "<A HREF='./notebooks/demo.ipynb'>Notebook</A>\n"
        '<IMG SRC="./fig/plot.png" alt="Plot">'
    )
    assert convert(readme, "0.0.6") == (
        f'[Example]({page}examples/demo.py#L12 "Example")\n'
        f'![Plot]({raw}fig/plot.png "Plot")\n'
        f"<A HREF='{page}notebooks/demo.ipynb'>Notebook</A>\n"
        f'<IMG SRC="{raw}fig/plot.png" alt="Plot">'
    )


def test_pypi_description_preserves_external_links_anchors_and_fenced_examples(setup_metadata):
    namespace, _ = setup_metadata
    readme = (
        "[Docs](https://example.org/guide) [Section](#section)\n"
        '<a href="https://example.org">External</a>\n'
        "```markdown\n[Example](./example.py)\n"
        '<picture><img src="./image.png"></picture>\n```\n'
        "```python\nprobs = model_logits.softmax(dim=1)\n```\n"
    )
    assert namespace["_pypi_description"](readme, "0.0.6") == readme


def test_pypi_description_rejects_picture_without_fallback(setup_metadata):
    namespace, _ = setup_metadata
    with pytest.raises(ValueError, match="fallback img"):
        namespace["_pypi_description"]('<picture><source srcset="./image.png"></picture>', "0.0.6")


def test_setup_description_preserves_readme_and_code_examples(setup_metadata):
    namespace, metadata = setup_metadata
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    description = metadata["long_description"]
    version = namespace["__version__"]
    assert metadata["version"] == version
    assert metadata["long_description_content_type"] == "text/markdown"
    assert "<picture>" in readme
    assert "<picture" not in description and "<source" not in description
    assert 'src="./' not in description and "](./" not in description
    for filename in ("monai_pancreas_rankseg.png", "benchmark_results.png"):
        assert f"https://raw.githubusercontent.com/rankseg/rankseg/v{version}/fig/{filename}" in description
    assert f"https://github.com/rankseg/rankseg/blob/v{version}/examples/pytorch_native_rankseg.py" in description
    assert re.findall(r"```.*?```", description, re.DOTALL) == re.findall(r"```.*?```", readme, re.DOTALL)
    assert (ROOT / "README.md").read_text(encoding="utf-8") == readme
