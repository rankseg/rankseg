import os
import re

from setuptools import find_packages, setup

__version__ = "0.0.6"


def _pypi_description(readme, version):
    """Adapt this repository's README without changing its GitHub presentation."""
    raw_base = f"https://raw.githubusercontent.com/rankseg/rankseg/v{version}/"
    page_base = f"https://github.com/rankseg/rankseg/blob/v{version}/"

    def picture_fallback(match):
        fallback = re.search(r"<img\b[^>]*>", match.group(0), flags=re.IGNORECASE)
        if fallback is None:
            raise ValueError("README picture must contain a fallback img for PyPI")
        return fallback.group(0)

    def html_url(match):
        tag = match.group(0)
        is_image = match.group(1).lower() == "img"
        attribute = "src" if is_image else "href"
        base = raw_base if is_image else page_base
        return re.sub(
            rf"(\s{attribute}\s*=\s*)([\"'])\./([^\"']+)\2",
            lambda attr: f"{attr[1]}{attr[2]}{base}{attr[3]}{attr[2]}",
            tag,
            flags=re.IGNORECASE,
        )

    def markdown_url(match):
        base = raw_base if match.group(1).startswith("!") else page_base
        return match.group(1) + base + match.group(2)

    # This README uses inline Markdown links and HTML pictures. Leave fenced
    # examples verbatim; this is intentionally not a general Markdown parser.
    parts = re.split(r"(^```[^\n]*\n.*?^```[ \t]*$)", readme, flags=re.MULTILINE | re.DOTALL)
    for index in range(0, len(parts), 2):
        part = re.sub(r"<picture\b[^>]*>.*?</picture\s*>", picture_fallback, parts[index], flags=re.IGNORECASE | re.DOTALL)
        part = re.sub(r"<(img|a)\b[^>]*>", html_url, part, flags=re.IGNORECASE)
        parts[index] = re.sub(r"(!?\[[^\]\n]*\]\()\./([^\s)]+)", markdown_url, part)
    return "".join(parts)


# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = _pypi_description(f.read(), __version__)

setup(
    name="rankseg",
    version=__version__,
    author="Ben Dai, Zixun Wang",
    author_email="bendai@cuhk.edu.hk",
    url="https://rankseg.readthedocs.io/en/latest/",
    description="RankSEG: A Statistically Consistent Segmentation Prediction Solver for Dice and IoU Metrics Optimization",
    license_expression="BSD-3-Clause",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["rankseg", "rankseg.*"]),
    install_requires=["torch>=2.0.0", "scipy", "numpy"],
    extras_require={
        "dev": ["pytest", "pytest-cov", "pre-commit", "commitizen", "torchmetrics"],
        "test": ["pytest", "pytest-cov", "torchmetrics"],
    },
    zip_safe=False,
    python_requires=">= 3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Operating System :: OS Independent",
    ],
    keywords="segmentation, deep-learning, pytorch, computer-vision, dice-loss, iou, rankseg",
    project_urls={
        "Documentation": "https://rankseg.readthedocs.io/en/latest/",
        "Source": "https://github.com/rankseg/rankseg",
        "Tracker": "https://github.com/rankseg/rankseg/issues",
    },
)
