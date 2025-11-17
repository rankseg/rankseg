from setuptools import setup

__version__ = "0.0.1"

setup(
    name="rankseg",
    version=__version__,
    author=["Ben Dai", "Zixun Wang"],
    author_email="bendai@cuhk.edu.hk",
    url="https://rankseg.readthedocs.io/en/latest/",
    description="RankSEG: A Statistically Consistent Segmentation Solver for Dice and IoU Metrics Optimization",
    packages=["rankseg"],
    install_requires=["torch", "scipy", "numpy"],
    zip_safe=False,
    python_requires=">= 3.9",
)