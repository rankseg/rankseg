from setuptools import setup
import os

__version__ = "0.0.2"

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="rankseg",
    version=__version__,
    author=["Ben Dai", "Zixun Wang"],
    author_email="bendai@cuhk.edu.hk",
    url="https://rankseg.readthedocs.io/en/latest/",
    description="RankSEG: A Statistically Consistent Segmentation Prediction Solver for Dice and IoU Metrics Optimization",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=["rankseg"],
    install_requires=["torch", "scipy", "numpy"],
    zip_safe=False,
    python_requires=">= 3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="segmentation, deep-learning, pytorch, computer-vision, dice-loss, iou, rankseg",
    project_urls={
        "Documentation": "https://rankseg.readthedocs.io/en/latest/",
        "Source": "https://github.com/rankseg/rankseg",
        "Tracker": "https://github.com/rankseg/rankseg/issues",
    },
)

# python -m build
# twine upload --skip-existing dist/*