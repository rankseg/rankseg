#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="rankseg",
    version="0.0.1",
    description="RankSEG: A Statistically Consistent Framework for Segmentation",
    author="Ben Dai",
    author_email="bendai@cuhk.edu.hk",
    url="https://github.com/user/project",
    install_requires=["torch", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
