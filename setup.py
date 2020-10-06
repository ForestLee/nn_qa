import os

import setuptools
from setuptools import setup

install_requires = [
    'pandas',
    "numpy",
    "keras",
    "tf-crf-layer",
    "tf-attention-layer",
    "tensorflow==2.2.0",
    "sklearn==0.23.2"
]


setup(
    # learn from TF how to release nightly build
    # _PKG_NAME will be used in Makefile for dev release
    name=os.getenv("_PKG_NAME", "sentence-encoding-qa"),
    version="0.1.0",
    packages=setuptools.find_packages(),
    include_package_data=True,
    url="",
    license="",
    author="",
    author_email="",
    description="sentence-encoding-qa",
    install_requires=install_requires,
)
