"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from setuptools import setup, find_packages

setup(
    name="abstention-bench",
    packages=find_packages(exclude=["tests"]),
)
