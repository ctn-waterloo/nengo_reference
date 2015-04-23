#!/usr/bin/env python
import imp
import os

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import find_packages, setup  # noqa: F811

root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    'version', os.path.join(root, 'nengo_reference', 'version.py'))
description = "Reference backend for nengo."
with open(os.path.join(root, 'README.rst')) as readme:
    long_description = readme.read()

url = "https://github.com/ctn-waterloo/nengo_reference"
setup(
    name="nengo_reference",
    version=version_module.version,
    author="CNRGlab at UWaterloo",
    author_email="celiasmith@uwaterloo.ca",
    packages=find_packages(),
    include_package_data=True,
    scripts=[],
    url=url,
    license=url + "/blob/master/LICENSE.md",
    description=description,
    install_requires=[
        "nengo",
    ],
)
