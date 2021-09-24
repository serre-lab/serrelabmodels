#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="serrelabmodels",
    version="0.0.23",
    author='Akash Nagaraj',
    author_email='akash_n@brown.edu',
    description='Package to import in-house models from the Serre Lab at Brown University.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/serre-lab/serrelabmodels',
    project_urls={
        "Documentation": "https://github.com/serre-lab/serrelabmodels/docs",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.6",
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
)
