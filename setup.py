#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This script provides setup requirements to install hrvanalysis via pip"""

import setuptools

# Get long description in READ.md file
with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="ecg_qc_viz",
    version="v1.0-b1",
    author="Alexis COMTE",
    license="GPLv3",
    author_email="alexis.g.comte@gmail.com",
    description="a streamlit app to vizualise ecg signal for signal quality prediction research",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="https://github.com/alexisgcomte",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        "ecg-qc>=1.0b3",
        "plotly>=4.14.1",
        "streamlit>=0.74.1"
    ],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ]
)
