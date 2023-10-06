#!/usr/bin/env python
from gettext import find
from setuptools import setup, find_packages
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    package_dir={'': 'spot-vegitation-navigation'},
    packages=find_packages(where='spot-vegitation-navigation', exclude=('siamese_network/models', 'siamese_network/aug_demo_models')),
)

setup(**setup_args)