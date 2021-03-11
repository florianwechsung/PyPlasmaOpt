from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import setuptools

__version__ = '0.0.1'

setup(
    name='PyPlasmaOpt',
    long_description='',
    install_requires=['sympy', 'property-manager', 'numpy', 'scipy', 'argparse', 'mpi4py', 'matplotlib', 'randomgen'],
    packages = ["pyplasmaopt"],
    package_dir = {"pyplasmaopt": "pyplasmaopt"},
    package_data={'pyplasmaopt': ['data/*', 'data/ncsx/*']},
    include_package_data=True,
    zip_safe=False,
)
