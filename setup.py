from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

setup(
    name="models_cla",
    packages=find_packages(),
    ext_modules=cythonize("models_cla/randomforest.pyx"),
    include_dirs=[np.get_include()]
)