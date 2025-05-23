from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "attention",
        ["attention.pyx"],
        extra_compile_args=['-fopenmp', '-mavx2', '-mfma', '-O3'],
        extra_link_args=['-fopenmp'],
        include_dirs=[np.get_include()]
    )
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()]
) 