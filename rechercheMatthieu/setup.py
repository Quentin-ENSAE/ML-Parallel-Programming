from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "attention_parallel",
        ["attention_parallel.pyx"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
        include_dirs=[np.get_include()]
    )
]

setup(
    name="attention_parallel",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    include_dirs=[np.get_include()]
) 