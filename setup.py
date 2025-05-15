import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    name="mmat",
    sources=["mmat.pyx", "mmat_impl.cpp"],
    include_dirs=[numpy.get_include()],
    language="c++",
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    extra_compile_args=["-O3", "-fopenmp", "-mavx2"],
    extra_link_args=["-fopenmp"],
)

setup(name="mmat", ext_modules=cythonize([ext], language_level="3"))
