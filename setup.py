import os
import platform

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize



# Use gcc for openMP on OSX
if 'darwin' in platform.platform().lower():
    os.environ["CC"] = "gcc-4.9"
    os.environ["CXX"] = "g++-4.9"


# Declare extension
extensions = [Extension("glove_cython", ["glove_cython.pyx"],
                        extra_link_args=["-fopenmp"],
                        extra_compile_args=['-fopenmp']),
              Extension("corpus_cython", ["corpus_cython.pyx"],
                        language='C++',
                        extra_compile_args=['-std=c++11', '-O3'])]

setup(
    ext_modules=cythonize(extensions)
)
