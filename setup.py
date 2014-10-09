import os
import platform

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize



# Use gcc for openMP on OSX
if 'darwin' in platform.platform().lower():
    os.environ["CC"] = "gcc-4.9"
    os.environ["CXX"] = "gcc-4.9"


# Declare extension
extensions = [Extension("*", ["*.pyx"],
                        extra_link_args=["-fopenmp"],
                        extra_compile_args=['-fopenmp'])]

setup(
    ext_modules=cythonize(extensions)
)
