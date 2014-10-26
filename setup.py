import os
import platform

from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize


# Use gcc for openMP on OSX
if 'darwin' in platform.platform().lower():
    os.environ["CC"] = "gcc-4.9"
    os.environ["CXX"] = "g++-4.9"


# Declare extension
extensions = [Extension("glove_cython", ["glove/glove_cython.pyx"],
                        extra_link_args=["-fopenmp"],
                        extra_compile_args=['-fopenmp']),
              Extension("corpus_cython", ["glove/corpus_cython.pyx"],
                        language='C++',
                        extra_compile_args=['-std=c++11', '-O3'])]

setup(
    name='glove',
    version='0.0.1',
    description=('Python implementation of Global Vectors '
                 'for Word Representation (GloVe)'),
    long_description='',
    packages=['glove'],
    install_requires=['numpy',
                      'cython',
                      'scipy'],
    author='Maciej Kula',
    url='https://github.com/maciejkula/glove-python',
    license='Apache 2.0',
    classifiers=['Development Status :: 3 - Alpha'],
    ext_modules=cythonize(extensions)
)
