import glob
import os
import platform
import subprocess
import sys

from setuptools import Command, Extension, setup
from setuptools.command.test import test as TestCommand


def define_extensions():

    compile_args = ['-fopenmp',
                    '-ffast-math']

    # There are problems with illegal ASM instructions
    # when using the Anaconda distribution (at least on OSX).
    # This could be because Anaconda uses its own assembler?
    # To work around this we do not add -march=native if we
    # know we're dealing with Anaconda
    if 'anaconda' not in sys.version.lower():
        compile_args.append('-march=native')

    return [Extension("glove.glove_cython", ["glove/glove_cython.c"],
                      extra_link_args=["-fopenmp"],
                      extra_compile_args=compile_args),
            Extension("glove.metrics.accuracy_cython",
                      ["glove/metrics/accuracy_cython.c"],
                      extra_link_args=["-fopenmp"],
                      extra_compile_args=compile_args),
            Extension("glove.corpus_cython", ["glove/corpus_cython.cpp"],
                      language='C++',
                      libraries=["stdc++"],
                      extra_link_args=['-std=c++11'] + compile_args,
                      extra_compile_args=['-std=c++11'] + compile_args)]


def set_gcc():
    """
    Try to find and use GCC on OSX for OpenMP support.
    """

    # For macports and homebrew
    patterns = ['/opt/local/bin/gcc-mp-[0-9].[0-9]',
                '/opt/local/bin/gcc-mp-[0-9]',
                '/usr/local/bin/gcc-[0-9].[0-9]',
                '/usr/local/bin/gcc-[0-9]']

    if 'darwin' in platform.platform().lower():

        gcc_binaries = []
        for pattern in patterns:
            gcc_binaries += glob.glob(pattern)
        gcc_binaries.sort()

        if gcc_binaries:
            _, gcc = os.path.split(gcc_binaries[-1])
            os.environ["CC"] = gcc

        else:
            raise Exception('No GCC available. Install gcc from Homebrew '
                            'using brew install gcc.')


class Cythonize(Command):
    """
    Compile the extension .pyx files.
    """

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        import Cython
        from Cython.Build import cythonize

        cythonize(define_extensions())


class Clean(Command):
    """
    Clean build files.
    """

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        pth = os.path.dirname(os.path.abspath(__file__))

        subprocess.call(['rm', '-rf', os.path.join(pth, 'build')])
        subprocess.call(['rm', '-rf', os.path.join(pth, '*.egg-info')])
        subprocess.call(['find', pth, '-name', '*.pyc', '-type', 'f', '-delete'])
        subprocess.call(['rm', os.path.join(pth, 'glove', 'corpus_cython.so')])
        subprocess.call(['rm', os.path.join(pth, 'glove', 'glove_cython.so')])


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ['tests/']

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name='glove',
    version='0.0.1',
    description=('Python implementation of Global Vectors '
                 'for Word Representation (GloVe)'),
    long_description='',
    packages=["glove"],
    install_requires=['numpy', 'scipy'],
    tests_require=['pytest'],
    cmdclass={'test': PyTest, 'cythonize': Cythonize, 'clean': Clean},
    author='Maciej Kula',
    url='https://github.com/maciejkula/glove-python',
    license='Apache 2.0',
    classifiers=['Development Status :: 3 - Alpha'],
    ext_modules=define_extensions()
)
