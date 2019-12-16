from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import setuptools

__version__ = '0.0.1'


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


class get_numpy_include(object):
    """Helper class to determine the numpy include path

    The purpose of this class is to postpone importing numpy
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self):
        pass

    def __str__(self):
        import numpy as np
        return np.get_include()


ext_modules = [
    Extension(
        'cppplasmaopt',
        ['cppplasmaopt/main.cpp', 'cppplasmaopt/biot_savart_all.cpp', 'cppplasmaopt/biot_savart_by_dcoilcoeff_all.cpp',
         'cppplasmaopt/biot_savart_B.cpp', 'cppplasmaopt/biot_savart_dB_by_dX.cpp', 'cppplasmaopt/biot_savart_d2B_by_dXdX.cpp',
         'cppplasmaopt/biot_savart_dB_by_dcoilcoeff.cpp', 'cppplasmaopt/biot_savart_d2B_by_dXdcoilcoeff.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_numpy_include(),
            os.path.join(sys.prefix, 'include'),
            os.path.join(sys.prefix, 'Library', 'include'),
            os.path.join('.', 'pybind11', 'include'),
            os.path.join('.', 'xtl', 'include'),
            os.path.join('.', 'xtensor', 'include'),
            os.path.join('.', 'xtensor-python', 'include'),
            os.path.join('.', 'blaze'),
        ],
        language='c++'
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++14 compiler flag  and errors when the flag is
    no available.
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    else:
        raise RuntimeError('C++14 support is required by cppplasmaopt!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        if 'CC' in os.environ and 'GCC' in os.environ['CC'].upper():
            c_opts['unix'] += ['-fopenmp']
            l_opts['unix'] += ['-fopenmp']
        else:
            darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
            c_opts['unix'] += darwin_opts
            l_opts['unix'] += darwin_opts
        

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)

setup(
    name='PlasmaOpt',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.4', 'sympy'],
    setup_requires=['pybind11>=2.4'],
    cmdclass={'build_ext': BuildExt},
    packages = ["pyplasmaopt"],
    zip_safe=False,
)
