from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='vsl_cpp',
      ext_modules=[cpp_extension.CppExtension('vsl_cpp', ['vsl.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})