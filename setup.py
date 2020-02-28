from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='vsl_op',
      ext_modules=[cpp_extension.CppExtension('vsl_op', ['vsl_op.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
