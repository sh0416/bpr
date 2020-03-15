from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vsl_cuda',
    ext_modules=[
        CUDAExtension('vsl_cuda', [
            'vsl_cuda.cpp',
            'vsl_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })