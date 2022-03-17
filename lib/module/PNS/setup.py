from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from os.path import join

project_root = 'PNS_Module'
sources = [join(project_root, file) for file in ['sa_ext.cpp',
                                                 'sa.cu','reference.cpp']]


nvcc_args = [
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_70,code=compute_70'
]
cxx_args = ['-std=c++11']

setup(
    name='self_cuda',
    ext_modules=[
        CUDAExtension('self_cuda_backend',
                      sources, extra_compile_args={'cxx': cxx_args,'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
