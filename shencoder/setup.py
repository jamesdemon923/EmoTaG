import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
_src_path = os.path.dirname(os.path.abspath(__file__))
nvcc_flags = ['-O3', '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']
if os.name == 'posix':
    c_flags = ['-O3', '-std=c++14']
elif os.name == 'nt':
    c_flags = ['/O2', '/std:c++17']

    def find_cl_path():
        import glob
        for edition in ['Enterprise', 'Professional', 'BuildTools', 'Community']:
            paths = sorted(glob.glob('C:\\\\Program Files (x86)\\\\Microsoft Visual Studio\\\\*\\\\%s\\\\VC\\\\Tools\\\\MSVC\\\\*\\\\bin\\\\Hostx64\\\\x64' % edition), reverse=True)
            if paths:
                return paths[0]
    if os.system('where cl.exe >nul 2>nul') != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError('Could not locate a supported Microsoft Visual C++ installation')
        os.environ['PATH'] += ';' + cl_path
if torch.__version__.startswith('2'):
    nvcc_flags[1] = '-std=c++17'
    c_flags[1] = '-std=c++17'
setup(name='shencoder', ext_modules=[CUDAExtension(name='_shencoder', sources=[os.path.join(_src_path, 'src', f) for f in ['shencoder.cu', 'bindings.cpp']], extra_compile_args={'cxx': c_flags, 'nvcc': nvcc_flags})], cmdclass={'build_ext': BuildExtension})
