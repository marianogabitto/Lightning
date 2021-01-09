from distutils.core import setup
import numpy as np
from Cython.Build import cythonize
from distutils.extension import Extension


use_openmp = False

extra_compiler_args = ['-ffast-math', '-march=native', '-mtune=native', '-ftree-vectorize']
extra_link_args = []
if use_openmp:
    extra_compiler_args.append("-fopenmp")
    extra_link_args.append("-fopenmp")
#    extra_link_args.append("-lomp")

setup(ext_modules=cythonize(Extension("LightC",
                            sources=["LightC.pyx",
                                      '../CXX/CounterIntEvent.cc',
                                      '../CXX/PointBase.cc',
                                      '../CXX/Point.cc',
                                      '../CXX/Center.cc',
                                      '../CXX/NeighborVisitor.cc',
                                      '../CXX/QTreeParam.cc',
                                      '../CXX/QTreeNode.cc',
                                      '../CXX/QTree.cc',
                                      '../CXX/Light.cc', ],
                            extra_compile_args=extra_compiler_args,
                            extra_link_args=extra_link_args,
                            language="c++",)),
      include_dirs=["../CXX", np.get_include()],
      library_dirs=['/usr/local/Cellar/llvm/10.0.0_3/lib'])
