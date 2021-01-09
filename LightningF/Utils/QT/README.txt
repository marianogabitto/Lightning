There are two folders:
   CXX/     contains all C++ code, does not assume/depend on python
   python/  contains the python wrapper to the C++ code in CXX


Building the python wrapper:
 * There is NO need to build the CXX directory first, the python builder will build the needed parts
 * cython needs to be installed first
 * Building it in place for trying examples:
      python setup.py build_ext --inplace
   (ignore warning about using deprecated NumPy API)
   This builds LightC.so, which is the python module
 * Running the test after building
      ./test1.py

A particular configuration in my computer installs the code by executing:
 * CC=/usr/local/opt/llvm/bin/clang-10 python setup.py build_ext --inplace