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

A particular configuration in my MAC computer installs the code by executing:
 * CC=/usr/local/opt/llvm/bin/clang-10 python setup.py build_ext --inplace

Troubleshooting Note on MAC:
- On Mac Os X El Capitan. First, I installed Homebrew. Next, "brew install llvm", "brew install gcc". Finally,
  the xcode command line tools should be installed.
- On Mac Os X Catalina. the xcode command line tools 12 give an error when trying to compile the libraries. For this reason,
  I ended installing xcode command line tools version 11.5 .

