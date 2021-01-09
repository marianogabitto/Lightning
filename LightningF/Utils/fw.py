import multiprocessing
import numpy as np

import ctypes
import ctypes.util
from numpy.ctypeslib import ndpointer
import os
import sys
import sysconfig


# ################################################################
# Fwd-Bwd Methods
def fw_bw(lpi_init, lpi_mat, log_soft_ev):
    init = np.exp(lpi_init)
    mat = np.exp(lpi_mat)

    log_soft_ev = np.asarray(log_soft_ev, dtype=np.float64)
    lognorm_c = np.max(log_soft_ev, axis=1)
    log_soft_ev = log_soft_ev - lognorm_c[:, np.newaxis]
    soft_ev = np.exp(log_soft_ev)

    resp, resp_pair, lmarg_pr_seq = FwdBwdAlg_cpp(init, mat, soft_ev)

    return resp, resp_pair, lmarg_pr_seq + lognorm_c.sum()


def FwdBwdAlg_cpp(initPi, transPi, SoftEv, order='C'):
    if not hasEigenLibReady:
        raise ValueError("Cannot find library %s. Please recompile."
                         % (libfilename))
    if order != 'C':
        raise NotImplementedError("LibFwdBwd only supports row-major order.")

    T, K = SoftEv.shape
    # Prep inputs
    initPi = np.asarray(initPi, order=order)
    transPi = np.asarray(transPi, order=order)
    SoftEv = np.asarray(SoftEv, order=order)

    # Allocate outputs
    resp = np.zeros((T, K), order=order)
    resp_pair = np.zeros((K, K), order=order)
    marg_pr_seq = np.zeros((1, 1), order=order)

    # Execute C++ code (fills in outputs in-place)
    lib.FwdBwdAlg(initPi, transPi, SoftEv, resp, resp_pair, marg_pr_seq, K, T)

    return resp, resp_pair, marg_pr_seq


def search_paths_for_file(filename, pathlist):
    for path in pathlist:
        candidate_file = os.path.join(path, filename)
        if os.path.isfile(candidate_file):
            return candidate_file


libpath = os.path.dirname(os.path.abspath(__file__))
libfilename = 'libfwdbwdcpp.so'
hasEigenLibReady = True

try:
    lib = ctypes.cdll.LoadLibrary(os.path.join(libpath, libfilename))
    # print('Found C++ Core:', libpath, libfilename)
    # library_name = 'libfwdbwdcpp' + sysconfig.get_config_var('EXT_SUFFIX')
    # library_path = search_paths_for_file(library_name, sys.path)
    # lib = ctypes.cdll.LoadLibrary(library_path)
    lib.FwdBwdAlg.restype = None
    lib.FwdBwdAlg.argtypes = \
        [ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ndpointer(ctypes.c_double),
         ctypes.c_int, ctypes.c_int]

except OSError:
    # No compiled C++ library exists
    print("Failed to Load Cpp Core")
    hasEigenLibReady = False

