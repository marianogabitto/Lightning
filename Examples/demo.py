import timeit
import numpy as np
from scipy import io as sio
import matplotlib
matplotlib.use("TkAgg")
from timeit import default_timer as timer
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

# PYTHON MODELS
from LightningF.Models.pdp_simple import TimeIndepentModelPython as ModelSpacePy

# C MODELS
from LightningF.Models.pdp_simple import TimeIndepentModelC as ModelSpaceC

# ###################################################################################################
# Import data
data = sio.loadmat('../LightningF/Datasets/temp.pkl.mat')
new = data['accu_detect_particles'][0, :]
# ###################################################################################################

# ###################################################################################################
# Fitting Datasets
modelpy = list()
modelc = list()
for i_ in np.arange(new.shape[0]):
    new[i_][:, 3] = new[i_][:, 3] ** 2
    new[i_][:, 4] = new[i_][:, 4] ** 2
    # Create Model
    modelpy.append(ModelSpacePy(data=new[i_], init_type='rl_cluster', infer_pi1=False, infer_alpha0=True))
    modelc.append(ModelSpaceC(data=new[i_], init_type='rl_cluster', infer_pi1=False, infer_alpha0=True))

    # Inference on Model
    st = timer()
    modelpy[i_].fit(iterations=100, pl=0, prt=1)
    print("TimePySp:{}".format(timer() - st))
    st = timer()
    modelc[i_].fit(iterations=100, pl=0, prt=1)
    print("TimeCSp:{}".format(timer() - st))
    modelpy[i_].fit_moves(iterations=100, pl=0, prt=0, which_moves=[True, True, True, True])
    modelpy[i_].print_number_moves()
    modelc[i_].fit_moves(iterations=100, pl=0, prt=0, which_moves=[True, True, True, True])
    modelc[i_].print_number_moves()
    modelpy[i_].pl_bl()
    modelc[i_].pl_bl()
# ###################################################################################################

print("Finish")
