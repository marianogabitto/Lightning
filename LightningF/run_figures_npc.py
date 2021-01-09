import timeit
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
from scipy.spatial.distance import cdist

import Lightning.Datasets.data as data
from Models.pdp_simple import PDpPoissGausDiagCovSimple as Model_Space
from Models.pdp_simple import PDpTPoissGausDiagCovSimpleDir as Model_Time
from sklearn.cluster import DBSCAN, OPTICS


# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# NPC-Nup107-Reis
new = data.data_import(pl=0, dataset='NPC-Nup107-Reis')

# NPC-Nup210-Reichter
# new = data.data_import(pl=1, dataset='NPC-Nup210-Reichter')

# NPC-Nup210-Sandrine
# new = data.data_import(pl=1, dataset='NPC-Nup210-Sandrine')
# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################

# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# Plotting Statistics of the Data
min_dist = np.array([])
acc_points =  new[1][0]
for i_ in np.arange(len(new[1])):
    acc_points = np.append(acc_points, new[1][i_], axis=0)
    y = cdist(new[1][i_][:, [1, 2]], new[1][i_][:, [1, 2]])
    for j_ in np.arange(y.shape[0]):
        y[j_, j_] = np.inf
    y = np.min(y, axis=0)
    min_dist = np.append(min_dist, y, axis=0)

with PdfPages('out/NPC_raw_dist.pdf') as pdf:
    plt.figure(figsize=(5, 5))
    plt.hist(min_dist, 1000)
    plt.title('Minimum Interpoint Distance. Raw Data')
    plt.xlabel('Distance')
    plt.ylabel('Counts')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

with PdfPages('out/NPC_raw_localization.pdf') as pdf:
    plt.figure(figsize=(5, 5))
    plt.hist(acc_points[:, 3], 100)
    plt.title('Localization Accuracy. Raw Data')
    plt.xlabel('Accuracy')
    plt.ylabel('Counts')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################

# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# iterate
space = list()
time = list()
for di_, dataset in enumerate(new[1][0:10]):
    print("Dataset: {}".format(di_))

    # Fit Models
    dataset[:, 3:] = (dataset[:, 3:] + 10)**2
    space.append(Model_Space(data=dataset, init_type='density', infer_pi1=False, infer_alpha0=False))
    space[di_].fit_moves(iterations=100, pl=0, prt=0, which_moves=[True, True, True, True])
    space[di_].sweep_moves(prt=True)
    space[di_].refine_clusters(prt=True)

    time.append(Model_Time(data=dataset, init_type='density', infer_pi1=False, rnk=space.Post.rnk))
    time[di_].fit_moves(iterations=100, pl=0, prt=0, which_moves=[False, False, True, True])
    time[di_].sweep_moves(prt=False)
# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################

# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# Calculating Fluorophore Distances
distances_space = list()
dis_array_space = list()
distances_time = list()
dis_array_time = list()
for i_ in np.arange(len(space)):
    print("Computing Distances Dataset:{}".format(i_))
    # Space
    mu = space[i_].Post.mu
    y = cdist(mu, mu)
    for l in np.arange(y.shape[0]):
        y[l, l] = np.inf
    y = np.min(y, axis=0)
    y = y[y > 5]
    distances_space.append(y)
    for j in np.arange(y.shape[0]):
        dis_array_space.append(y[j])
    # Time
    mu = time[i_].Post.mu
    y = cdist(mu, mu)
    for l in np.arange(y.shape[0]):
        y[l, l] = np.inf
    y = np.min(y, axis=0)
    y = y[y > 5]
    distances_time.append(y)
    for j in np.arange(y.shape[0]):
        dis_array_time.append(y[j])

with PdfPages('out/NPC_infer_dist_space.pdf') as pdf:
    plt.figure(figsize=(5, 5))
    plt.hist(dis_array_space, 50)
    plt.title('Interpoint Distance. Inference. Mean:' + np.mean(dis_array_space).__str__()[0:5])
    plt.xlabel('Distance')
    plt.ylabel('Counts')
    pdf.savefig()
    plt.close()

with PdfPages('out/NPC_infer_fluo_space.pdf') as pdf:
    multp = 3
    plt.figure(figsize=(5, 5))
    fig, ax = plt.subplots(multp, multp)
    for i in np.arange(multp):
        for j in np.arange(multp):
            space[i * multp + j].pl(ax=ax[i, j])
    pdf.savefig()
    plt.close()

with PdfPages('out/NPC_infer_convergence_space.pdf') as pdf:
    multp = 3
    plt.figure(figsize=(5, 5))
    fig, ax = plt.subplots(multp, multp)
    for i in np.arange(multp):
        for j in np.arange(multp):
            ax[i, j].plot(space[i * multp + j].elbo)
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

with PdfPages('out/NPC_infer_dist_time.pdf') as pdf:
    plt.figure(figsize=(5, 5))
    plt.hist(dis_array_time, 50)
    plt.title('Interpoint Distance. Inference. Mean:' + np.mean(dis_array_time).__str__()[0:5])
    plt.xlabel('Distance')
    plt.ylabel('Counts')
    pdf.savefig()
    plt.close()

with PdfPages('out/NPC_infer_fluo_time.pdf') as pdf:
    multp = 3
    plt.figure(figsize=(5, 5))
    fig, ax = plt.subplots(multp, multp)
    for i in np.arange(multp):
        for j in np.arange(multp):
            time[i * multp + j].pl(ax=ax[i, j])
    pdf.savefig()
    plt.close()

with PdfPages('out/NPC_infer_convergence_time.pdf') as pdf:
    multp = 3
    plt.figure(figsize=(5, 5))
    fig, ax = plt.subplots(multp, multp)
    for i in np.arange(multp):
        for j in np.arange(multp):
            ax[i, j].plot(time[i * multp + j].elbo)
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################

# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# Calculating Time Intervals
total = list()
inter = list()
on = list()
on_off = list()
for m_ in np.arange(len(space)):
    for k_ in np.arange(space[m_].Post.K):
        l = np.where(space[m_].Post.rnk[:, k_] > 0.1)
        total.append(np.max(l) - np.min(l))
        dd = np.diff(l)
        for j_ in np.arange(l[0].shape[0] - 1):
            inter.append(dd[0, j_])
        if l[0].shape[0] > 2:
            on_off.append(np.float(l[0].shape[0]) / np.float(np.max(l[0]) - np.min(l[0])))
total = np.array(total)
inter = np.array(inter)

with PdfPages('out/NPC_infer_time_length_space.pdf') as pdf:
    plt.figure(figsize=(5, 5))
    plt.hist(np.array(total), 1000)
    plt.title('Infer Fluorophore Time Duration')
    plt.xlabel('Time')
    plt.ylabel('Counts')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

with PdfPages('out/NPC_infer_inter_event_space.pdf') as pdf:
    plt.figure(figsize=(5, 5))
    plt.hist(np.array(inter), 1000)
    plt.title('Infer Time in between events')
    plt.xlabel('Time')
    plt.ylabel('Counts')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

total = list()
inter = list()
on = list()
on_off = list()
for m_ in np.arange(len(time)):
    for k_ in np.arange(time[m_].Post.K):
        l = np.where(time[m_].Post.rnk[:, k_] > 0.1)
        total.append(np.max(l) - np.min(l))
        dd = np.diff(l)
        for j_ in np.arange(l[0].shape[0] - 1):
            inter.append(dd[0, j_])
        if l[0].shape[0] > 2:
            on_off.append(np.float(l[0].shape[0]) / np.float(np.max(l[0]) - np.min(l[0])))
total = np.array(total)
inter = np.array(inter)

with PdfPages('out/NPC_infer_time_length_time.pdf') as pdf:
    plt.figure(figsize=(5, 5))
    plt.hist(np.array(total), 1000)
    plt.title('Infer Fluorophore Time Duration')
    plt.xlabel('Time')
    plt.ylabel('Counts')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

with PdfPages('out/NPC_infer_inter_event_time.pdf') as pdf:
    plt.figure(figsize=(5, 5))
    plt.hist(np.array(inter), 1000)
    plt.title('Infer Time in between events')
    plt.xlabel('Time')
    plt.ylabel('Counts')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
# ###################################################################################################

save = False
if save:
    # ###################################################################################################
    # Saving Localizations
    for i_ in np.arange(len(space)):
        filename = 'fluo' + i_.__str__() + '.txt'
        fid = open(filename, 'w')
        for x_, y_ in zip(space[i_].Post.mu[:, 0], space[i_].Post.mu[:, 1]):
            fid.write('%s\t%s\n'%(str(x_),str(y_)))
        fid.close()
    # ###################################################################################################

    # ###################################################################################################
    # Saving Times
    for i_ in np.arange(len(space)):
        filename = 'fluoTimes' + i_.__str__() + '.txt'
        fid = open(filename, 'w')
        for k_ in np.arange(space[i_].Post.K):
            k_t = np.where(space[i_].Post.rnk[:, k_] > 0.1)

            for t_ in k_t[0]:
                fid.write('%s\t%s\n'%(str(k_), str(t_)))
        fid.close()
    # ###################################################################################################
print('End')
