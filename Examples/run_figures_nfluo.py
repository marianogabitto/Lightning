import timeit
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle

import LightningF.Datasets.data as data
from LightningF.Models.pdp_simple import TimeIndepentModelC as ModelSpaceC
from LightningF.Models.pdp_simple import TimeIndepentModelPython as ModelSpacePy
from sklearn.cluster import DBSCAN, OPTICS


# ###########################################################
# DNA ORIGAMI PLATFORM - 2 FLUO - DISTANCE, NOISE
new = data.data_import(pl=0, dataset='Origami-AF647')

# Iterating Structures
ite_separation = np.arange(40, 17.5, -2.5)
n_separation = len(ite_separation)
ite_nfluo = np.arange(3, 4, 1)
n_nfluo = len(ite_nfluo)
n_samples = 2
noise_level = 0.5

elbo = np.zeros((n_nfluo, n_separation, 3, n_samples))
distance_space = np.zeros((n_nfluo, n_separation, 2, n_samples))
distance_time = np.zeros((n_nfluo, n_separation, 2, n_samples))
distance_optics = np.zeros((n_nfluo, n_separation, 2, n_samples))
n_clust_space = np.zeros((n_nfluo, n_separation, n_samples))
n_clust_time = np.zeros((n_nfluo, n_separation, n_samples))
n_clust_optics = np.zeros((n_nfluo, n_separation, n_samples))

# iterate
for sepi_, sep_ in enumerate(ite_separation):
    for ni_, n_ in enumerate(ite_nfluo):
        for s_ in np.arange(0, n_samples):
            # Create Datasets
            dataset, true_labels, true_location = data.create_nfluo(dist=sep_, fluos=new, n=n_, noise=noise_level, pl=0,
                                                                    plcircles=0, seed=s_)
            dataset[:, 3:] = dataset[:, 3:]/10

            """
            if (sepi_ == 0) and (ni_ == 0) and (s_ == 0):
                plot_dataset = True
            else:
                plot_dataset = False
            if plot_dataset:
                colorsd = np.mod(np.argmax(true_labels, axis=1), 6)
                fig, ax = plt.subplots(figsize=(8, 13))
                ax.scatter(dataset[:, 1], dataset[:, 2], c=colorsd, marker=".")
                ax.plot(true_location[:, 0], true_location[:, 1], 'r+')
                pp = PdfPages('out/NFluo_configuration_TRUE.pdf')
                pp.savefig(fig)
                pp.close()
                plt.close(fig)
            """
            # Fit Models
            print("Fit Dataset. Dist:{}, Number:{}, Noise:{}, Seed:{}".format(sep_, n_**2, noise_level, s_))
            # spacePy = ModelSpacePy(data=dataset, init_type='density', infer_pi1=True, infer_alpha0=True, prt=0)
            # spacePy.fit(iterations=200, pl=0, prt=True)
            spaceC = ModelSpaceC(data=dataset, init_type='density', infer_pi1=True, infer_alpha0=True, prt=0)
            spaceC.fit(iterations=200, pl=0, prt=True)
            spaceC.fit_moves(iterations=100, pl=0, prt=True, which_moves=[False, True, False, True])

            space = ModelSpacePy(data=dataset, infer_pi1=False, infer_alpha0=False, prt=0, init_type='post',
                                post=spaceQT.Post, a=spaceQT.Post.a, b=spaceQT.Post.b,
                                gamma1=spaceQT.Post.gamma1, gamma2=spaceQT.Post.gamma2)
            space.fit(iterations=10, pl=0, prt=True)
            space.refine_clusters(prt=True, which_moves=np.array([False, True, False, True]), update=True)
            space.fit_moves(iterations=100, pl=0, prt=True, which_moves=[False, True, False, True])
raise SystemExit

for _ in 1:
            space.sweep_moves(prt=True)
            space.fit(iterations=100, pl=0, prt=True, empty=True)
            space.refine_clusters(prt=True)
            time = Model_Time(data=dataset, init_type='rl_cluster', infer_pi1=True, rnk=space.Post.rnk)
            time.fit_moves(iterations=100, pl=0, prt=False, which_moves=[False, False, True, True])
            time.sweep_moves(prt=True)

            # Plot Individual Figure
            if (sepi_ == 0) and (ni_ == 0) and (s_ == 0):
                plot_figure = True
            else:
                plot_figure = False
            if plot_figure:
                fig, ax = plt.subplots(figsize=(8, 13))
                colors = np.concatenate([['k+'], np.tile(['g.', 'r.', 'b.', 'y.', 'c.'], 100)], axis=0)
                for klass, color in zip(range(0, space.Post.K), colors):
                    Xk = dataset[space.Post.rnk[:, klass] > 0.1, 1:3]
                    ax.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
                    ax.set_title('Space\nLIGHTNING')
                pp = PdfPages('out/NFluo_configuration_space.pdf')
                pp.savefig(fig)
                pp.close()
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(8, 13))
                colors = np.concatenate([['k+'], np.tile(['g.', 'r.', 'b.', 'y.', 'c.'], 100)], axis=0)
                for klass, color in zip(range(0, time.Post.K), colors):
                    Xk = dataset[time.Post.rnk[:, klass] > 0.1, 1:3]
                    ax.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
                    ax.set_title('Time\nLIGHTNING')
                pp = PdfPages('out/NFluo_configuration_time.pdf')
                pp.savefig(fig)
                pp.close()
                plt.close(fig)

            # Number of Clusters
            n_clust_space[ni_, sepi_, s_] = space.Post.C
            n_clust_time[ni_, sepi_, s_] = time.Post.C
            print("NTrue:{}, Distance:{}, NSp:{}, NT:{}, SAMPLE NUMBER:{}".format(n_**2, sep_,
                                                                                  n_clust_space[ni_, sepi_, s_],
                                                                                  n_clust_time[ni_, sepi_, s_],
                                                                                  s_))
            # Calculate Distance to Centers
            distance_space[ni_, sepi_, :, s_] = data.metric_distance(space.Post.mu, true_location)
            distance_time[ni_, sepi_, :, s_] = data.metric_distance(time.Post.mu, true_location)
            print(" ")
# ##################################################################################################################

"""
# ##################################################################################################################
# PLOT SPACE RESULTS
for separation in np.arange(n_separation):
    fig, ax = plt.subplots(figsize=(8, 13))
    plt.title("Number of Fluorophores Space. Distance:{}, Noise Level:{}".format(ite_separation[separation], noise_level))
    ax.errorbar(ite_nfluo**2, np.mean(n_clust_space[:, separation, :], axis=1), yerr=np.std(n_clust_space[:, separation, :],
                                                                                            axis=1), fmt="+")
    plt.show()
    pp = PdfPages('out/NFluo_n_space_' + separation.__str__() + '.pdf')
    pp.savefig(fig)
    pp.close()

    fig, ax = plt.subplots(figsize=(8, 13))
    plt.title("Center Distance Space. Distance:{}, Noise Level:{}".format(ite_separation[separation], noise_level))
    ax.errorbar(ite_nfluo**2, np.mean(distance_space[:, separation, 0, :], axis=1),
                yerr=np.std(distance_space[:, separation, 0, :], axis=1), fmt="+")
    pp = PdfPages('out/NFluo_center_distance_space_' + separation.__str__() + '.pdf')
    pp.savefig(fig)
    pp.close()

    # PLOT TIME RESULTS
    fig, ax = plt.subplots(figsize=(8, 13))
    plt.title("Number of Fluorophores Time. Distance:{}, Noise Level:{}".format(ite_separation[separation], noise_level))
    ax.errorbar(ite_nfluo**2, np.mean(n_clust_time[:, separation, :], axis=1), yerr=np.std(n_clust_time[:, separation, :],
                                                                                            axis=1), fmt="+")
    plt.show()
    pp = PdfPages('out/NFluo_n_time_' + separation.__str__() + '.pdf')
    pp.savefig(fig)
    pp.close()

    fig, ax = plt.subplots(figsize=(8, 13))
    plt.title("Center Distance Time. Distance:{}, Noise Level:{}".format(ite_separation[separation], noise_level))
    ax.errorbar(ite_nfluo**2, np.mean(distance_time[:, separation, 0, :], axis=1),
                yerr=np.std(distance_time[:, separation, 0, :], axis=1), fmt="+")
    pp = PdfPages('out/NFluo_center_distance_time_' + separation.__str__() + '.pdf')
    pp.savefig(fig)
    pp.close()
"""
print("Finish")
# ##################################################################################################################
# ##################################################################################################################
# ##################################################################################################################
