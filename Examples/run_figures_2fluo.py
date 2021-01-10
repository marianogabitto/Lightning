import timeit
import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle

import Lightning.Datasets.data as data
from Models.pdp_simple import PDpPoissGausDiagCovSimple as Model_Space
from Models.pdp_simple import PDpTPoissGausDiagCovSimpleDir as Model_Time
from sklearn.cluster import DBSCAN, OPTICS


# ##################################################################################################################
# ##################################################################################################################
# ##################################################################################################################
# DNA ORIGAMI PLATFORM - 2 FLUO - DISTANCE, NOISE
new = data.data_import(pl=0, dataset='Origami-AF647')

# Iterating Structures
ite_noise = np.arange(0.1, 2.6, 0.5)
n_noise = len(ite_noise)
ite_dist = np.arange(40, 20, -2.5)
n_dist = len(ite_dist)
n_samples = 4

elbo = np.zeros((n_noise, n_dist, 3, n_samples))
distance_space = np.zeros((n_noise, n_dist, 2, n_samples))
distance_time = np.zeros((n_noise, n_dist, 2, n_samples))
distance_optics = np.zeros((n_noise, n_dist, 2, n_samples))
n_clust_space = np.zeros((n_noise, n_dist, n_samples))
n_clust_time = np.zeros((n_noise, n_dist, n_samples))
n_clust_optics = np.zeros((n_noise, n_dist, n_samples))

found_space = np.zeros((n_noise, n_dist, n_samples))
found_time = np.zeros((n_noise, n_dist, n_samples))

# iterate
for ni_, n_ in enumerate(ite_noise):
    for di_, d_ in enumerate(ite_dist):
        for s_ in np.arange(0, n_samples):
            # Create Datasets
            # dataset, true_labels = data.create_twofluo(dist=d_, fluos=new, noise=n_, pl=0, plcircles=0, seed=s_)
            dataset, true_labels, loc = data.create_nfluo(dist=d_, fluos=new, n=3, noise=n_, pl=0, plcircles=0,
                                                          seed=s_)

            # Calculate True Configuration
            space_true = Model_Space(data=dataset, rnk=true_labels, infer_pi1=True, infer_alpha0=True, prt=False)
            space_true.fit(iterations=100, pl=0, prt=False, empty=False)
            space_true_elbo = space_true.calc_elbo()
            space_true.merge_components(1, 2)
            space_true.fit(iterations=10, empty=False)
            space_true.split_move()

            # Fit Models
            space = Model_Space(data=dataset, init_type='density', infer_pi1=True, infer_alpha0=True, prt=False)
            space.fit(iterations=100, pl=0, prt=False)
            space.fit_moves(iterations=100, pl=0, which_moves=[True, True, True, True], prt=False)
            space.sweep_moves(prt=False, iterations=10)
            space.refine_clusters(prt=False)
            space.load_configuration()

            if space.Post.K == 10:
                found_space[ni_, di_, s_] = 1
            elif np.sum(space.calc_elbo()) > space_true_elbo:
                found_space[ni_, di_, s_] = -1

            # Calculate True Configuration
            time_true = Model_Space(data=dataset, rnk=true_labels, infer_pi1=True, infer_alpha0=True, prt=False)
            time_true.fit(iterations=100, pl=0, prt=False, empty=False)
            time_true_elbo = time_true.calc_elbo()

            time = Model_Time(data=dataset, init_type='density', infer_pi1=True, rnk=space.Post.rnk)
            time.fit(prt=False, iterations=10)
            time.sweep_moves(prt=False)
            time.load_configuration()

            if time.Post.K == 10:
                found_time[ni_, di_, s_] = 1
            elif np.sum(time.calc_elbo()) > time_true_elbo:
                found_time[ni_, di_, s_] = -1

            """
            # Fit Best DBSCAN
            min_v = 1e5
            min_l = []
            for eps_ in np.arange(0.5, 5, 0.5):
                for min_s in np.arange(1, 10):
                    cl_dbscan = DBSCAN(eps=eps_, min_samples=min_s).fit(dataset[:, 1:3])
                    n_c = len(np.unique(cl_dbscan.labels_))
                    if np.abs(n_c - 3) < min_v:
                        min_v = np.abs(n_c - 3)
                        min_l = [eps_, min_s]
            cl_dbscan = DBSCAN(eps=min_l[0], min_samples=min_l[1]).fit(dataset[:, 1:3])

            # Fit OPTICS
            cl_optics = OPTICS(xi=.05, eps=min_l[0], min_samples=min_l[1])
            cl_optics.fit(dataset[:, 1:3])

            # Plot Individual Figure
            plot_figure = False
            if plot_figure:
                plt.figure(figsize=(10, 7))
                ax = plt.subplot(131)
                colors = ['g.', 'r.', 'b.', 'y.', 'c.']
                for klass, color in zip(range(0, 5), colors):
                    Xk = dataset[cl_optics.labels_ == klass, 1:3]
                    ax.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
                    ax.plot(dataset[cl_optics.labels_ == -1, 1], dataset[cl_optics.labels_ == -1, 2], 'k+', alpha=0.1)
                    ax.set_title('Automatic Clustering\nOPTICS')

                ax = plt.subplot(132)
                colors = ['g.', 'r.', 'b.', 'y.', 'c.']
                for klass, color in zip(range(0, 5), colors):
                    Xk = dataset[cl_dbscan.labels_ == klass, 1:3]
                    ax.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
                    ax.plot(dataset[cl_dbscan.labels_ == -1, 1], dataset[cl_dbscan.labels_ == -1, 2], 'k+', alpha=0.1)
                    ax.set_title('Best Clustering\nDBSCAN')

                ax = plt.subplot(133)
                colors = ['k+', 'g.', 'r.', 'b.', 'y.', 'c.']
                for klass, color in zip(range(0, np.min([5, space[s_].Post.K])), colors):
                    Xk = dataset[space[s_].Post.rnk[:, klass] > 0.5, 1:3]
                    ax.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
                    ax.set_title('Space\nLIGHTNING')
            """
            # Number of Clusters
            n_clust_space[ni_, di_, s_] = space.Post.K
            n_clust_time[ni_, di_, s_] = time.Post.K
            # n_clust_optics[ni_, di_, s_] = len(np.unique(cl_optics.labels_))
            print("Noise:{}, Distance:{}, NSp:{}, NT:{}, NOpt:{}, SAMPLE NUMBER:{}".format(n_, d_,
                                                                                           n_clust_space[ni_, di_, s_],
                                                                                           n_clust_time[ni_, di_, s_],
                                                                                           n_clust_optics[ni_, di_, s_],
                                                                                           s_))
            """
            # Calculate Distance to Centers
            distance_space[ni_, di_, 0, s_] = np.min(np.linalg.norm(space.Post.mu, axis=0))
            distance_space[ni_, di_, 1, s_] = np.min(np.linalg.norm(space.Post.mu - np.array([d_, 0])[None, :], axis=0))
            distance_time[ni_, di_, 0, s_] = np.min(np.linalg.norm(time.Post.mu, axis=0))
            distance_time[ni_, di_, 1, s_] = np.min(np.linalg.norm(time.Post.mu - np.array([d_, 0])[None, :], axis=0))
            optics_mu = np.array([np.mean(dataset[cl_optics.labels_ == c_, 1:3], axis=0) for c_ in
                                  np.unique(cl_optics.labels_) if len(dataset[cl_optics.labels_ == c_, 1:3]) > 0])
            distance_optics[ni_, di_, 0, s_] = np.min(np.linalg.norm(optics_mu, axis=0))
            distance_optics[ni_, di_, 1, s_] = np.min(np.linalg.norm(optics_mu - np.array([d_, 0])[None, :], axis=0))

            # Compare ELBO when Considering the Correct/Wrong Model
            if space.Post.K == 1:
                elbo[ni_, di_, 0, s_] = space.calc_elbo()
                space.Post.rnk = true_labels
                space.vb_update_global()
                elbo[ni_, di_, 2, s_] = space.calc_elbo()
                space.merge_component(1, 2)
                elbo[ni_, di_, 1, s_] = space.calc_elbo()
                space.delete_component(1)
                elbo[ni_, di_, 0, s_] = space.calc_elbo()
            elif space.Post.K == 2:
                elbo[ni_, di_, 1, s_] = space.calc_elbo()
                space.split_component(1, true_labels[:, 1:])
                elbo[ni_, di_, 2, s_] = space.calc_elbo()
                space.delete_component(2)
                space.delete_component(1)
                elbo[ni_, di_, 0, s_] = space.calc_elbo()
            elif space.Post.K == 3:
                elbo[ni_, di_, 2, s_] = space.calc_elbo()
                space.merge_components(1, 2)
                elbo[ni_, di_, 1, s_] = space.calc_elbo()
                space.delete_component(1)
                elbo[ni_, di_, 0, s_] = space.calc_elbo()
            else:
                while space.Post.K > 3:
                    space.delete_component(space.Post.C)
                space.Post.rnk = true_labels
                space.vb_update_global()
                elbo[ni_, di_, 2, s_] = space.calc_elbo()
                space.merge_components(1, 2)
                elbo[ni_, di_, 1, s_] = space.calc_elbo()
                space.delete_component(1)
                elbo[ni_, di_, 0, s_] = space.calc_elbo()
            """
# ##################################################################################################################
# ##################################################################################################################
# ##################################################################################################################

# ##################################################################################################################
# ##################################################################################################################
# ##################################################################################################################
# PLOT RESULTS
# N_clusters Calculations
xi = np.repeat(ite_noise, n_dist)
yi = np.tile(ite_dist, n_noise)
zi_space = np.mean(n_clust_space, axis=2).flatten()
zi_time = np.mean(n_clust_time, axis=2).flatten()
zi_optics = np.mean(n_clust_optics, axis=2).flatten()

fig = plt.figure(figsize=(8, 3))
ax3 = fig.add_axes([0.05, 0.15, 0.9, 0.15])
cmap = mpl.colors.ListedColormap([[0., .8, 1.], [0., .5, 0.5], [0., .5, 0.], [.5, .5, 0.], [.9, .9, 0.], [1., .4, 0.]])
cmap.set_over((1., 0., 0.))
cmap.set_under((0., 0., 1.))
bounds = [1.5, 2, 2.70, 3.3, 4., 4.5, 5.5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb3 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, norm=norm, boundaries=[-10] + bounds + [10], extend='both',
                                extendfrac='auto', ticks=bounds, spacing='uniform', orientation='horizontal')
plt.show()
pp = PdfPages('out/2Fluo_N_clusters_colorbar.pdf')
pp.savefig(fig)
pp.close()

fig, ax = plt.subplots(figsize=(8, 13))
plt.title("Number of Fluorophores. Noise Level vs Fluorophore Distance")
ax.scatter(yi, xi + 0.1, s=50, c=zi_space, cmap=cmap, norm=norm)
ax.scatter(yi, xi, s=50, c=zi_time, cmap=cmap, norm=norm)
ax.scatter(yi, xi - 0.1,  s=50, c=zi_optics, cmap=cmap, norm=norm)
plt.show()
pp = PdfPages('out/2Fluo_N_clusters.pdf')
pp.savefig(fig)
pp.close()

fig, ax = plt.subplots(figsize=(8, 13))
plt.title("Elbo")
elbo_gap = elbo[:, :, 2, :] - elbo[:, :, 1, :]
plt.plot(ite_dist, np.mean(elbo_gap[0, :, :], axis=1))
plt.show()
pp = PdfPages('out/2Fluo_Elbo.pdf')
pp.savefig(fig)
pp.close()

fig, ax1 = plt.subplots(figsize=(8, 13))
plt.title("Center Distance")
d_space = np.mean(distance_space[0, :, :, :], axis=2)
d_time = np.mean(distance_time[0, :, :, :], axis=2)
d_optics = np.mean(distance_optics[0, :, :, :], axis=2)
ax1.errorbar(ite_dist, np.mean(d_space, axis=1), yerr=np.std(d_space, axis=1)/d_space.shape[1], label="Space")
ax1.errorbar(ite_dist, np.mean(d_time, axis=1), yerr=np.std(d_time, axis=1)/d_time.shape[1], label="Time")
ax1.errorbar(ite_dist, np.mean(d_optics, axis=1), yerr=np.std(d_optics, axis=1)/d_optics.shape[1], label="Optics")
ax1.legend()
pp = PdfPages('out/2Fluo_Center_distance.pdf')
pp.savefig(fig)
pp.close()
# ##################################################################################################################
# ##################################################################################################################
# ##################################################################################################################
