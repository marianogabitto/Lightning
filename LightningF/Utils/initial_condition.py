import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree as Tree
from scipy.sparse.csgraph import connected_components as cc

from .NumericUtil import convert_to_n0, e_log_n, inplace_exp_normalize_rows_numpy as e_norm
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# #####################################################
# #####################################################
# #####################################################
# #####################################################
# DEFINE PRIORS AND CALL INIT ROUTINES
def priors(x, mu0=None, sigma02=None, alpha0=None, log_p0=None, aij0=None,
           pi1=None, gamma1=None, gamma2=None, a=None, b=None, gamma0=None):
    # Compute Priors if not given
    area = (np.max(x[:, 0]) - np.min(x[:, 0])) * \
           (np.max(x[:, 1]) - np.min(x[:, 1]))
    if mu0 is None:
        mu0 = np.mean(x, axis=0)
    if sigma02 is None:
        sigma02 = area ** 2. * np.ones(x.shape[1])
    if log_p0 is None:
        log_p0 = - 1.5 * np.log(area)  # np.min([-10, - np.log(1.5 * area)])
    if pi1 is None:
        pi1 = 0.1
    if a is None:
        a = 1.
    if b is None:
        b = 10.
    if gamma0 is None:
        gamma0 = 100.
    if aij0 is None:
        aij0 = np.ones((2, 3))
        aij0[1, 2] = 0

    # DP Priors. alpha0, parameter of the gem distribution.
    #            gamma1 and gamma2 hyper-priors (mean=100, std=100).
    if alpha0 is None:
        alpha0 = 10.
    if gamma1 is None:
        gamma1 = 1.
    if gamma2 is None:
        gamma2 = 0.01

    return mu0, sigma02, alpha0, log_p0, pi1, a, b, gamma1, gamma2, gamma0, aij0


def initiliaze(x, sigma2_x, init_type, alpha0, condition=None, prt=False, post=None):
    # Display State of the Algorithm
    if prt:
        print("Calculating Initial Conditions: " + init_type)

    # Compute Initial mu, sigma and alpha
    if init_type == 'post':
        if post is None:
            print("Post should be passed as a valid structure")
            raise SystemExit
        else:
            mu_init = post.mu
            sigma2_init = post.sigma2
            eta1_init = post.eta1
            eta0_init = post.eta0
            k0 = post.C

    elif init_type == 'density':

        mu_init, mu_count = density(x)
        k0 = mu_init.shape[0]

        avg_sigma = np.mean(sigma2_x)
        sigma2_init = avg_sigma * np.ones((k0, x.shape[1]))

        eta1_init = 1 + mu_count
        eta0_init = alpha0 + convert_to_n0(mu_count)

    elif init_type == 'rl_cluster':
        mu_init, mu_count = rl_cluster(x, condition=condition)
        k0 = mu_init.shape[0]

        avg_sigma = np.mean(sigma2_x)
        sigma2_init = avg_sigma * np.ones((k0, x.shape[1]))

        eta1_init = 1 + mu_count
        eta0_init = alpha0 + convert_to_n0(mu_count)

    elif init_type == 'points':
        k0 = x.shape[0]
        mu_init = x

        sigma2_init = 5. * np.ones(x.shape)  # (avg_sigma / 10.) * np.ones(x.shape)

        eta1_init = 1 + np.ones(k0)
        eta0_init = alpha0 + convert_to_n0(np.ones(k0))

    elif init_type == 'points_refine':
        mu_init, count = init_points(points=x)
        k0 = mu_init.shape[0]

        avg_sigma = np.mean(sigma2_x)
        sigma2_init = avg_sigma * np.ones((k0, x.shape[1]))

        eta1_init = 1 + count
        eta0_init = alpha0 + convert_to_n0(count)
    else:
        raise NotImplementedError('Unrecognized initname ' + init_type +
                                  '. Available: density, rl_cluster, points, points_refine.')

    # Display State of the Algorithm
    if prt:
        print("Initial Condition Calculations Finished")

    return mu_init, sigma2_init, eta0_init, eta1_init, k0


# #####################################################
# #####################################################
# #####################################################
# #####################################################
# INITIAL CONDITIONS ROUTINES
# #####################################################
# ROUTINE: REFINE_POINTS
def init_points(points, merge_distance=20):
    # Merging Close-by Points
    p_qt = Tree(points)
    taken = -1 * np.ones(points.shape[0], dtype=np.int32)
    for i_, p_ in enumerate(points):
        if i_ % 250000 == 0:
            print('Processing:', i_, ' datapoints')
        idx = p_qt.query_ball_point(p_, merge_distance)
        if len(idx) > 1:
            d = cdist(p_.reshape(1, 2), points[idx, :])
            pp = np.where(d == d[d > 0].min())[1][0]
            if taken[i_] == -1:
                if taken[idx[pp]] == -1:
                    taken[idx[pp]] = i_
                    taken[i_] = i_
                else:
                    taken[i_] = taken[idx[pp]]
            elif taken[i_] > -1:
                if taken[idx[pp]] == -1:
                    taken[idx[pp]] = taken[i_]
                else:
                    # merge centers
                    center1 = taken[idx[pp]]
                    center2 = taken[i_]
                    if not (center1 == center2):
                        taken[taken == center2] = center1
                        taken[center2] = center1

    # Calculating Centers
    sort_index = np.argsort(taken)
    taken = taken[sort_index]
    sort_points = points[sort_index, :]
    unique_elements, index, mu_count = np.unique(taken, return_counts=True, return_index=True)
    mu_count = mu_count[1:]
    k = unique_elements.shape[0] - 1
    mu = np.zeros((k, 2))
    for i_ in np.arange(index.shape[0] - 2):
        mu[i_, :] = np.mean(sort_points[index[i_ + 1]:index[i_ + 2], :], axis=0)
    mu[-1, :] = np.mean(sort_points[index[-1]:, :], axis=0)

    return mu, mu_count


# ############################################
# ROUTINE: R-L CLUSTER
def rl_cluster(x, evaluate_dataset=0, gaus_kernel=None, condition=None):
    # Init Parameters
    if condition is None:
        condition = 5.
    if gaus_kernel is None:
        gaus_kernel = 35

    # Evaluate dataset or run with hard-coded conditions
    rho, dist = rho_delta(x, gaus_kernel)
    if evaluate_dataset == 1:
        plt.plot(rho, dist, '.', markersize=0.1)
        plt.pause(10.)

    # Compute Assignments
    mu, mu_count = rl_center_calc(x, dist, condition)

    return mu, mu_count


def rho_delta(x, gk):
    m = x.shape[0]
    data = Tree(x)
    rho = np.zeros(m, dtype=np.float16)

    # Compute Rho
    for i in range(m):
        points = data.query_ball_point(x[i, :], gk)
        rho[i] = np.max([len(points), 0])
    ordrho = np.argsort(rho)[::-1]

    # Compute delta
    delta = np.zeros(m, dtype=np.float16)
    x_sort = x[ordrho, :]
    data_srt = Tree(x_sort)
    for i in np.arange(1, m):
        points = np.array(data_srt.query_ball_point(x_sort[i, :], 3 * gk))
        points = points[points < i]
        min_d = 5 * gk
        if len(points):
            d = np.sqrt(np.min(np.sum((x_sort[points] - x_sort[i]) ** 2, axis=1)))
            if d < min_d:
                min_d = d
        delta[i] = min_d
    delta[0] = np.max(delta)

    # Re-shuffle delta correctly
    out_delta = np.zeros(m, dtype=np.float16)
    out_delta[ordrho] = delta

    return rho, out_delta


def rl_center_calc(x, dist, condition):
    # Find Clusters based on delta condition
    cluster_centers = np.array(np.where(dist > condition)[0])
    mu = x[cluster_centers, :]
    mu_count = np.ones(mu.shape[0])

    return mu, mu_count


# ############################################
# ROUTINE: DENSITY
def density(x):
    # parameters
    n = x.shape[0]
    p_r = 50
    p_t = 5

    data = Tree(x)
    dist = np.zeros(n)
    for i in np.arange(n):
        points = data.query_ball_point(x[i, :], p_r)
        dist[i] = len(points) > p_t

    keep_points = np.where(dist == True)[0]
    remove_points = np.where(dist == False)[0]
    assignment = np.arange(n)
    for n_ in keep_points:
        # decide root
        root = assignment[n_]
        points = data.query_ball_point(x[n_, :], p_t)
        assignment[points] = root
    assignment[remove_points] = -1

    sort_index = np.argsort(assignment)
    assignment = assignment[sort_index]
    sort_points = x[sort_index, :]
    unique_elements, index, mu_count = np.unique(assignment, return_counts=True, return_index=True)
    mu_count = mu_count[1:]
    k = unique_elements.shape[0] - 1
    if k > 2:
        mu = np.zeros((k, 2))
        for i_ in np.arange(index.shape[0] - 2):
            mu[i_, :] = np.mean(sort_points[index[i_ + 1]:index[i_ + 2], :], axis=0)
        mu[-1, :] = np.mean(sort_points[index[-1]:, :], axis=0)
    else:
        print("Failed Initial Condition")
        raise SystemExit

    return mu, mu_count
# #####################################################
# #####################################################
# #####################################################


# #####################################################
# #####################################################
# #####################################################
# BIRTH PROPOSAL ROUTINES WITH HANDLE
def handle_birth(noise_points, noise_index, gap_routine):
    # Compute Parameters
    noise_n = len(noise_points)

    # 1) Look for High Density Points
    idx_points = np.array(refine_high_density(noise_points, noise_n))
    gap, evaluate, params = gap_routine(noise_index[idx_points])
    evaluate=True # TODO:remove
    # 2) KMeans++.
    if not evaluate:
        idx_points = refine_kmeanpp(noise_points, noise_n)
        gap, evaluate, params = gap_routine(noise_index[idx_points])

    # 3) Try Density Cluster.
    if not evaluate:
        idx_points, k0 = refine_density(noise_points, noise_n)
        if k0 != 0:
            gap, evaluate, params = gap_routine(noise_index[idx_points])

    return gap, evaluate, params


def refine_high_density(points, noise_n, rad=15):
    p_qt = Tree(points)
    max_len = 0
    max_idx = []
    for n_ in np.arange(noise_n):
        idx_points = p_qt.query_ball_point(points[n_], rad)
        if len(idx_points) > max_len:
            max_idx = idx_points
            max_len = len(idx_points)

    return max_idx


def refine_kmeanpp(points, noise_n):
    # Parameters
    if noise_n < 10:
        k0 = 2
    else:
        k0 = 5

    # Compute k-means++
    kmeans = KMeans(n_clusters=k0, init='k-means++').fit(points)

    # Count Points in labels
    n_labels = np.zeros(kmeans.n_clusters)
    for k_ in np.arange(kmeans.n_clusters):
        n_labels[k_] = np.sum(kmeans.labels_ == k_)

    # Assign Points to cluster
    idx_points = kmeans.labels_ == np.argmax(n_labels)

    return idx_points


def refine_density(points, noise_n):
    # Check that the calculation fits in memory
    if noise_n > 20000:
        return np.zeros(0), 0

    # Parameters
    p_r = 100
    p_t = 5

    # Cluster Calculation
    y = cdist(points, points)
    dist = np.sum(y < p_r, axis=0) > p_t
    idx = np.where(dist == True)[0]  # This step might send some points to noise
    points = y[np.ix_(idx, idx)] < p_t
    n_comp, clist = cc(points)

    # If there are zero clusters, return empty
    if n_comp < 1:
        return np.zeros(0), 0
    # Else return the biggest cluster
    else:
        # Count Points in labels
        n_labels = np.zeros(n_comp)
        for n_ in np.arange(n_comp):
            n_labels[n_] = np.sum(clist == n_)

        # Assign Points to cluster
        idx_points = clist == np.argmax(n_labels)

        return idx[idx_points], n_comp
# #####################################################
# #####################################################
# #####################################################


# #####################################################
# #####################################################
# #####################################################
# SPLIT PROPOSAL ROUTINE WITH HANDLE
def handle_split(params, split_gap, propagate_routine):
    """

    :param params: Structure having the following data:
                         k_split:
    :param split_gap: Routine to evaluate the gap of the new configuration.
    :param propagate_routine: Given center locations, routine that propagate the model forward.
    :return:
    """

    # 1) Try Kmeans as a mean to divide the cluster
    new_rnk = split_kmeans(params, propagate_routine)
    gap, evaluate, out_param = split_gap(k=params['k_split'], new_rnk=new_rnk)

    # 2) Try Centers at different angles
    if not evaluate:
        for ang_ in np.arange(0, 2 * np.pi, 0.25 * np.pi):
            new_rnk = split_circular(params, propagate_routine, ang=ang_)
            gap, evaluate, out_param = split_gap(k=params['k_split'], new_rnk=new_rnk)

    return gap, evaluate, out_param


def split_kmeans(params, propagate):
    # Compute k-means++
    kmeans = KMeans(n_clusters=2, init='k-means++').fit(params['points'])

    # Validate Labels
    n_labels = np.array([np.sum(kmeans.labels_ == k_) for k_ in np.arange(2)])
    if (kmeans.n_clusters < 2) or (np.any(n_labels == 0)):
        return None
    new_rnk = propagate(params, mus=kmeans.cluster_centers_)

    return new_rnk


def split_circular(params, propagate, ang):
    points = params["points"]
    # Determine bounding circle
    mean_x = np.mean(points, axis=0)
    rad = np.max(np.abs(points - mean_x[None, :]))

    # Given bounding circle, propose to clusters at a certain angle
    mu1 = mean_x + 0.5 * rad * np.array([np.cos(ang), np.sin(ang)])
    mu2 = mean_x + 0.5 * rad * np.array([np.cos(ang + np.pi), np.sin(ang + np.pi)])

    # Compute new rnk based on new mus
    new_rnk = propagate(params, mus=np.vstack([mu1, mu2]))

    return new_rnk
# #####################################################
# #####################################################
# #####################################################
