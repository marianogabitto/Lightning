import sys
import copy
import numpy as np
from sklearn.cluster import KMeans
from scipy.special import psi, gammaln, digamma
from abc import ABCMeta, abstractmethod
from scipy.spatial import cKDTree as Tree
from ..Utils.fw import fw_bw
from ..Utils.QT.python import LightC as LightC

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ..Utils.ParamBag3 import ParamBag
from ..Utils.initial_condition import initiliaze, priors, handle_birth, handle_split

from ..Utils.NumericUtil import convert_to_n0, e_log_beta, e_log_n, inplace_exp_normalize_rows_numpy as e_norm
from ..Utils.NumericUtil import dotatb, calc_beta_expectations, calc_entropy, Lalloc
from ..Utils.NumericUtil import c_h, c_alpha, c_beta, e_gamma, elog_gamma, c_dir, c_gamma, delta_c, c_obs


CLT_BELONGING = 0.4
dtarget_min_count = 1.5
debug = False


# Abstract class model
class AbstractModel(object):
    __meta__ = ABCMeta

    def __init__(self):
        self.Post = []
        self.elbo = np.array(())
        self.move_counter = np.zeros((4, 2))

        # Save Best Configuration Explored
        self.best_configuration = []
        self.best_elbo = - np.inf

    # #########################################################################################################
    # Abstract Methods
    @abstractmethod
    def vb_update(self, empty=False, prt=False, iteration=-1):
        raise NotImplementedError

    @abstractmethod
    def find_nearby_centers(self, target_radius):
        raise NotImplementedError

    @abstractmethod
    def propagate(self, params, mus):
        return None

    @abstractmethod
    def get_rnk_k(self, k):
        raise NotImplementedError
    # #########################################################################################################

    # #########################################################################################################
    # Fitting Methods
    def fit(self, prt=False, pl=False, iterations=100, empty=True, circles=True, tolerance=1e-10):

        # Init ELBO
        ax = None
        self.elbo = np.zeros([iterations])
        self.vb_update(empty=False, prt=prt, iteration=0)
        self.elbo[0] = self.calc_elbo()

        # Iterating
        if prt:
            print("Iterating")
        for i_ in np.arange(1, iterations):
            self.vb_update(empty=empty, prt=prt, iteration=i_)
            self.elbo[i_] = self.calc_elbo()

            # Save Best Configuration
            if np.sum(self.elbo[i_]) > self.best_elbo:
                self.best_elbo = np.sum(self.elbo[i_])
                self.save_configuration()

            # Check Elbo
            if np.abs((self.elbo[i_] - self.elbo[i_ - 1])) < tolerance:
                break
            if (self.elbo[i_] - self.elbo[i_ - 1]) < - 0.1:
                print("ELBO INCREASED BY:{}".format(self.elbo[i_] - self.elbo[i_ - 1]))

            # Print, Plot
            if prt and (i_ % 1 == 0):
                self.print_iteration(elbo=self.elbo[i_], iteration=i_, components=self.Post.K)
            if pl:
                ax = self.pl_bl(ax=ax, circles=circles)

        if prt:
            print("Finish Looping")

    def fit_moves(self, iterations=100, pl=False, prt=False, tolerance=1e-5,
                  which_moves=np.array([True, True, True, True])):
        # Init ELBO
        ax = None
        self.elbo = np.zeros([iterations])
        self.vb_update(empty=False, prt=prt, iteration=0)
        self.elbo[0] = self.calc_elbo()

        # Iterating
        if prt:
            print("Iterating")
        clusters = -1
        for i_ in np.arange(1, iterations):
            # Updates
            if ((self.elbo[i_ - 1] - self.elbo[i_ - 2]) > tolerance) or (self.Post.K != clusters):
                clusters = self.Post.K
                self.vb_update(empty=1, prt=prt, iteration=i_)
                self.elbo[i_] = self.calc_elbo()
            else:
                self.elbo[i_] = self.elbo[i_ - 1]

            # Save Best Configuration
            if np.sum(self.elbo[i_]) > self.best_elbo:
                self.best_elbo = np.sum(self.elbo[i_])
                self.save_configuration()

            # Print
            if prt:
                self.print_iteration(elbo=self.elbo[i_], iteration=i_, components=self.Post.K)

            # Moves
            self.moves(iteration=i_, which=which_moves, prt=prt)
            if pl:
                ax = self.pl_bl(ax=ax, circles=1)

        if prt:
            print("Finish Looping")

    def sweep_moves(self, prt=False, which_moves=np.array([True, True, True, True]), iterations=None):
        # Sweeping
        if iterations is None:
            iterations = int(self.Post.K / 2)

        if prt:
            print("Sweeping through Clusters")
            self.print_iteration(elbo=self.elbo[-1], iteration=-1, components=self.Post.K)
        self.elbo = np.hstack([self.elbo, 0])

        for _ in np.arange(iterations):
            if which_moves[0]:
                self.birth_move(iteration=-1)
            if which_moves[2]:
                self.split_move(iteration=-1)
            if which_moves[3]:
                self.merge_move(iteration=-1)
            if which_moves[1]:
                self.dead_move(iteration=-1)
            self.redundant(prt=prt, iteration=-1)
            self.vb_update(empty=1, prt=prt)

        # Wrap Up
        self.vb_update(empty=1, prt=prt)
        self.elbo[-1] = self.calc_elbo()

        # Save Best Configuration
        if np.sum(self.elbo[-1]) > self.best_elbo:
            self.best_elbo = np.sum(self.elbo[-1])
            self.save_configuration()

        if prt:
            self.print_iteration(elbo=self.elbo[-1], iteration=-1, components=self.Post.K)
            print("Finish Sweeping")

    def refine_clusters(self, prt=False, which_moves=np.array([True, True, True, True]), update=True):
        """
        Sweeps a number of times through the moves trying to find a better optima.
        :param prt: Print Elbo after Sweeping.
        :param which_moves: Boolean Vector. Birth, Dead, Split, Merge.
        :param update:
        :return: -
        """
        # Refining
        if prt:
            print("Refining through Clusters")
            self.print_iteration(elbo=self.elbo[-1], iteration=-1, components=self.Post.K)
        self.elbo = np.hstack([self.elbo, 0])

        for _ in np.arange(int(self.Post.C / 10)):
            if which_moves[0]:
                self.birth_move(iteration=-1)

        i_ = 0
        while i_ < self.Post.C:
            if which_moves[2]:
                self.split_move(iteration=-1, k_split=i_ + 1)
            i_ += 1

        i_ = 0
        while i_ < self.Post.C:
            if which_moves[3]:
                self.merge_move(iteration=-1, merge_1=i_)
            i_ += 1
        i_ = 0

        while i_ < self.Post.C:
            if which_moves[1]:
                self.dead_move(iteration=-1, random=i_ + 1)
            i_ += 1

        # Wrap Up
        self.redundant(prt=prt, iteration=-1)
        if update:
            self.vb_update(empty=1, prt=prt)
        self.elbo[-1] = self.calc_elbo()
        if prt:
            self.print_iteration(elbo=self.elbo[-1], iteration=-1, components=self.Post.K)
            print("Finish Sweeping")
    # #########################################################################################################

    # #########################################################################################################
    # COMPONENT MODIFICATIONS
    def delete_components(self, comp):
        self.Post.removeComps(comp, noise=True)

    def add_component(self, params):
        self.Post.insert1Comp_fromNoise(params)

    def merge_components(self, comp1, comp2, params):
        self.Post.mergeComps(comp1, comp2, params)

    def split_component(self, comp, params):
        self.Post.splitComp(comp, params)
    # #########################################################################################################

    # #########################################################################################################
    # MOVES
    def moves(self, iteration, which=np.array([True, True, True, True]), prt=False):

        # Birth Move
        if which[0]:
            birth_coordinator = 10
            birth_start = 15
            if ((iteration + birth_start) % birth_coordinator) == 0:
                self.birth_move(iteration=iteration, prt=prt)

        # Delete Move
        if which[1]:
            del_coordinator = 10
            del_start = 13
            if ((iteration + del_start) % del_coordinator) == 0:
                self.dead_move(iteration=iteration, prt=prt)

        # Split Move
        if which[2]:
            split_coordinator = 10
            split_start = 5
            if ((iteration + split_start) % split_coordinator) == 0:
                self.split_move(iteration=iteration, prt=prt)

        # Merge Move
        if which[3]:
            merge_coordinator = 10
            merge_start = 8
            if ((iteration + merge_start) % merge_coordinator) == 0:
                self.merge_move(iteration=iteration, prt=prt)

    def empty(self, prt=0, iteration=-1):

        # First Plan Components to erase
        empty_components = np.sort(np.flatnonzero(self.Post.rk < dtarget_min_count) + 1)[::-1]
        if empty_components.shape[0] > 0:
            self.Post.removeComps(empty_components, noise=True)
            if prt == 1:
                self.print_iteration(iteration=iteration, message='Empty Move. Delete Component:{}'.
                                     format(empty_components))
            self.vb_update(empty=0)

    def redundant(self, prt=0, iteration=-1):
        # CHECK REDUNDANT CLUSTERS
        target_radius = 1.

        # First Plan Components to erase
        if self.Post.C > 1:
            centers = self.find_nearby_centers(target_radius)
            if len(centers) > 1:
                self.delete_components(centers)

                if prt == 1:
                    self.print_iteration(iteration=iteration,
                                         message='Redundant Move. Delete Components:{}'.format(centers[:-1]))
                self.vb_update_global()

    def birth_move(self, iteration=-1, prt=0):
        # #################################################################
        # Parameters
        thres_noise = 0.5

        # Identify Noise points
        noise_index = np.flatnonzero(self.Post.rn0_vector > thres_noise)
        noise_n = noise_index.shape[0]

        # If there are more than 5 points. Try to search for clusters
        if noise_n > 5:
            gap, evaluate, params = handle_birth(noise_points=self.x[noise_index], noise_index=noise_index,
                                                 gap_routine=self.gap_birth)
            # if improved, update best_model:
            if debug:
                evaluate = True
                self.pl()
                plt.plot(self.x[noise_index, 0], self.x[noise_index, 1], '.')
                plt.plot(self.x[params["idx"], 0], self.x[params["idx"], 1], '+')
                plt.plot(params['mu'][0, 0], params['mu'][0, 1], 'or')
                plt.show()

            if evaluate:
                self.move_counter[0, 0] += 1
                if debug:
                    old_elbo = self.calc_elbo(deb=True)
                self.add_component(params)
                if debug:
                    new_elbo = self.calc_elbo(deb=True)
                    print("Old ELbo:{}".format(old_elbo))
                    print("New ELbo:{}".format(new_elbo))
                    print("True Gap:{}".format(new_elbo - old_elbo))
                    print("Estimated Gap:{}".format(gap))
                    print("EstimatedGap:{}    TrueGap{}".format(np.sum(gap), np.sum(new_elbo - old_elbo)))
                    self.pl_bl()
                    plt.show()
                self.vb_update(empty=False)
                if prt == 1:
                    self.print_iteration(iteration=iteration, message='Birth Move. Add 1 Component')
            else:
                self.move_counter[0, 1] += 1

    def dead_move(self, random=None, prt=0, iteration=-1):

        # Select Component to erase
        if (random is None) and (self.Post.K > 1):
            choices = np.arange(1, self.Post.K)
            random_component = np.random.choice(choices)
        else:
            random_component = random

        if random_component is not None:
            gap, evaluate = self.gap_delete(random_component)

            # if improved, update best_model:
            if debug:
                evaluate = True
            if evaluate:
                self.move_counter[1, 0] += 1
                if debug:
                    old_elbo = self.calc_elbo(deb=True)
                self.delete_components(random_component)
                if debug:
                    new_elbo = self.calc_elbo(deb=True)
                    print("Old ELbo:{}".format(old_elbo))
                    print("New ELbo:{}".format(new_elbo))
                    print("True Gap:{}".format(new_elbo - old_elbo))
                    print("Estimated Gap:{}".format(gap))
                    print("EstimatedGap:{}    TrueGap{}".format(np.sum(gap), np.sum(new_elbo - old_elbo)))
                if prt == 1:
                    self.print_iteration(iteration=iteration,
                                         message='Dead Move. Delete Component:{}'.format(random_component))
            else:
                self.move_counter[1, 1] += 1

    def merge_move(self, prt=0, iteration=-1, merge_1=None):
        # Parameters
        target_radius = 4 * self.mean_sigma

        # Randomly Select Component to Merge
        if self.Post.K > 2:
            if merge_1 is None:
                merge_1 = np.random.choice(np.arange(self.Post.C))
            p_qt = Tree(self.Post.mu)
            centers = p_qt.query_ball_point(self.Post.mu[merge_1], target_radius)
            if len(centers) > 1:
                centers.remove(merge_1)
                centers = np.sort(centers)[::-1]
                for merge_2 in centers:
                    gap, evaluate, params = self.gap_merge(merge_1 + 1, merge_2 + 1)
                    if debug:
                        evaluate = True
                        plt.plot(self.x[:, 0], self.x[:, 1], ".")
                        plt.plot(self.Post.mu[merge_1, 0], self.Post.mu[merge_1, 1], '+')
                        plt.plot(self.Post.mu[merge_2, 0], self.Post.mu[merge_2, 1], '+')
                        plt.plot(params['mu'][0, 0], params['mu'][0, 1], 'or')
                    if evaluate:
                        self.move_counter[2, 0] += 1
                        if debug:
                            old_elbo = self.calc_elbo(deb=True)
                        self.merge_components(merge_1 + 1, merge_2 + 1, params)
                        if merge_2 < merge_1:
                            merge_1 += -1
                        if debug:
                            new_elbo = self.calc_elbo(deb=True)
                            print("DEBUG MERGE MOVE")
                            print("Old ELbo:{}".format(old_elbo))
                            print("New ELbo:{}".format(new_elbo))
                            print("True Gap:{}".format(new_elbo - old_elbo))
                            print("Estimated Gap:{}".format(gap))
                            print("EstimatedGap:{}    TrueGap{}".format(np.sum(gap), np.sum(new_elbo - old_elbo)))
                            self.vb_update(empty=False, iteration=1)
                            new_elbo = self.calc_elbo(deb=True)
                            print("EvolvedGap{}".format(new_elbo - old_elbo))
                            print("EvolvedGap{}".format(np.sum(new_elbo - old_elbo)))
                        self.vb_update(empty=False)
                        if prt == 1:
                            self.print_iteration(iteration=iteration,
                                                 message='Merge Move. Components:{}'.format([merge_1, merge_2]))
                    else:
                        self.move_counter[2, 1] += 1

    def split_move(self, prt=0, iteration=-1, k_split=None):

        # ##########################################################################################
        # Randomly Select Component to erase, weighted by the number of observations in the cluster
        if (k_split is None) and (self.Post.K > 1):
            pvals = self.Post.rk**2 / np.sum(self.Post.rk**2)
            k_split = np.argmax(np.random.multinomial(1, pvals)) + 1
        # ##########################################################################################

        if (k_split is not None) and (k_split <= self.Post.K):
            # ##########################################################################################
            # Isolate Cluster Points
            rnk = self.get_rnk_k(k_split)
            idx_points = rnk > CLT_BELONGING
            # Add nearby Noise Points
            p_qt = Tree(self.x)
            idx = np.array(p_qt.query_ball_point(self.Post.mu[k_split-1], 50))
            sub_idx = np.where(self.Post.rn0_vector[idx] > CLT_BELONGING)[0]
            idx_points[idx[sub_idx]] = True
            # ##########################################################################################

            # ##########################################################################################
            if np.sum(idx_points) > 5:
                param = {"points": self.x[idx_points], "sigmas2": self.sigma2_x[idx_points], "k_split": k_split,
                         "idx_points": idx_points, "rnk_split": rnk}
                if self.infer_pi1:
                    param["a"] = self.Post.a
                    param["b"] = self.Post.b
                gap, evaluate, split_params = handle_split(param, self.gap_split, self.propagate)
                if debug:
                    evaluate = True
                if evaluate:
                    self.move_counter[3, 0] += 1
                    if debug:
                        old_elbo = self.calc_elbo(deb=True)
                        plt.plot(self.x[:, 0], self.x[:, 1], '+k')
                        plt.plot(self.x[idx_points, 0], self.x[idx_points, 1], '+r')
                        plt.plot(self.Post.mu[k_split-1, 0], self.Post.mu[k_split-1, 1], 'og')
                        plt.plot(split_params['mu'][:, 0], split_params['mu'][:, 1], 'or')
                    self.split_component(comp=k_split, params=split_params)
                    if debug:
                        new_elbo = self.calc_elbo(deb=True)
                        print("DEBUG SPLIT MOVE")
                        print("Old ELbo:{}".format(old_elbo))
                        print("New ELbo:{}".format(new_elbo))
                        print("True Gap:{}".format(new_elbo - old_elbo))
                        print("Estimated Gap:{}".format(gap))
                        print("")
                        print("EstimatedGap:{}    TrueGap{}".format(np.sum(gap), np.sum(new_elbo - old_elbo)))
                    if prt == 1:
                        self.print_iteration(iteration=iteration,
                                             message='Split Move. Component:{}'.format(k_split))
                else:
                    self.move_counter[3, 1] += 1
            # ##########################################################################################
    # #########################################################################################################

    # #########################################################################################################
    # Plotting/Printing/Saving/Loading
    def save_configuration(self):
        self.best_configuration = copy.deepcopy(self.Post)
        if hasattr(self.Post, 'rnk'):
            self.best_configuration.removeField('rnk')

    def load_configuration(self):
        self.Post = copy.deepcopy(self.best_configuration)
        self.vb_update_local()

    def pl(self, ax=None, circles=False):
        # Bind Data
        x = self.x
        n = x.shape[0]
        sigma2_x = self.sigma2_x

        if ax is None:
            fig, ax = plt.subplots()
        ax.cla()

        # Plot all points in black
        ax.plot(x[:, 0], x[:, 1], '.k', markersize=0.5)
        if circles:
            if n < 1000:
                for n_ in np.arange(n):
                    ax.add_artist(plt.Circle((x[n_, 0], x[n_, 1]), np.sqrt(sigma2_x[n_, 0]), fill=False, lw=0.5))
            else:
                print("Cannot Show Circles for more than 1000 points")

        # Points in Clusters have colors
        colors = (i for i in 'bgrmy' * self.Post.K)

        for k_, c in zip(np.arange(self.Post.C), colors):
            if hasattr(self.Post, 'rnk'):
                idx = self.Post.rnk[:, k_ + 1] > CLT_BELONGING
            else:
                idx = self.get_rnk_k(k_) > CLT_BELONGING
            ax.scatter(x[idx, 0], x[idx, 1], c=c, s=1)
            ax.plot(self.Post.mu[k_, 0], self.Post.mu[k_, 1], '+' + c)
        plt.show()

        return ax

    def bl(self):
        """
        Plots the assignment of each point to Noise and clusters.
        :return:
        """
        if self.x.shape[0] > 1000:
            print("Cannot display cluster assignment for more than 1000points")
            return
        if hasattr(self.Post, 'rnk'):
            rnk = self.Post.rnk
        else:
            rnk = post.rn0_vector
            for k_ in np.arange(self.Post.K):
                rnk = np.concatenate([rnk, self.get_rnk_k(k_)])

        plt.imshow(rnk, aspect='auto')
        plt.show()

    def pl_bl(self, ax=None, circles=False, pdf_name=None):
        # Bind Data
        x = self.x
        sigma2_x = self.sigma2_x

        if (ax is None) or (pdf_name is not None):
            fig, ax = plt.subplots(1, 2)
        else:
            fig = []
        ax[0].cla()

        # plot all points in red with a circle
        ax[0].plot(x[:, 0], x[:, 1], '.k', markersize=0.5)
        if circles:
            for n_ in np.arange(x.shape[0]):
                ax[0].add_artist(plt.Circle((x[n_, 0], x[n_, 1]), np.sqrt(sigma2_x[n_, 0]), fill=False, lw=0.5))

        # Points in Clusters have colors
        colors = (i for i in 'bgrmy' * self.Post.K)
        for k_, c in zip(np.arange(self.Post.C), colors):
            if hasattr(self.Post, 'rnk'):
                idx = self.Post.rnk[:, k_ + 1] > CLT_BELONGING
            else:
                idx = self.get_rnk_k(k_ + 1) > CLT_BELONGING
            ax[0].scatter(x[idx, 0], x[idx, 1], c=c, s=1)
            ax[0].plot(self.Post.mu[k_, 0], self.Post.mu[k_, 1], '+' + c)
            ax[0].plot(self.Post.mu[k_, 0], self.Post.mu[k_, 1], '+' + c)

        if hasattr(self.Post, 'rnk'):
            rnk = self.Post.rnk
        else:
            rnk = self.Post.rn0_vector
            for k_ in np.arange(1, self.Post.K):
                rnk = np.vstack([rnk, self.get_rnk_k(k_)])
            rnk = rnk.transpose()

        ax[1].imshow(rnk, aspect='auto')
        plt.show()

        if pdf_name is not None:
            pp = PdfPages(pdf_name)
            pp.savefig(fig)
            pp.close()

        return ax

    @staticmethod
    def print_iteration(message=None, elbo=None, iteration=None, components=None):
        """
        Prints Iteration Specific Information.
        A message can be added to the statistics collected.
        :param message:
        :param elbo:
        :param iteration:
        :param components:
        :return:
        """
        if message is not None:
            print('iteration:{0:<5d}  {1:10s}'.format(iteration, message))
        else:
            print('iteration:{0:<5d}   ELBO: {1:<7.0f}   K Comps:{2:<5d}'.format(iteration, elbo, components))

    def print_number_moves(self):
        print("NUMBER OF ACCEPTED MOVES. Birth:{}, Dead:{}, Merge{}:, Split:{}".format(self.move_counter[0],
                                                                                       self.move_counter[1],
                                                                                       self.move_counter[2],
                                                                                       self.move_counter[3]))
    # #########################################################################################################


class TimeIndependentModel(AbstractModel):
    def __init__(self, data=None, x=None, sigma2=None, time=None,
                 mu0=None, sigma02=None, log_p0=None, rnk=None,
                 infer_pi1=False, pi1=None, a=None, b=None,
                 infer_alpha0=False, alpha0=None, gamma1=None, gamma2=None,
                 init_type='points', condition=None, prt=False, post=None, **kwargs):

        super(TimeIndependentModel, self).__init__()

        # #############################################################################################
        # #############################################################################################
        # Verify Data Dim and Bind it Inside Class
        if x is None:
            # 2D data
            if data.shape[1] == 4:
                if prt:
                    print("2D data detected.")
                self.x = data[:, :2]
                self.sigma2_x = data[:, 2:]
            # 3D data
            elif data.shape[1] == 6:
                if prt:
                    print("3D data detected.")
                self.x = data[:, :3]
                self.sigma2_x = data[:, 3:]
            # 2D data + Time
            elif data.shape[1] == 5:
                if prt:
                    print("2D data detected + time component.")
                self.time = data[:, 0]
                self.T = int(np.max(self.time))
                self.x = data[:, 1:3]
                self.sigma2_x = data[:, 3:]
            # 3D data + Time
            elif data.shape[1] == 7:
                if prt:
                    print("3D data detected + time component.")
                self.time = data[:, 0]
                self.T = int(np.max(self.time))
                self.x = data[:, 1:4]
                self.sigma2_x = data[:, 4:]
            else:
                sys.exit('Wrong Data Format')
        else:
            self.time = time
            self.T = int(np.max(self.time))
            self.x = x
            self.sigma2_x = sigma2
        self.mean_sigma = np.sqrt(np.mean(self.sigma2_x))
        # #############################################################################################
        # #############################################################################################

        # #############################################################################################
        # #############################################################################################
        # Compute prior
        mu0, sigma02, alpha0, log_p0, pi1, a, b, gamma1, gamma2, gamma0, _ = \
            priors(x=self.x, mu0=mu0, sigma02=sigma02, alpha0=alpha0,
                   pi1=pi1, log_p0=log_p0, gamma1=gamma1, gamma2=gamma2, a=a, b=b)

        # #############################################################################################
        # #############################################################################################

        # #############################################################################################
        # #############################################################################################
        # Build PRIOR containers
        self.Prior = ParamBag(K=0, D=self.x.shape[1])
        self.Prior.setField('mu0', mu0, dims='D')
        self.Prior.setField('sigma02', sigma02, dims='D')
        self.Prior.setField('logp0', log_p0, dims=None)

        # Pi1 containers
        self.infer_pi1 = infer_pi1
        if infer_pi1 is False:
            self.Prior.setField('pi1', pi1, dims=None)
        elif infer_pi1 is True:
            self.Prior.setField('a', a, dims=None)
            self.Prior.setField('b', b, dims=None)

        # Alpha0 containers
        self.infer_alpha0 = infer_alpha0
        if infer_alpha0 is False:
            self.Prior.setField('alpha0', alpha0, dims=None)
        else:
            self.Prior.setField('gamma1', gamma1, dims=None)
            self.Prior.setField('gamma2', gamma2, dims=None)
        # #############################################################################################
        # #############################################################################################

        # #############################################################################################
        # #############################################################################################
        # Build POSTERIOR containers
        if rnk is None:
            # Compute Initial Condition
            mu_init, sigma2_init, eta0_init, eta1_init, k0 = \
                initiliaze(x=self.x, sigma2_x=self.sigma2_x, alpha0=alpha0, condition=condition,
                           init_type=init_type, post=post)

            self.Post = ParamBag(K=k0 + 1, D=self.x.shape[1], N=self.x.shape[0], C=k0)
            self.Post.setField('mu', mu_init.copy(), dims=('C', 'D'))
            self.Post.setField('sigma2', sigma2_init.copy(), dims=('C', 'D'))
            self.Post.setField('eta0', eta0_init, dims='C')
            self.Post.setField('eta1', eta1_init, dims='C')
            if infer_pi1 is True:
                self.Post.setField('a', a, dims=None)
                self.Post.setField('b', b, dims=None)
            if infer_alpha0 is True:
                self.Post.setField('gamma1', gamma1, dims=None)
                self.Post.setField('gamma2', gamma2, dims=None)
        else:
            self.Post = ParamBag(K=rnk.shape[1], D=self.x.shape[1], N=self.x.shape[0], C=rnk.shape[1] - 1)
            self.Post.setField('rnk', rnk, dims=('N', 'K'))
            if infer_pi1 is True:
                self.Post.setField('a', a, dims=None)
                self.Post.setField('b', b, dims=None)
            if infer_alpha0 is True:
                self.Post.setField('gamma1', gamma1, dims=None)
                self.Post.setField('gamma2', gamma2, dims=None)
            self.vb_update_global()
        # #############################################################################################
        # #############################################################################################

    def vb_update(self, empty=False, prt=False, iteration=-1):

        self.vb_update_local()
        self.vb_update_global()

        # Empty-Redundant Cluster Move.
        if empty:
            self.empty(prt=prt, iteration=iteration)
            self.redundant(prt=prt, iteration=iteration)

    def calc_elbo(self, deb=False):
        """
        Calculates ELBO terms after an update !

        :return:
        """

        # Binding Data
        post = self.Post
        prior = self.Prior

        # Parameters
        n, k = post.N, post.K
        c = k - 1

        # Calculating Poisson Noise ELBO terms
        if self.infer_pi1 is True:
            e_noise = post.rn0_vector.sum() * prior.logp0
            e_noise += c_beta(eta1=prior.a, eta0=prior.b) - c_beta(eta1=post.a, eta0=post.b)
        else:
            e_noise = post.rn0_vector.sum() * (np.log(prior.pi1 / (1 - prior.pi1)) + prior.logp0)
            e_noise += n * np.log(1 - prior.pi1)

        e_obs = c * c_obs(prior.mu0, prior.sigma02)
        e_obs += post.cobs
        e_obs += - c_obs(post.mu, post.sigma2)

        if self.infer_alpha0 is True:
            e_sb = c_alpha(prior.gamma1, prior.gamma2, post.gamma1 / post.gamma2) - \
                   c_alpha(post.gamma1, post.gamma2, post.gamma1 / post.gamma2)
        else:
            e_sb = c * c_beta(1, prior.alpha0)
        e_sb += - c_beta(post.eta1, post.eta0)

        e_entropy = - self.calc_all_entropy()

        if deb:
            return np.array([e_noise, e_sb, e_entropy, e_obs])
        else:
            return e_noise + e_sb + e_entropy + e_obs

    def propagate(self, params, mus):

        # 0: Project Points To Centers' Proposals -> Get Initial rnk
        prior = self.Prior
        points = params["points"]
        sigmas2 = params["sigmas2"]
        n = points.shape[0]
        c, d = mus.shape
        new_rnk = np.zeros((n, c + 1))
        new_rnk[:, 0] = -100
        new_rnk[:, 1:] = e_log_n(points, sigmas2, mus, np.zeros((c, d)))
        e_norm(new_rnk)

        # 1: Iterate through the model n_iterations
        n_iterations = 10

        # Pi1
        if self.infer_pi1 is False:
            value0 = np.log(prior.pi1 / (1 - prior.pi1)) + prior.logp0
            l_1mpi1 = 0.
        else:
            l_pi1 = psi(params["a"])
            l_1mpi1 = psi(params["b"])
            value0 = l_pi1 - l_1mpi1 + prior.logp0
            l_1mpi1 = 0.

        for _ in np.arange(n_iterations):
            # Global Update
            # rc = np.sum(new_rnk[:, 1:], axis=0)
            sigma2c = (dotatb(new_rnk[:, 1:], np.reciprocal(sigmas2)) + prior.sigma02 ** -1) ** -1
            mus = (dotatb(new_rnk[:, 1:], points * np.reciprocal(sigmas2)) + prior.mu0 * (prior.sigma02 ** -1)) * \
                sigma2c

            # alphakt
            """
            eta1 = 1 + rc
            if self.infer_alpha0 is True:
                post.setField('eta0', post.gamma1 / post.gamma2 + convert_to_n0(post.rk), dims='C')
            else:
                post.setField('eta0', prior.alpha0 + convert_to_n0(post.rk), dims='C')
            elog_u, elog1m_u = calc_beta_expectations(post.eta1, post.eta0)
            if self.infer_alpha0 is True:
                post.setField('gamma1', prior.gamma1 + post.C, dims=None)
                post.setField('gamma2', prior.gamma2 - post.elog1m_u.sum(), dims=None)
            """

            new_rnk[:, 0] = value0
            new_rnk[:, 1:] = l_1mpi1 + e_log_n(points, sigmas2, mus, sigma2c)
            e_norm(new_rnk)

        # 2: Validate New Clusters (a_ points in them, b_ they are not too close)
        out_rnk = np.zeros((self.Post.N, new_rnk.shape[1] - 1))
        out_rnk[params['idx_points'], :] = new_rnk[:, 1:] / new_rnk[:, 1:].sum(axis=1)[:, None]
        out_rnk *= params['rnk_split'][:, None]
        out_rnk[np.isnan(out_rnk)] = 0.

        return out_rnk

    # ##############################################################
    # ##############################################################
    # GAPS
    def gap_delete(self, k):

        # Binding Parameters
        prior = self.Prior
        post = self.Post

        # Points in cluster
        rnk_k = self.get_rnk_k(k)
        idx_k = np.flatnonzero(rnk_k > CLT_BELONGING)

        if len(idx_k) < 1:
            gap = 0
            evaluation = False
        else:
            # Calculating mu, sigma ELBO terms
            gap_obs = - c_obs(mu=self.x, sigma2=self.sigma2_x, mult=rnk_k) \
                      - c_obs(mu=prior.mu0, sigma2=prior.sigma02) \
                      + c_obs(mu=post.mu[k - 1], sigma2=post.sigma2[k - 1])

            # Calculating Poisson Noise ELBO terms
            add_rn0 = np.sum(rnk_k)

            if self.infer_pi1 is True:
                gap_pois = add_rn0 * prior.logp0

                added = post.rn0_vector + rnk_k
                after_a = prior.a + added.sum()
                after_b = prior.b + (1. - added).sum()
                gap_pois += - c_beta(eta1=after_a, eta0=after_b) + c_beta(eta1=post.a, eta0=post.b)
            else:
                gap_pois = add_rn0 * (np.log(prior.pi1 / (1. - prior.pi1)) + prior.logp0)

            # Calculating Stick Break ELBO terms
            if self.infer_alpha0 is True:
                # After Erasing Cluster k
                after_gamma1 = post.gamma1 - 1.
                after_gamma2 = post.gamma2 + post.elog1m_u[k - 1]
                gap_sb = - c_alpha(after_gamma1, after_gamma2, after_gamma1 / after_gamma2)

                new_nn = np.delete(post.rk, k - 1, axis=0)
                after_eta1 = 1. + new_nn
                after_eta0 = after_gamma1 / after_gamma2 + convert_to_n0(new_nn)
                gap_sb += - c_beta(eta1=after_eta1, eta0=after_eta0)

                # Before
                gap_sb += c_alpha(post.gamma1, post.gamma2, post.gamma1 / post.gamma2)
                gap_sb += c_beta(eta1=post.eta1, eta0=post.eta0)
            else:
                new_nn = np.delete(post.rk, k - 1, axis=0)
                after_eta1 = 1. + new_nn
                after_eta0 = prior.alpha0 + convert_to_n0(new_nn)

                gap_sb = - c_beta(eta1=1, eta0=prior.alpha0)
                gap_sb += - c_beta(eta1=after_eta1, eta0=after_eta0) + c_beta(eta1=post.eta1, eta0=post.eta0)
                # gap_sb += - c_beta(eta1=after_eta1[:k-1], eta0=after_eta0[:k-1]) \
                #           + c_beta(eta1=post.eta1[:k], eta0=post.eta0[:k])
                # Subtraction of terms after k and k-1 respectively are zero.

            # Calculating Entropy ELBO terms
            # After
            gap_entropy = - calc_entropy(post.rn0_vector + rnk_k)
            # Before
            gap_entropy += calc_entropy(post.rn0_vector) + calc_entropy(rnk_k)

            gap = np.array([gap_pois, gap_sb, gap_entropy, gap_obs])
            evaluation = np.sum(gap) > 0

        return gap, evaluation

    def gap_merge(self, k1, k2):

        # Check k1 < k2
        if k1 > k2:
            k1, k2 = k2, k1

        # Binding Parameters
        prior = self.Prior
        post = self.Post

        # Estimate New Center
        rnk_k1 = self.get_rnk_k(k1)
        rnk_k2 = self.get_rnk_k(k2)
        rnk_k1_k2 = np.vstack([rnk_k1, rnk_k2]).transpose()

        # Calculating mu, sigma ELBO terms
        new_rnk = np.sum(rnk_k1_k2, axis=1)
        new_sigma2 = (np.sum(new_rnk[:, None] * (self.sigma2_x ** -1), axis=0) + self.Prior.sigma02 ** -1) ** -1
        new_mu = (np.sum(new_rnk[:, None] * self.x * (self.sigma2_x ** -1), axis=0) +
                  self.Prior.mu0 * (self.Prior.sigma02 ** -1)) * new_sigma2
        params = {"mu": new_mu[None, :], "sigma2": new_sigma2[None, :]}

        gap_obs = - c_obs(mu=prior.mu0, sigma2=prior.sigma02) - c_obs(mu=new_mu, sigma2=new_sigma2) + c_obs(
            mu=post.mu[[k1 - 1, k2 - 1], :], sigma2=post.sigma2[[k1 - 1, k2 - 1], :])
        # The next terms should be always zero
        # gap_obs += c_obs(mu=self.x, sigma2=self.sigma2_x, mult=new_rnk) + \
        #           - c_obs(mu=self.x, sigma2=self.sigma2_x, mult=post.rnk[:, k1]) \
        #           - c_obs(mu=self.x, sigma2=self.sigma2_x, mult=post.rnk[:, k2])

        # Calculating Stick Break ELBO terms
        if self.infer_alpha0 is True:
            # After Adding Cluster k at the beginning
            after_gamma1 = post.gamma1 - 1
            after_gamma2 = post.gamma2 + np.sum(post.elog1m_u[k2 - 1])

            after_eta1 = copy.copy(post.eta1[k1 - 1:k2 - 1])
            after_eta1[0] += post.eta1[k2 - 1] - 1
            after_eta0 = copy.copy(post.eta0[k1 - 1:k2 - 1])
            after_eta0 += - (post.eta1[k2 - 1] - 1) - post.gamma1 / post.gamma2 + after_gamma1 / after_gamma2

            gap_sb = - c_beta(eta1=after_eta1, eta0=after_eta0)
            gap_sb += c_beta(eta1=post.eta1[k1 - 1:k2], eta0=post.eta0[k1 - 1:k2])

            gap_sb += - c_alpha(after_gamma1, after_gamma2, after_gamma1 / after_gamma2)
            gap_sb += c_alpha(post.gamma1, post.gamma2, post.gamma1 / post.gamma2)
            params["eta0"] = after_eta0
            params["eta1"] = after_eta1
            params["gamma1"] = after_gamma1
            params["gamma2"] = after_gamma2

        else:
            # Merging Clusters k1, k2. Everything in between gets modify.
            after_eta1 = copy.copy(post.eta1[k1 - 1:k2 - 1])
            after_eta1[0] += post.eta1[k2 - 1] - 1
            after_eta0 = copy.copy(post.eta0[k1 - 1:k2 - 1])
            after_eta0 += - (post.eta1[k2 - 1] - 1)

            gap_sb = - c_beta(eta1=1., eta0=prior.alpha0) - c_beta(eta1=after_eta1, eta0=after_eta0) + c_beta(
                eta1=post.eta1[k1 - 1:k2], eta0=post.eta0[k1 - 1:k2])
            params["eta0"] = after_eta0
            params["eta1"] = after_eta1

        # Calculating Entropy ELBO terms
        gap_entropy = - calc_entropy(np.sum(rnk_k1_k2, axis=1)) + calc_entropy(rnk_k1) + calc_entropy(rnk_k2)

        gap = np.array([0, gap_sb, gap_entropy, gap_obs])
        evaluation = (gap_obs + gap_sb + gap_entropy) > 0

        return gap, evaluation, params

    def gap_birth(self, idx_points):

        # Initial check
        if len(idx_points) == 0:
            return -1, False, 0, 0

        # Binding Parameters
        post = self.Post
        prior = self.Prior
        x_k = self.x[idx_points]
        sigma2_xk = self.sigma2_x[idx_points]

        r_n0 = post.rn0_vector[idx_points]
        add_r_n0 = np.sum(r_n0)

        # New Sigma, mu
        sx = np.sum(r_n0[:, None] * (sigma2_xk ** -1), axis=0)
        sigma2_k = (sx + self.Prior.sigma02 ** -1) ** -1
        xx = np.sum(r_n0[:, None] * x_k * (sigma2_xk ** -1), axis=0)
        mu_k = (xx + self.Prior.mu0 * (self.Prior.sigma02 ** -1)) * sigma2_k
        params = {"mu": mu_k[None, :], "sigma2": sigma2_k[None, :], "idx": idx_points}

        # Calculating mu, sigma ELBO terms
        gap_obs = + c_obs(mu=x_k, sigma2=sigma2_xk, mult=r_n0) + c_obs(mu=prior.mu0, sigma2=prior.sigma02) - c_obs(
            mu=mu_k, sigma2=sigma2_k)

        # Calculating Stick Break ELBO terms
        # Inserting Cluster at the beginning creates minimum disturbances
        if self.infer_alpha0 is True:
            after_gamma1 = prior.gamma1 + post.C + 1
            after_gamma2 = prior.gamma2 - post.elog1m_u.sum()

            new_nn = np.insert(post.rk, 0, add_r_n0, axis=0)
            after_eta1 = 1. + new_nn
            after_eta0 = after_gamma1 / after_gamma2 + convert_to_n0(new_nn)
            params["rk"] = [add_r_n0]
            params["eta0"] = after_eta0
            params["eta1"] = after_eta1
            params["gamma1"] = after_gamma1
            params["gamma2"] = after_gamma2

            gap_sb = - c_alpha(after_gamma1, after_gamma2, after_gamma1 / after_gamma2)
            gap_sb += - c_beta(eta1=after_eta1, eta0=after_eta0)

            gap_sb += c_alpha(post.gamma1, post.gamma2, post.gamma1 / post.gamma2)
            gap_sb += c_beta(eta1=post.eta1, eta0=post.eta0)

        else:
            new_nn = np.insert(post.rk, 0, add_r_n0, axis=0)
            after_eta1 = 1. + new_nn
            after_eta0 = prior.alpha0 + convert_to_n0(new_nn)
            params["rk"] = [add_r_n0]
            params["eta0"] = after_eta0
            params["eta1"] = after_eta1

            gap_sb = c_beta(eta1=1, eta0=prior.alpha0)
            gap_sb += - c_beta(eta1=after_eta1[0], eta0=after_eta0[0])

        # Calculating Poisson Noise ELBO terms
        if self.infer_pi1 is True:
            gap_pois = - add_r_n0 * prior.logp0

            added = copy.copy(post.rn0_vector)
            added[idx_points] += - r_n0
            after_a = prior.a + added.sum()
            after_b = prior.b + (1. - added).sum()
            gap_pois += - c_beta(eta1=after_a, eta0=after_b) + c_beta(eta1=post.a, eta0=post.b)
            params["a"] = after_a
            params["b"] = after_b

        else:
            gap_pois = - add_r_n0 * (np.log(prior.pi1 / (1. - prior.pi1)) + prior.logp0)

        # Calculating Entropy ELBO terms
        # Entropy should be 0 as we just move the terms in r_n0 to r_nk_new
        """
        # After
        new_rn0 = copy.copy(self.Post.rnk[:, 0])
        new_rn0[idx_points] += - r_n0
        gap_entropy = - calc_entropy(new_rn0) - calc_entropy(r_n0)
        # Before
        gap_entropy += calc_entropy(self.Post.rnk[:, 0])
        """
        gap_entropy = 0

        gap = np.array([gap_pois, gap_sb, gap_entropy, gap_obs])
        evaluation = np.sum(gap) > 0

        return gap, evaluation, params

    def gap_split(self, k, new_rnk):
        # Initial Validation
        if new_rnk is None:
            return -np.inf, False

        # Binding Parameters
        prior = self.Prior
        post = self.Post
        xk = self.x
        sigma2_xk = self.sigma2_x
        rnk_k = self.get_rnk_k(k)

        # Calculating mu, sigma ELBO terms
        sx = np.sum(new_rnk[:, :, None] * (sigma2_xk ** -1)[:, None, :], axis=0)
        new_sigma2 = (sx + (self.Prior.sigma02 ** -1)) ** -1
        xx = np.sum(new_rnk[:, :, None] * xk[:, None, :] * (sigma2_xk ** -1)[:, None, :], axis=0)
        new_mu = (xx + self.Prior.mu0 * (self.Prior.sigma02 ** -1)) * new_sigma2
        params = {"mu": new_mu, "sigma2": new_sigma2}

        # Calculating Obs Model GAP terms.
        gap_obs = + c_obs(mu=prior.mu0, sigma2=prior.sigma02) - c_obs(mu=new_mu, sigma2=new_sigma2) + c_obs(
            mu=post.mu[k - 1], sigma2=post.sigma2[k - 1])
        # The next terms should be zero
        # gap_obs += c_obs(mu=self.x, sigma2=self.sigma2_x, mult=new_rnk) \
        #           - c_obs(mu=self.x, sigma2=self.sigma2_x, mult=post.rnk[:, k])

        # Calculating Stick Break GAP terms
        if self.infer_alpha0 is True:
            # After Adding Cluster k
            after_gamma1 = prior.gamma1 + post.C + 1
            after_gamma2 = prior.gamma2 - np.sum(post.elog1m_u) + post.elog1m_u[k - 1]

            new_nn = np.insert(post.rk, k - 1, 0, axis=0)
            new_nn[k - 1:k + 1] = np.sum(new_rnk, axis=0)
            after_eta1 = 1. + new_nn
            after_eta0 = after_gamma1 / after_gamma2 + convert_to_n0(new_nn)
            params["rk"] = np.sum(new_rnk, axis=0)
            params["eta0"] = after_eta0[k-1:k+1]
            params["eta1"] = after_eta1[k-1:k+1]
            params["gamma1"] = after_gamma1
            params["gamma2"] = after_gamma2

            gap_sb = - c_alpha(after_gamma1, after_gamma2, after_gamma1 / after_gamma2)
            gap_sb += c_alpha(post.gamma1, post.gamma2, post.gamma1 / post.gamma2)
            gap_sb += - c_beta(eta1=after_eta1, eta0=after_eta0)
            gap_sb += c_beta(eta1=post.eta1, eta0=post.eta0)
        else:
            # Splitting Cluster k into cluster k, k+1.
            """
            new_nn = np.insert(post.nn, k-1, 0, axis=0)
            new_nn[k-1:k+1] = np.sum(new_rnk, axis=0)
            after_eta1 = 1. + new_nn
            after_eta0 = prior.alpha0 + convert_to_n0(new_nn)
            gap_sb = c_beta(eta1=1, eta0=prior.alpha0)
            gap_sb += - c_beta(eta1=after_eta1[k - 1:k + 1], eta0=after_eta0[k - 1:k + 1])
            gap_sb += c_beta(eta1=post.eta1[k - 1], eta0=post.eta0[k - 1])
            """
            new_nn = np.sum(new_rnk, axis=0)
            after_eta1 = 1. + new_nn
            after_eta0 = post.eta0[k - 1] * np.ones(2)
            after_eta0[0] += new_nn[1]

            params["rk"] = np.sum(new_rnk, axis=0)
            params["eta0"] = after_eta0[k-1:k+1]
            params["eta1"] = after_eta1[k-1:k+1]

            gap_sb = c_beta(eta1=1, eta0=prior.alpha0)
            gap_sb += - c_beta(eta1=after_eta1, eta0=after_eta0)
            gap_sb += c_beta(eta1=post.eta1[k - 1], eta0=post.eta0[k - 1])

        # Calculating Entropy GAP terms
        gap_entropy = - calc_entropy(new_rnk) + calc_entropy(rnk_k)

        gap = np.array([0, gap_sb, gap_entropy, gap_obs])
        evaluation = (gap_obs + gap_sb + gap_entropy) > 0

        return gap, evaluation, params
    # ##############################################################
    # ##############################################################

    # #########################################################################################################
    # Abstract Methods
    @abstractmethod
    def calc_all_entropy(self):
        raise NotImplementedError

    @abstractmethod
    def get_rnk_k(self, k):
        raise NotImplementedError

    @abstractmethod
    def vb_update_local(self):
        raise NotImplementedError

    @abstractmethod
    def vb_update_global(self):
        raise NotImplementedError

    @abstractmethod
    def find_nearby_centers(self, target_radius):
        raise NotImplementedError
    # #########################################################################################################


class TimeIndepentModelPython(TimeIndependentModel):

    def __init__(self, **kwargs):
        super(TimeIndepentModelPython, self).__init__(**kwargs)

    def calc_all_entropy(self):
        return calc_entropy(self.Post.rnk)

    def get_rnk_k(self, k):
        return self.Post.rnk[:, k]

    def find_nearby_centers(self, target_radius):
        p_qt = Tree(self.Post.mu)
        nearby = list()
        for c_ in np.arange(self.Post.C):
            centers = p_qt.query_ball_point(self.Post.mu[c_], target_radius)
            centers.remove(c_)
            if len(centers) > 1:
                centers = np.sort(centers)[::-1]
                elements = centers[[not (x in nearby) for x in centers]]
                [nearby.append(e_) for e_ in elements]

        return np.sort(nearby) + 1

    # INFERENCE ROUTINES
    def vb_update_local(self):
        # Bind Data
        x = self.x
        sigma2_x = self.sigma2_x
        post = self.Post
        prior = self.Prior

        # Pi1
        if self.infer_pi1 is False:
            value0 = np.log(prior.pi1 / (1 - prior.pi1)) + prior.logp0
            l_1mpi1 = 0.
        else:
            l_pi1 = psi(post.a)  # - psi(self.Post.a + self.Post.b)
            l_1mpi1 = psi(post.b)  # - psi(self.Post.a + self.Post.b)
            value0 = l_pi1 - l_1mpi1 + prior.logp0
            l_1mpi1 = 0.

        # rnk update
        elog_beta_k = e_log_beta(eta1=post.eta1, eta0=post.eta0)
        elog_n_nk = e_log_n(x, sigma2_x, post.mu, post.sigma2)

        rnk = np.zeros((post.N, post.K))
        rnk[:, 0] = value0
        rnk[:, 1:] = l_1mpi1 + elog_n_nk + elog_beta_k[None, :]
        e_norm(rnk)
        post.setField('rnk', rnk, dims=('N', 'K'))
        post.setField('rn0_vector', rnk[:, 0], dims='N')
        post.setField('cobs', c_obs(x, sigma2_x, (1. - post.rn0_vector)), dims=None)

    def vb_update_global(self):
        # Bind data
        x = self.x
        sigma2_x = self.sigma2_x
        post = self.Post
        prior = self.Prior

        # Calculate Sufficient Stats
        post.setField('rk', np.sum(post.rnk[:, 1:], axis=0), dims='C')
        post.setField('xx', dotatb(post.rnk[:, 1:], x * np.reciprocal(sigma2_x)), dims=('C', 'D'))
        post.setField('sx', dotatb(post.rnk[:, 1:], np.reciprocal(sigma2_x)), dims=('C', 'D'))

        # sigma, mu
        post.setField('sigma2', (post.sx + prior.sigma02 ** -1) ** -1, dims=('C', 'D'))
        post.setField('mu', (post.xx + prior.mu0 * (prior.sigma02 ** -1)) * post.sigma2, dims=('C', 'D'))

        # alphakt
        post.setField('eta1', 1 + post.rk, dims='C')
        if self.infer_alpha0 is True:
            post.setField('eta0', post.gamma1 / post.gamma2 + convert_to_n0(post.rk), dims='C')
        else:
            post.setField('eta0', prior.alpha0 + convert_to_n0(post.rk), dims='C')
        elog_u, elog1m_u = calc_beta_expectations(post.eta1, post.eta0)
        post.setField('elog_u', elog_u, dims='C')
        post.setField('elog1m_u', elog1m_u, dims='C')
        if self.infer_alpha0 is True:
            post.setField('gamma1', prior.gamma1 + post.C, dims=None)
            post.setField('gamma2', prior.gamma2 - post.elog1m_u.sum(), dims=None)

        # pi1
        if self.infer_pi1 is True:
            post.setField('a', prior.a + np.sum(post.rn0_vector), dims=None)
            post.setField('b', prior.b + np.sum(1 - post.rn0_vector), dims=None)
    # ##############################################################
    # ##############################################################


class TimeIndepentModelC(TimeIndependentModel):
    def __init__(self, **kwargs):

        super(TimeIndepentModelC, self).__init__(**kwargs)

        self.LC = LightC.LightC()
        # perm_forward, perm_reverse = self.LC.find_cache_friendly_permutation(np.float64(self.x).copy())
        # self.x = self.x[perm_forward, :]
        # self.sigma2_x = self.sigma2_x[perm_forward, :]
        self.LC.load_points(np.float64(self.x).copy(), np.float64(self.sigma2_x).copy())

    def calc_all_entropy(self):
        return self.LC.calc_entropy()

    def get_rnk_k(self, k):
        if k == 0:
            rnk_k = self.Post.rn0_vector
            idx_k = np.flatnonzero(rnk_k)
        else:
            rnk_k, idx_k = self.LC.get_rnk_given_c(k)
        out_rnk_k = np.zeros(self.Post.N)
        out_rnk_k[idx_k] = rnk_k

        return out_rnk_k

    def find_nearby_centers(self, target_radius):
        near = self.LC.get_nearby_centers(target_radius)
        collapsed_components = np.array(np.where(near)[0])

        return collapsed_components

    def vb_update_local(self, search_radius=4):
        # Bind Data
        post = self.Post
        prior = self.Prior

        if post.K > 1:
            # ###### LOCAL UPDATE
            # Pi1
            if self.infer_pi1 is False:
                value0 = np.log(prior.pi1 / (1 - prior.pi1)) + prior.logp0
                l_1mpi1 = 0.
            else:
                l_pi1 = psi(post.a)  # - psi(self.Post.a + self.Post.b)
                l_1mpi1 = psi(post.b)  # - psi(self.Post.a + self.Post.b)
                value0 = l_pi1 - l_1mpi1 + prior.logp0
                l_1mpi1 = 0.

            # rnk update
            elog_beta_k = e_log_beta(eta1=post.eta1, eta0=post.eta0)
            self.LC.load_centers(post.mu, post.sigma2, elog_beta_k)
            self.LC.build_kdtree_centers()
            xx, sx, rk, rn0_vector, cobs = self.LC.points_to_centers(l_1mpi1, value0, search_radius)

            # ###### GLOBAL UPDATE
            # mu, sigma2
            post.setField('sigma2', (sx + prior.sigma02 ** -1) ** -1, dims=('C', 'D'))
            post.setField('mu', (xx + prior.mu0 * (prior.sigma02 ** -1)) * post.sigma2, dims=('C', 'D'))
            post.setField('rk', rk, dims='C')
            post.setField('rn0_vector', rn0_vector, dims='N')
            post.setField('cobs', cobs, dims=None)

            # alphakt
            post.setField('eta1', 1 + rk, dims='C')
            if self.infer_alpha0 is True:
                post.setField('eta0', post.gamma1 / post.gamma2 + convert_to_n0(rk), dims='C')
            else:
                post.setField('eta0', prior.alpha0 + convert_to_n0(rk), dims='C')
            elog_u, elog1m_u = calc_beta_expectations(post.eta1, post.eta0)
            post.setField('elog_u', elog_u, dims='C')
            post.setField('elog1m_u', elog1m_u, dims='C')
            if self.infer_alpha0 is True:
                post.setField('gamma1', prior.gamma1 + post.C, dims=None)
                post.setField('gamma2', prior.gamma2 - post.elog1m_u.sum(), dims=None)

            # pi1
            if self.infer_pi1 is True:
                post.setField('a', prior.a + post.rn0_vector.sum(), dims=None)
                post.setField('b', prior.b + (1. - post.rn0_vector).sum(), dims=None)

    def vb_update_global(self):
        pass
    # ##############################################################
    # ##############################################################


class TimeDependentModel(AbstractModel):
    def __init__(self, data=None, x=None, sigma2=None, time=None,
                 mu0=None, sigma02=None, log_p0=None, rnk=None,
                 infer_pi1=False, pi1=None, a=None, b=None,
                 infer_alpha0=False, alpha0=None, gamma1=None, gamma2=None,
                 init_type='points', condition=None, prt=False, post=None, **kwargs):

        super(TimeDependentModel, self).__init__()

        # #############################################################################################
        # #############################################################################################
        # Verify Data Dim and Bind it Inside Class
        if x is None:
            # 2D data
            if data.shape[1] == 4:
                if prt:
                    print("2D data detected.")
                self.x = data[:, :2]
                self.sigma2_x = data[:, 2:]
            # 3D data
            elif data.shape[1] == 6:
                if prt:
                    print("3D data detected.")
                self.x = data[:, :3]
                self.sigma2_x = data[:, 3:]
            # 2D data + Time
            elif data.shape[1] == 5:
                if prt:
                    print("2D data detected + time component.")
                self.time = data[:, 0]
                self.T = int(np.max(self.time))
                self.x = data[:, 1:3]
                self.sigma2_x = data[:, 3:]
            # 3D data + Time
            elif data.shape[1] == 7:
                if prt:
                    print("3D data detected + time component.")
                self.time = data[:, 0]
                self.T = int(np.max(self.time))
                self.x = data[:, 1:4]
                self.sigma2_x = data[:, 4:]
            else:
                sys.exit('Wrong Data Format')
        else:
            self.time = time
            self.T = int(np.max(self.time))
            self.x = x
            self.sigma2_x = sigma2
        self.mean_sigma = np.sqrt(np.mean(self.sigma2_x))
        # #############################################################################################
        # #############################################################################################

        # #############################################################################################
        # #############################################################################################
        # Compute prior
        mu0, sigma02, alpha0, log_p0, pi1, a, b, gamma1, gamma2, gamma0, _ = \
            priors(x=self.x, mu0=mu0, sigma02=sigma02, alpha0=alpha0,
                   pi1=pi1, log_p0=log_p0, gamma1=gamma1, gamma2=gamma2, a=a, b=b)

        # #############################################################################################
        # #############################################################################################

        # #############################################################################################
        # #############################################################################################
        # Build PRIOR containers
        self.Prior = ParamBag(K=0, D=self.x.shape[1])
        self.Prior.setField('mu0', mu0, dims='D')
        self.Prior.setField('sigma02', sigma02, dims='D')
        self.Prior.setField('logp0', log_p0, dims=None)

        # Pi1 containers
        self.infer_pi1 = infer_pi1
        if infer_pi1 is False:
            self.Prior.setField('pi1', pi1, dims=None)
        elif infer_pi1 is True:
            self.Prior.setField('a', a, dims=None)
            self.Prior.setField('b', b, dims=None)

        # Alpha0 containers
        self.infer_alpha0 = infer_alpha0
        self.Prior.setField('alpha0', alpha0, dims=None)

        # Time dependent terms
        aij = np.zeros((4, 4))
        aij[0, 0] = self.Prior.alpha0
        aij[0, 1] = 1
        aij[1, 1:] = 1
        aij[2, 1:3] = 1
        aij[3, 3] = 1
        self.Prior.setField('aij', aij, dims=('A', 'A'))
        # #############################################################################################
        # #############################################################################################

        # #############################################################################################
        # #############################################################################################
        # Build POSTERIOR containers
        if rnk is None:
            # Compute Initial Condition
            mu_init, sigma2_init, eta0_init, eta1_init, k0 = \
                initiliaze(x=self.x, sigma2_x=self.sigma2_x, alpha0=alpha0, condition=condition,
                           init_type=init_type, post=post)

            self.Post = ParamBag(K=k0 + 1, D=self.x.shape[1], N=self.x.shape[0], C=k0)
            self.Post.setField('mu', mu_init.copy(), dims=('C', 'D'))
            self.Post.setField('sigma2', sigma2_init.copy(), dims=('C', 'D'))
            if infer_pi1 is True:
                self.Post.setField('a', a, dims=None)
                self.Post.setField('b', b, dims=None)
            self.Post.setField('alpha', eta0_init, dims='C')
        else:
            self.Post = ParamBag(K=rnk.shape[1], D=self.x.shape[1], N=self.x.shape[0], C=rnk.shape[1] - 1)
            self.Post.setField('rnk', rnk, dims=('N', 'K'))
            self.vb_update_global()

        # Time dependent terms
        self.Post.setField('aij', self.Prior.aij + self.init_aij(), dims=('A', 'A'))
        self.Post.setField('const', None, dims=None)
        # #############################################################################################
        # #############################################################################################

    def init_aij(self):
        counts = np.zeros((4, 4))
        counts[0, 1] = 10  # self.Post.C
        counts[1, 2] = 10  # self.Post.C
        counts[1, 3] = 10

        for c_ in np.arange(np.min([10, self.Post.C])):
            times = self.time[np.where(self.Post.rnk[c_ + 1, :] > 0)]
            n_times = len(times)
            d_times = np.diff(times)
            if n_times > 1:
                a11 = np.sum(d_times == 1)
                length = times[-1] - times[0]
                counts[0, 0] += times[0]
                counts[1, 1] += a11
                counts[1, 2] += n_times - a11 - 1
                counts[2, 1] += counts[1, 2] + 1
                counts[2, 2] += length - n_times - counts[2, 1] - counts[1, 2]

        return counts

    def vb_update(self, empty=False, prt=False, iteration=-1):

        self.vb_update_local()
        self.vb_update_global()

        # Empty-Redundant Cluster Move.
        if empty:
            self.empty(prt=prt, iteration=iteration)
            self.redundant(prt=prt, iteration=iteration)

    def calc_elbo(self, deb=False):
        """
        Calculates ELBO terms after an update !

        :return:
        """

        # Binding Data
        post = self.Post
        prior = self.Prior
        n, k = post.N, post.K
        c = k - 1

        # Calculating Poisson Noise ELBO terms
        if self.infer_pi1 is True:
            e_noise = post.rn0_vector.sum() * prior.logp0
            e_noise += c_beta(eta1=prior.a, eta0=prior.b) - c_beta(eta1=post.a, eta0=post.b)
        else:
            e_noise = post.rn0_vector.sum() * (np.log(prior.pi1 / (1 - prior.pi1)) + prior.logp0)
            e_noise += n * np.log(1 - prior.pi1)

        e_obs = c * c_obs(prior.mu0, prior.sigma02)
        e_obs += post.cobs
        e_obs += - c_obs(post.mu, post.sigma2)

        e_sb = c * c_gamma(prior.alpha0, 1) - c_gamma(post.alpha, 1)

        e_entropy = - self.calc_all_entropy()

        if deb:
            return np.array([e_noise, e_sb, e_entropy, e_obs])
        else:
            return e_noise + e_sb + e_entropy + e_obs

    def vb_update_tterms(self):

        counts, const = self.calc_time_series()

        self.Post.setField('aij', self.Prior.aij + counts, dims=('A', 'A'))
        self.Post.setField('const', const.squeeze(), dims=None)

    def calc_elbo_tterms(self):

        if np.isnan(self.Post.const):
            self.vb_update_tterms()

        t_elbo = 0
        for r_ in np.arange(self.Post.A):
            aij_prior = self.Prior.aij[r_]
            aij_post = self.Post.aij[r_]
            t_elbo += c_dir(aij_prior[aij_prior > 0]) - c_dir(aij_post[aij_post > 0])

        t_elbo += self.Post.const * self.Post.C

        return t_elbo

    def calc_time_series(self):

        # Calculate an average decaying time trace for an average fluorophore
        c = self.Post.C
        log_tmat = psi(self.Post.aij) - psi(np.sum(self.Post.aij, axis=1))[:, None]
        log_tmat = log_tmat[1:, 1:]
        e_g_c = self.Post.alpha / self.Post.alpha.sum()
        log_likelihood = np.array([-np.mean(e_g_c), -1e-10, -1e-10])[None, :] * np.ones((10000, 3))
        resp, resp_pair, const = fw_bw(np.array([0, -1e10, -1e10]), log_tmat, log_likelihood)

        # First Order Approximation, Only 1 when rnk is 1.
        # Then, keep initial position and final position.
        # Second Order approximation,
        # In between one, calculate exact decay.
        counts = np.zeros((4, 4))
        counts[1:, 1:] += resp_pair * c
        counts[0, 1] = c
        counts[3, 3] = 0
        for c_ in np.arange(c):
            times = self.time[np.where(self.get_rnk_k(c_ + 1) > 0)]
            n_times = len(times)
            d_times = np.diff(times)
            if n_times > 1:
                a11 = np.sum(d_times == 1)
                length = times[-1] - times[0]
                counts[0, 0] += times[0]
                counts[1, 1] += a11
                counts[1, 2] += n_times - a11 - 1
                counts[2, 1] += n_times - a11
                counts[2, 2] += length - n_times - n_times + a11 - n_times + a11 + 1

        return counts, const

    def propagate(self, params, mus):

        # 0: Project Points To Centers' Proposals -> Get Initial rnk
        prior = self.Prior
        points = params["points"]
        sigmas2 = params["sigmas2"]
        n = points.shape[0]
        c, d = mus.shape
        new_rnk = np.zeros((n, c + 1))
        new_rnk[:, 0] = -100
        new_rnk[:, 1:] = e_log_n(points, sigmas2, mus, np.zeros((c, d)))
        e_norm(new_rnk)

        # 1: Iterate through the model n_iterations
        n_iterations = 10

        # Pi1
        if self.infer_pi1 is False:
            value0 = np.log(prior.pi1 / (1 - prior.pi1)) + prior.logp0
            l_1mpi1 = 0.
        else:
            l_pi1 = psi(params["a"])
            l_1mpi1 = psi(params["b"])
            value0 = l_pi1 - l_1mpi1 + prior.logp0
            l_1mpi1 = 0.

        for _ in np.arange(n_iterations):
            # Global Update
            # rc = np.sum(new_rnk[:, 1:], axis=0)
            sigma2c = (dotatb(new_rnk[:, 1:], np.reciprocal(sigmas2)) + prior.sigma02 ** -1) ** -1
            mus = (dotatb(new_rnk[:, 1:], points * np.reciprocal(sigmas2)) + prior.mu0 * (prior.sigma02 ** -1)) * \
                sigma2c

            # alphakt
            """
            eta1 = 1 + rc
            if self.infer_alpha0 is True:
                post.setField('eta0', post.gamma1 / post.gamma2 + convert_to_n0(post.rk), dims='C')
            else:
                post.setField('eta0', prior.alpha0 + convert_to_n0(post.rk), dims='C')
            elog_u, elog1m_u = calc_beta_expectations(post.eta1, post.eta0)
            if self.infer_alpha0 is True:
                post.setField('gamma1', prior.gamma1 + post.C, dims=None)
                post.setField('gamma2', prior.gamma2 - post.elog1m_u.sum(), dims=None)
            """

            new_rnk[:, 0] = value0
            new_rnk[:, 1:] = l_1mpi1 + e_log_n(points, sigmas2, mus, sigma2c)
            e_norm(new_rnk)

        # 2: Validate New Clusters (a_ points in them, b_ they are not too close)
        out_rnk = np.zeros((self.Post.N, new_rnk.shape[1] - 1))
        out_rnk[params['idx_points'], :] = new_rnk[:, 1:] / new_rnk[:, 1:].sum(axis=1)[:, None]
        out_rnk *= params['rnk_split'][:, None]
        out_rnk[np.isnan(out_rnk)] = 0.

        return out_rnk

    # ##############################################################
    # ##############################################################
    # GAPS
    def gap_delete(self, k):

        # Binding Parameters
        prior = self.Prior
        post = self.Post

        # Points in cluster
        rnk_k = self.get_rnk_k(k)
        idx_k = np.flatnonzero(rnk_k > CLT_BELONGING)

        if len(idx_k) < 1:
            gap = 0
            evaluation = False
        else:
            # Calculating mu, sigma ELBO terms
            gap_obs = - c_obs(mu=self.x, sigma2=self.sigma2_x, mult=rnk_k) \
                      - c_obs(mu=prior.mu0, sigma2=prior.sigma02) \
                      + c_obs(mu=post.mu[k - 1], sigma2=post.sigma2[k - 1])

            # Calculating Poisson Noise ELBO terms
            add_rn0 = np.sum(rnk_k)

            if self.infer_pi1 is True:
                gap_pois = add_rn0 * prior.logp0

                added = post.rn0_vector + rnk_k
                after_a = prior.a + added.sum()
                after_b = prior.b + (1. - added).sum()
                gap_pois += - c_beta(eta1=after_a, eta0=after_b) + c_beta(eta1=post.a, eta0=post.b)
            else:
                gap_pois = add_rn0 * (np.log(prior.pi1 / (1. - prior.pi1)) + prior.logp0)

            # Calculating Stick Break ELBO terms
            if self.infer_alpha0 is True:
                # After Erasing Cluster k
                after_gamma1 = post.gamma1 - 1.
                after_gamma2 = post.gamma2 + post.elog1m_u[k - 1]
                gap_sb = - c_alpha(after_gamma1, after_gamma2, after_gamma1 / after_gamma2)

                new_nn = np.delete(post.rk, k - 1, axis=0)
                after_eta1 = 1. + new_nn
                after_eta0 = after_gamma1 / after_gamma2 + convert_to_n0(new_nn)
                gap_sb += - c_beta(eta1=after_eta1, eta0=after_eta0)

                # Before
                gap_sb += c_alpha(post.gamma1, post.gamma2, post.gamma1 / post.gamma2)
                gap_sb += c_beta(eta1=post.eta1, eta0=post.eta0)
            else:
                new_nn = np.delete(post.rk, k - 1, axis=0)
                after_eta1 = 1. + new_nn
                after_eta0 = prior.alpha0 + convert_to_n0(new_nn)

                gap_sb = - c_beta(eta1=1, eta0=prior.alpha0)
                gap_sb += - c_beta(eta1=after_eta1, eta0=after_eta0) + c_beta(eta1=post.eta1, eta0=post.eta0)
                # gap_sb += - c_beta(eta1=after_eta1[:k-1], eta0=after_eta0[:k-1]) \
                #           + c_beta(eta1=post.eta1[:k], eta0=post.eta0[:k])
                # Subtraction of terms after k and k-1 respectively are zero.

            # Calculating Entropy ELBO terms
            # After
            gap_entropy = - calc_entropy(post.rn0_vector + rnk_k)
            # Before
            gap_entropy += calc_entropy(post.rn0_vector) + calc_entropy(rnk_k)

            gap = np.array([gap_pois, gap_sb, gap_entropy, gap_obs])
            evaluation = np.sum(gap) > 0

        return gap, evaluation

    def gap_merge(self, k1, k2):

        # Check k1 < k2
        if k1 > k2:
            k1, k2 = k2, k1

        # Binding Parameters
        prior = self.Prior
        post = self.Post

        # Estimate New Center
        rnk_k1 = self.get_rnk_k(k1)
        rnk_k2 = self.get_rnk_k(k2)
        rnk_k1_k2 = np.vstack([rnk_k1, rnk_k2]).transpose()

        # Calculating mu, sigma ELBO terms
        new_rnk = np.sum(rnk_k1_k2, axis=1)
        new_sigma2 = (np.sum(new_rnk[:, None] * (self.sigma2_x ** -1), axis=0) + prior.sigma02 ** -1) ** -1
        new_mu = (np.sum(new_rnk[:, None] * self.x * (self.sigma2_x ** -1), axis=0) +
                  prior.mu0 * (prior.sigma02 ** -1)) * new_sigma2
        params = {"mu": new_mu[None, :], "sigma2": new_sigma2[None, :]}

        gap_obs = - c_obs(mu=prior.mu0, sigma2=prior.sigma02) - c_obs(mu=new_mu, sigma2=new_sigma2) + c_obs(
            mu=post.mu[[k1 - 1, k2 - 1], :], sigma2=post.sigma2[[k1 - 1, k2 - 1], :])
        # The next terms should be always zero
        # gap_obs += c_obs(mu=self.x, sigma2=self.sigma2_x, mult=new_rnk) + \
        #           - c_obs(mu=self.x, sigma2=self.sigma2_x, mult=post.rnk[:, k1]) \
        #           - c_obs(mu=self.x, sigma2=self.sigma2_x, mult=post.rnk[:, k2])

        # Calculating Stick Break ELBO terms
        # Calculating Gamma ELBO terms, Merging Clusters k1, k2.
        alpha_k12 = post.alpha[[k1 - 1, k2 - 1]]
        alpha_merge = np.sum(alpha_k12) - prior.alpha0 / post.C
        gap_gamma = - c_gamma(alpha_merge, 1) + c_gamma(alpha_k12, 1) - c_gamma(self.Prior.alpha0, 1)
        params["alpha"] = alpha_merge

        # Calculating Entropy ELBO terms
        gap_entropy = - calc_entropy(np.sum(rnk_k1_k2, axis=1)) + calc_entropy(rnk_k1) + calc_entropy(rnk_k2)

        gap = np.array([0, gap_gamma, gap_entropy, gap_obs])
        evaluation = (gap_obs + gap_gamma + gap_entropy) > 0

        return gap, evaluation, params

    def gap_birth(self, idx_points):

        # Initial check
        if len(idx_points) == 0:
            return -1, False, 0, 0

        # Binding Parameters
        post = self.Post
        prior = self.Prior
        x_k = self.x[idx_points]
        sigma2_xk = self.sigma2_x[idx_points]

        r_n0 = post.rn0_vector[idx_points]
        add_r_n0 = np.sum(r_n0)

        # New Sigma, mu
        sx = np.sum(r_n0[:, None] * (sigma2_xk ** -1), axis=0)
        sigma2_k = (sx + self.Prior.sigma02 ** -1) ** -1
        xx = np.sum(r_n0[:, None] * x_k * (sigma2_xk ** -1), axis=0)
        mu_k = (xx + self.Prior.mu0 * (self.Prior.sigma02 ** -1)) * sigma2_k
        params = {"mu": mu_k[None, :], "sigma2": sigma2_k[None, :], "idx": idx_points}

        # Calculating mu, sigma ELBO terms
        gap_obs = + c_obs(mu=x_k, sigma2=sigma2_xk, mult=r_n0) + c_obs(mu=prior.mu0, sigma2=prior.sigma02) - c_obs(
            mu=mu_k, sigma2=sigma2_k)

        # Calculating Stick Break ELBO terms
        # Inserting Cluster at the beginning creates minimum disturbances
        if self.infer_alpha0 is True:
            after_gamma1 = prior.gamma1 + post.C + 1
            after_gamma2 = prior.gamma2 - post.elog1m_u.sum()

            new_nn = np.insert(post.rk, 0, add_r_n0, axis=0)
            after_eta1 = 1. + new_nn
            after_eta0 = after_gamma1 / after_gamma2 + convert_to_n0(new_nn)
            params["rk"] = [add_r_n0]
            params["eta0"] = after_eta0
            params["eta1"] = after_eta1
            params["gamma1"] = after_gamma1
            params["gamma2"] = after_gamma2

            gap_sb = - c_alpha(after_gamma1, after_gamma2, after_gamma1 / after_gamma2)
            gap_sb += - c_beta(eta1=after_eta1, eta0=after_eta0)

            gap_sb += c_alpha(post.gamma1, post.gamma2, post.gamma1 / post.gamma2)
            gap_sb += c_beta(eta1=post.eta1, eta0=post.eta0)

        else:
            new_nn = np.insert(post.rk, 0, add_r_n0, axis=0)
            after_eta1 = 1. + new_nn
            after_eta0 = prior.alpha0 + convert_to_n0(new_nn)
            params["rk"] = [add_r_n0]
            params["eta0"] = after_eta0
            params["eta1"] = after_eta1

            gap_sb = c_beta(eta1=1, eta0=prior.alpha0)
            gap_sb += - c_beta(eta1=after_eta1[0], eta0=after_eta0[0])

        # Calculating Poisson Noise ELBO terms
        if self.infer_pi1 is True:
            gap_pois = - add_r_n0 * prior.logp0

            added = copy.copy(post.rn0_vector)
            added[idx_points] += - r_n0
            after_a = prior.a + added.sum()
            after_b = prior.b + (1. - added).sum()
            gap_pois += - c_beta(eta1=after_a, eta0=after_b) + c_beta(eta1=post.a, eta0=post.b)
            params["a"] = after_a
            params["b"] = after_b

        else:
            gap_pois = - add_r_n0 * (np.log(prior.pi1 / (1. - prior.pi1)) + prior.logp0)

        # Calculating Entropy ELBO terms
        # Entropy should be 0 as we just move the terms in r_n0 to r_nk_new
        """
        # After
        new_rn0 = copy.copy(self.Post.rnk[:, 0])
        new_rn0[idx_points] += - r_n0
        gap_entropy = - calc_entropy(new_rn0) - calc_entropy(r_n0)
        # Before
        gap_entropy += calc_entropy(self.Post.rnk[:, 0])
        """
        gap_entropy = 0

        gap = np.array([gap_pois, gap_sb, gap_entropy, gap_obs])
        evaluation = np.sum(gap) > 0

        return gap, evaluation, params

    def gap_split(self, k, new_rnk):
        # Initial Validation
        if new_rnk is None:
            return -np.inf, False

        # Binding Parameters
        prior = self.Prior
        post = self.Post
        xk = self.x
        sigma2_xk = self.sigma2_x
        rnk_k = self.get_rnk_k(k)

        # Calculating mu, sigma ELBO terms
        sx = np.sum(new_rnk[:, :, None] * (sigma2_xk ** -1)[:, None, :], axis=0)
        new_sigma2 = (sx + (self.Prior.sigma02 ** -1)) ** -1
        xx = np.sum(new_rnk[:, :, None] * xk[:, None, :] * (sigma2_xk ** -1)[:, None, :], axis=0)
        new_mu = (xx + self.Prior.mu0 * (self.Prior.sigma02 ** -1)) * new_sigma2
        params = {"mu": new_mu, "sigma2": new_sigma2}

        # Calculating Obs Model GAP terms.
        gap_obs = + c_obs(mu=prior.mu0, sigma2=prior.sigma02) - c_obs(mu=new_mu, sigma2=new_sigma2) + c_obs(
            mu=post.mu[k - 1], sigma2=post.sigma2[k - 1])
        # The next terms should be zero
        # gap_obs += c_obs(mu=self.x, sigma2=self.sigma2_x, mult=new_rnk) \
        #           - c_obs(mu=self.x, sigma2=self.sigma2_x, mult=post.rnk[:, k])

        # Calculating Stick Break GAP terms
        if self.infer_alpha0 is True:
            # After Adding Cluster k
            after_gamma1 = prior.gamma1 + post.C + 1
            after_gamma2 = prior.gamma2 - np.sum(post.elog1m_u) + post.elog1m_u[k - 1]

            new_nn = np.insert(post.rk, k - 1, 0, axis=0)
            new_nn[k - 1:k + 1] = np.sum(new_rnk, axis=0)
            after_eta1 = 1. + new_nn
            after_eta0 = after_gamma1 / after_gamma2 + convert_to_n0(new_nn)
            params["rk"] = np.sum(new_rnk, axis=0)
            params["eta0"] = after_eta0[k-1:k+1]
            params["eta1"] = after_eta1[k-1:k+1]
            params["gamma1"] = after_gamma1
            params["gamma2"] = after_gamma2

            gap_sb = - c_alpha(after_gamma1, after_gamma2, after_gamma1 / after_gamma2)
            gap_sb += c_alpha(post.gamma1, post.gamma2, post.gamma1 / post.gamma2)
            gap_sb += - c_beta(eta1=after_eta1, eta0=after_eta0)
            gap_sb += c_beta(eta1=post.eta1, eta0=post.eta0)
        else:
            # Splitting Cluster k into cluster k, k+1.
            """
            new_nn = np.insert(post.nn, k-1, 0, axis=0)
            new_nn[k-1:k+1] = np.sum(new_rnk, axis=0)
            after_eta1 = 1. + new_nn
            after_eta0 = prior.alpha0 + convert_to_n0(new_nn)
            gap_sb = c_beta(eta1=1, eta0=prior.alpha0)
            gap_sb += - c_beta(eta1=after_eta1[k - 1:k + 1], eta0=after_eta0[k - 1:k + 1])
            gap_sb += c_beta(eta1=post.eta1[k - 1], eta0=post.eta0[k - 1])
            """
            new_nn = np.sum(new_rnk, axis=0)
            after_eta1 = 1. + new_nn
            after_eta0 = post.eta0[k - 1] * np.ones(2)
            after_eta0[0] += new_nn[1]

            params["rk"] = np.sum(new_rnk, axis=0)
            params["eta0"] = after_eta0[k-1:k+1]
            params["eta1"] = after_eta1[k-1:k+1]

            gap_sb = c_beta(eta1=1, eta0=prior.alpha0)
            gap_sb += - c_beta(eta1=after_eta1, eta0=after_eta0)
            gap_sb += c_beta(eta1=post.eta1[k - 1], eta0=post.eta0[k - 1])

        # Calculating Entropy GAP terms
        gap_entropy = - calc_entropy(new_rnk) + calc_entropy(rnk_k)

        gap = np.array([0, gap_sb, gap_entropy, gap_obs])
        evaluation = (gap_obs + gap_sb + gap_entropy) > 0

        return gap, evaluation, params
    # ##############################################################
    # ##############################################################

    # #########################################################################################################
    # Abstract Methods
    @abstractmethod
    def calc_all_entropy(self):
        raise NotImplementedError

    @abstractmethod
    def get_rnk_k(self, k):
        raise NotImplementedError

    @abstractmethod
    def vb_update_local(self):
        raise NotImplementedError

    @abstractmethod
    def vb_update_global(self):
        raise NotImplementedError

    @abstractmethod
    def find_nearby_centers(self, target_radius):
        raise NotImplementedError
    # #########################################################################################################


class TimeDepentModelPython(TimeDependentModel):

    def __init__(self, **kwargs):
        super(TimeDepentModelPython, self).__init__(**kwargs)
        self.active_c = []

    def calc_all_entropy(self):
        return calc_entropy(self.Post.rnk)

    def get_rnk_k(self, k):
        return self.Post.rnk[:, k]

    def find_nearby_centers(self, target_radius):
        p_qt = Tree(self.Post.mu)
        nearby = list()
        for c_ in np.arange(self.Post.C):
            centers = p_qt.query_ball_point(self.Post.mu[c_], target_radius)
            centers.remove(c_)
            if len(centers) > 1:
                centers = np.sort(centers)[::-1]
                elements = centers[[not (x in nearby) for x in centers]]
                [nearby.append(e_) for e_ in elements]

        return np.sort(nearby) + 1

    # INFERENCE ROUTINES
    def vb_update_local(self):

        # Bind Data
        x = self.x
        sigma2_x = self.sigma2_x
        post = self.Post
        prior = self.Prior

        if post.K == 1:
            post.setField('rnk', np.ones((post.N, post.K)), dims=('N', 'K'))

        else:
            # Pi1
            if self.infer_pi1 is False:
                value0 = np.log(prior.pi1 / (1 - prior.pi1)) + prior.logp0
                l_1mpi1 = 0.
            else:
                l_pi1 = psi(post.a)  # - psi(self.Post.a + self.Post.b)
                l_1mpi1 = psi(post.b)  # - psi(self.Post.a + self.Post.b)
                value0 = l_pi1 - l_1mpi1 + prior.logp0
                l_1mpi1 = 0.

            # rnk update
            elog_n_nk = e_log_n(x, sigma2_x, post.mu, post.sigma2)
            tree = Tree(post.mu)

            rnk = -np.inf * np.ones((post.N, post.K))
            rnk[:, 0] = value0

            active_c = [[] for _ in np.arange(post.C)]
            for t_ in np.arange(self.T):
                idx_nt = np.where(self.time == t_)[0]
                if len(idx_nt) > 0:
                    points = self.x[idx_nt, :]
                    centers = np.int32(np.unique(np.concatenate(tree.query_ball_point(points, 30))))
                    for c_ in centers:
                        active_c[c_].append(t_)
                    # TO DO: VERIFY THAT I AM SUMMING ALL ACTIVE CENTERS AT TIME T.
                    elog_pi_k = digamma(post.alpha[centers]) - digamma(post.alpha[centers].sum())
                    rnk[np.ix_(idx_nt, centers + 1)] = l_1mpi1 + elog_n_nk[np.ix_(idx_nt, centers)] + elog_pi_k[None, :]

            self.active_c = active_c
            e_norm(rnk)
            post.setField('rnk', rnk, dims=('N', 'K'))

        post.setField('rn0_vector', post.rnk[:, 0], dims='N')
        post.setField('cobs', c_obs(x, sigma2_x, (1. - post.rn0_vector)), dims=None)

    def vb_update_global(self):
        # Bind data
        x = self.x
        sigma2_x = self.sigma2_x
        post = self.Post
        prior = self.Prior

        # Calculate Sufficient Stats
        post.setField('rk', np.sum(post.rnk[:, 1:], axis=0), dims='C')
        post.setField('xx', dotatb(post.rnk[:, 1:], x * np.reciprocal(sigma2_x)), dims=('C', 'D'))
        post.setField('sx', dotatb(post.rnk[:, 1:], np.reciprocal(sigma2_x)), dims=('C', 'D'))

        # sigma, mu
        post.setField('sigma2', (post.sx + prior.sigma02 ** -1) ** -1, dims=('C', 'D'))
        post.setField('mu', (post.xx + prior.mu0 * (prior.sigma02 ** -1)) * post.sigma2, dims=('C', 'D'))

        # alphakt
        post.setField('eta1', 1 + post.rk, dims='C')

        # pi1
        if self.infer_pi1 is True:
            post.setField('a', prior.a + np.sum(post.rn0_vector), dims=None)
            post.setField('b', prior.b + np.sum(1 - post.rn0_vector), dims=None)
    # ##############################################################
    # ##############################################################


class TimeDepentModelC(TimeDependentModel):
    def __init__(self, **kwargs):

        super(TimeDepentModelC, self).__init__(**kwargs)

        self.LC = LightC.LightC()
        # perm_forward, perm_reverse = self.LC.find_cache_friendly_permutation(np.float64(self.x).copy())
        # self.x = self.x[perm_forward, :]
        # self.sigma2_x = self.sigma2_x[perm_forward, :]
        self.LC.load_points(np.float64(self.x).copy(), np.float64(self.sigma2_x).copy())

    def calc_all_entropy(self):
        return self.LC.calc_entropy()

    def get_rnk_k(self, k):
        if k == 0:
            rnk_k = self.Post.rn0_vector
            idx_k = np.flatnonzero(rnk_k)
        else:
            rnk_k, idx_k = self.LC.get_rnk_given_c(k)
        out_rnk_k = np.zeros(self.Post.N)
        out_rnk_k[idx_k] = rnk_k

        return out_rnk_k

    def find_nearby_centers(self, target_radius):
        near = self.LC.get_nearby_centers(target_radius)
        collapsed_components = np.array(np.where(near)[0])

        return collapsed_components

    def vb_update_local(self, search_radius=4):
        # Bind Data
        post = self.Post
        prior = self.Prior

        if post.K > 1:
            # ###### LOCAL UPDATE
            # Pi1
            if self.infer_pi1 is False:
                value0 = np.log(prior.pi1 / (1 - prior.pi1)) + prior.logp0
                l_1mpi1 = 0.
            else:
                l_pi1 = psi(post.a)  # - psi(self.Post.a + self.Post.b)
                l_1mpi1 = psi(post.b)  # - psi(self.Post.a + self.Post.b)
                value0 = l_pi1 - l_1mpi1 + prior.logp0
                l_1mpi1 = 0.

            # rnk update
            elog_beta_k = e_log_beta(eta1=post.eta1, eta0=post.eta0)
            self.LC.load_centers(post.mu, post.sigma2, elog_beta_k)
            self.LC.build_kdtree_centers()
            xx, sx, rk, rn0_vector, cobs = self.LC.points_to_centers(l_1mpi1, value0, search_radius)

            # ###### GLOBAL UPDATE
            # mu, sigma2
            post.setField('sigma2', (sx + prior.sigma02 ** -1) ** -1, dims=('C', 'D'))
            post.setField('mu', (xx + prior.mu0 * (prior.sigma02 ** -1)) * post.sigma2, dims=('C', 'D'))
            post.setField('rk', rk, dims='C')
            post.setField('rn0_vector', rn0_vector, dims='N')
            post.setField('cobs', cobs, dims=None)

            # alphakt
            post.setField('eta1', 1 + rk, dims='C')
            if self.infer_alpha0 is True:
                post.setField('eta0', post.gamma1 / post.gamma2 + convert_to_n0(rk), dims='C')
            else:
                post.setField('eta0', prior.alpha0 + convert_to_n0(rk), dims='C')
            elog_u, elog1m_u = calc_beta_expectations(post.eta1, post.eta0)
            post.setField('elog_u', elog_u, dims='C')
            post.setField('elog1m_u', elog1m_u, dims='C')
            if self.infer_alpha0 is True:
                post.setField('gamma1', prior.gamma1 + post.C, dims=None)
                post.setField('gamma2', prior.gamma2 - post.elog1m_u.sum(), dims=None)

            # pi1
            if self.infer_pi1 is True:
                post.setField('a', prior.a + post.rn0_vector.sum(), dims=None)
                post.setField('b', prior.b + (1. - post.rn0_vector).sum(), dims=None)

    def vb_update_global(self):
        pass
    # ##############################################################
    # ##############################################################
