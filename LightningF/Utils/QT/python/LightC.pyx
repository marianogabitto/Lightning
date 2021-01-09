import numpy as np
cimport numpy as np
from libc.math cimport log
import sys


cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        vector()
        void push_back(T&)
        T& operator[](int)
        T& at(int)
        iterator begin()
        iterator end()
        unsigned size() const

        
cdef extern from "Light.hh":

    cdef cppclass Center2D:
        int npoints_
        vector[PointToCenter *] center_to_points_
        
    cdef cppclass PointToCenter:
        int point_ix
        int center_ix
        double rnk_
        
    cdef cppclass Point2D:
        vector[PointToCenter] point_to_centers_

    cdef cppclass Light2D:
        int n_centers()
        const Center2D &center(int ix) const
        int n_points()
        const Point2D &point(int ix)
        void find_cache_friendly_permutation(int n, double *x, int *perm_forward, int *perm_reverse)
        void load_points(int np, double *x, double *sigma2)
        void load_centers(int nc, double *x, double *sigma2, double *elog_beta)
        void build_kdtree_centers()
        void points_to_centers(double l_1mpi1, double value0, double search_radius)
        void get_next_center_data(double *next_mu, double *next_sigma, double *rk, double *rn0, double &cobs)
        void get_points_to_centers(int point_ix[], int center_ix[], double log_nk[], double rnk[])
        void nearby_centers(int list_ix[], double search_radius)
        double calc_entropy()
        void get_rnk_given_k(int k)
        # Time Dependent Functions
        void get_time_data(double *acc_0, double *acc_n0, int *nt, double *xi_t)
        void load_points_time(int np, double *x, double *sigma2, int *time)
        void points_to_centers_time(int ntime, double *value0, double *value1, double *gamma, double search_radius)
        void get_next_center_data_time(double *next_mu, double *next_sigma, double *rk, double *rk2)
        void dump_state(const char *filename)
        

cdef class LightC:

    # Underlying C++ object
    cdef Light2D *light
    
    def __init__(self):
        self.light = new Light2D()

    def __dealloc__(self):
        del self.light
        self.light = NULL

    def find_cache_friendly_permutation(self, np.ndarray[double, ndim=2, mode="c"] x not None):
        cdef int n = x.shape[0]
        assert(x.shape[1] == 2)

        # Forward and reverse permutations
        cdef np.ndarray[int, ndim=1, mode="c"] perm_forward
        cdef np.ndarray[int, ndim=1, mode="c"] perm_reverse

        perm_forward = np.zeros(n, dtype=np.int32)
        perm_reverse = np.zeros(n, dtype=np.int32)

        self.light.find_cache_friendly_permutation(n, &x[0,0], &perm_forward[0], &perm_reverse[0])

        return perm_forward, perm_reverse

    def load_points(self, np.ndarray[double, ndim=2, mode="c"] x not None, np.ndarray[double, ndim=2, mode="c"] sigma2 not None):
        cdef int n = x.shape[0]
        assert(x.shape[0] == sigma2.shape[0])
        assert(x.shape[1] == 2)
        assert(sigma2.shape[1] == 2)
        self.light.load_points(n, &x[0,0], &sigma2[0,0])

    def load_centers(self, np.ndarray[double, ndim=2, mode="c"] x not None,
                     np.ndarray[double, ndim=2, mode="c"] sigma2 not None,
                     np.ndarray[double, ndim=1, mode="c"] elog_beta not None):
        cdef int n = x.shape[0]
        assert(x.shape[0] == sigma2.shape[0])
        assert(x.shape[0] == elog_beta.shape[0])
        assert(x.shape[1] == 2)
        assert(sigma2.shape[1] == 2)
        self.light.load_centers(n, &x[0,0], &sigma2[0,0], &elog_beta[0])
        
    def build_kdtree_centers(self):
        self.light.build_kdtree_centers()

    def points_to_centers(self, double l_1mpi1, double value0, double search_radius):
        self.light.points_to_centers(l_1mpi1, value0, search_radius)

        cdef int ncenters = self.light.n_centers()
        cdef int npoints = self.light.n_points()

        cdef np.ndarray[double, ndim=2, mode="c"] next_mu = np.zeros([ncenters, 2])
        cdef np.ndarray[double, ndim=2, mode="c"] next_sigma = np.zeros([ncenters, 2])
        cdef np.ndarray[double, ndim=1, mode="c"] rk = np.zeros([ncenters])
        cdef np.ndarray[double, ndim=1, mode="c"] rn0 = np.zeros([npoints])
        cdef double cobs = 0.;

        self.light.get_next_center_data(&next_mu[0, 0], &next_sigma[0, 0], &rk[0], &rn0[0], cobs)

        return next_mu, next_sigma, rk, rn0, cobs

    def calc_entropy(self):
        return self.light.calc_entropy()

    def get_nearby_centers(self, double search_radius):
        cdef np.ndarray[int, ndim=1, mode="c"] list_ix
        list_ix = np.zeros(self.light.n_centers(), dtype=np.int32)
        self.light.nearby_centers(&list_ix[0], search_radius)

        return list_ix


    def retrieve_points_to_centers(self):
        # Build the list of ndarray k indices
        # Note: This is slow - it would be better to use an ndarray here (!!!)
        klist = []
        cdef int npoints = self.light.n_points()
        cdef int ix
        cdef const Point2D *p
        cdef int nc
        cdef int ncmax = 1

        kpoints = list()
        for _ in np.arange(self.light.n_centers()):
            kpoints.append([])

        for ixp in range(npoints):
            p = &self.light.point(ixp)

            nc = p.point_to_centers_.size()
            if nc > ncmax:
                ncmax = nc

            l = []
            for ixc in range(nc):
                l.append(p.point_to_centers_[ixc].center_ix)
                kpoints[p.point_to_centers_[ixc].center_ix].append(ixp)
            klist.append(l)

        # Build rnk:
        cdef np.ndarray[double, ndim=2, mode="c"] rnk = np.zeros([npoints, ncmax])

        for ixp in range(npoints):
            p = &self.light.point(ixp)

            nc = p.point_to_centers_.size()
            for ixc in range(nc):
                rnk[ixp, ixc] = p.point_to_centers_[ixc].rnk_

        return klist, rnk, kpoints

    
    def get_rnk_given_c(self, int k):
        cdef const Center2D *center = &self.light.center(k-1)
        cdef int ncp = center.center_to_points_.size()

        cdef np.ndarray[int, ndim=1, mode="c"] idx_k = np.zeros([ncp], dtype=np.int32)
        cdef np.ndarray[double, ndim=1, mode="c"] rnk = np.zeros([ncp], dtype=np.double)

        for ix in range(ncp):
            idx_k[ix] = center.center_to_points_[ix].point_ix
            rnk[ix] = center.center_to_points_[ix].rnk_

        return rnk, idx_k

    # ################################################
    # TIME ROUTINES
    def load_points_time(self, np.ndarray[double, ndim=2, mode="c"] x not None, np.ndarray[double, ndim=2, mode="c"] sigma2 not None, np.ndarray[int, ndim=1, mode="c"] time not None):
        cdef int n = x.shape[0]
        assert(x.shape[0] == sigma2.shape[0])
        assert(x.shape[0] == time.shape[0])
        assert(x.shape[1] == 2)
        assert(sigma2.shape[1] == 2)
        self.light.load_points_time(n, &x[0,0], &sigma2[0,0], &time[0])

    def points_to_centers_time(self, np.ndarray[double, ndim=1, mode="c"] value0 not None,
                                     np.ndarray[double, ndim=1, mode="c"] value1 not None,
                                     np.ndarray[double, ndim=1, mode="c"] gamma not None,
                                     double search_radius):
        cdef int ntime = value0.shape[0]
        assert(value1.shape[0] == ntime)

        cdef int ncenters = self.light.n_centers()
        assert(gamma.shape[0] == ncenters)

        self.light.points_to_centers_time(ntime, &value0[0], &value1[0], &gamma[0], search_radius)

        cdef np.ndarray[double, ndim=2, mode="c"] next_mu = np.zeros([ncenters, 2])
        cdef np.ndarray[double, ndim=2, mode="c"] next_sigma = np.zeros([ncenters, 2])
        cdef np.ndarray[double, ndim=1, mode="c"] rk = np.zeros([ncenters])
        cdef np.ndarray[double, ndim=1, mode="c"] rk2 = np.zeros([ncenters])

        cdef np.ndarray[double, ndim=1, mode="c"] acc_0 = np.zeros([ntime])
        cdef np.ndarray[double, ndim=1, mode="c"] acc_n0 = np.zeros([ntime])
        cdef np.ndarray[int, ndim=1, mode="c"] nt = np.zeros([ntime])
        cdef np.ndarray[double, ndim=1, mode="c"] xi_t = np.zeros([ntime])

        self.light.get_next_center_data_time(&next_mu[0, 0], &next_sigma[0, 0], &rk[0], &rk2[0])
        self.light.get_time_data(&acc_0[0], &acc_n0[0], &nt[0], &xi_t[0])

        return next_mu, next_sigma, rk, rk2, acc_0, acc_n0, nt, xi_t


    def dump_state(self, filename):
        self.light.dump_state(bytes(filename, 'utf-8'))

