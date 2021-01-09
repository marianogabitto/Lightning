#ifndef __CENTER_HH
#define __CENTER_HH

#include "PointBase.hh"
#include "PointToCenter.hh"
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#define OMP_LOCK_DECLARE(lock)  omp_lock_t lock
#define OMP_LOCK_INIT(lock) omp_init_lock(&lock)
#define OMP_LOCK_SET(lock) omp_set_lock(&lock)
#define OMP_LOCK_UNSET(lock) omp_unset_lock(&lock)
#else
#define OMP_LOCK_DECLARE(lock)
#define OMP_LOCK_INIT(lock)
#define OMP_LOCK_SET(lock)
#define OMP_LOCK_UNSET(lock)
#endif

//
// Center: all information associated with a center
//         PointBase gives x/y coordinates
//
template<int D>
struct Center : public PointBase<D>
{
    // Index of center in the array
    int ix_;

    // Omp lock variable
    OMP_LOCK_DECLARE(lock);

    // Variances
    double sigma2_[D];

    // Elog_beta iteration parameter
    double Elog_beta_;

    // New values for center calculated during the iteration
    int npoints_;
    double rk_;
    double rk2_;
    double next_sigma2_[D];
    double next_x_[D];

    // Have we seen this center in this time step
    bool marked_;

    // Points that are associated with this center
    // NOTE: these are just pointers to the objects that are stored in Point::point_to_centers_
    //       so be VERY CAREFUL about memory management (!!!)
    std::vector<PointToCenter *> center_to_points_;
    
    // Initialize the center
    void init(int ix);
};


template<int D>
void Center<D>::init(int ix)
{
    ix_ = ix;
    npoints_ = 0;
    rk_ = 0.0;
    rk2_ = 0.0;
    for (int id = 0; id < D; id++) {
        next_sigma2_[id] = 0.0;
        next_x_[id] = 0.0;
    }
    marked_ = false;
    OMP_LOCK_INIT(lock);
}


typedef Center<2> Center2D;


#endif
