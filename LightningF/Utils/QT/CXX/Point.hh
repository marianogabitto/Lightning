#ifndef __POINT_HH
#define __POINT_HH

#include "PointBase.hh"
#include "PointToCenter.hh"
#include "Center.hh"
#include <deque>
#include <cmath>


//
// Point: all information associated with a point
//         PointBase gives x/y coordinates
//
template<int D>
struct Point : public PointBase<D>
{
    // Index of point in the array
    int ix_;
    
    // Variances
    double sigma2_[D];

    // Time index
    int time_;
    
    // Precomputed log variance: -log(2*M_PI) - 0.5*log(p->sigma2x_*p->sigma2y_);
    double logvar_;
    double x2var_logvar_;
    
    // Centers that this point is associated with
    // (can't be a vector<> because its elements are referenced by pointer in Center::center_to_points)
    std::deque<PointToCenter> point_to_centers_;

    // Initialize the point
    void init(int ix);
};


template<int D>
void Point<D>::init(int ix)
{
    ix_ = ix;

    time_ = 0;
    
    double sigma2_all = sigma2_[0];
    for (int ixd = 1; ixd < D; ixd++) {
	sigma2_all *= sigma2_[ixd];
    }
    logvar_ = - D/2.0 * log(2*M_PI) - 0.5*log(sigma2_all);
    x2var_logvar_ = - 2. * logvar_;
}


typedef Point<2> Point2D;


#endif
