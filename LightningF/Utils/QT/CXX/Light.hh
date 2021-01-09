#ifndef __LIGHT_HH
#define __LIGHT_HH

#include "Point.hh"
#include "Center.hh"
//#include "QTree.hh"
#include "KDTree.hh"
#include <vector>
#include <iostream>
#include <fstream>

#include <omp.h>
#include "CounterIntEvent.hh"
#include <cmath>
#include <limits>
#include <iomanip>


struct TimeInfo
{
    double acc_0;
    double acc_n0;

    int nt;
    double xi_t;
};


template<int D>
class Light
{
    std::vector< Point<D> > points_;
    std::vector< Center<D> > centers_;

    std::vector< TimeInfo > time_data_;
    
    // Tree of centers
    //QTree *qtree_;
    KDTree<D, Center<D> > *kdtree_;

    // Noise cluster accumulator
    double cobs_;
    double ent_;
    
public:
    Light();
    ~Light();
    
    //QTree *qtree() { return qtree_; }

    // Return the number of centers
    int n_centers() const { return centers_.size(); }

    // Return a specific center
    const Center<D> &center(int ix) const { return centers_[ix]; }

    // Return the number of points
    int n_points() const  { return points_.size(); }

    // Return a specific point
    const Point<D> &point(int ix) const { return points_[ix]; }

    // Clear structures that match points to centers
    void clear_points_to_centers();
    
    // Load points from a file
    void load_points(std::istream &is);

    // Load points (position/variance)
    void load_points(int np, double *x, double *sigma2);

    // Load points (position/variance/time)
    void load_points_time(int np, double *x, double *sigma2, int *time);

    // Load centers from a file
    void load_centers(std::istream &is);

    // Load points from an array with 4 columns (x, y, sigmax2, sigmay2)
    void load_centers(int nc, double *x, double *sigma2, double *Elog_beta);

    // Set the iteration parameters
    void set_iter_params(double l_1mpi1, double value0, double search_radius);

#if 0
    // Build the quad tree of centers
    void build_qtree_centers(double x1, double y1, double x2, double y2, double min_cell_size);
#endif
    
    // Build the kdtree of centers
    void build_kdtree_centers();
    
    // Assign the points to centers
    void points_to_centers(double l_1mpi1, double value0, double search_radius);
    void points_to_centers_time(int ntime, double *value0, double *value1, double *gamma, double search_radius);

    // Retrieve the new center mean/variance info
    void get_next_center_data(double *next_mu, double *next_sigma, double *rk, double *rn0, double &cobs);
    void get_next_center_data_time(double *next_mu, double *next_sigma, double *rk, double *rk2);

    void calc_psi_t(double *gamma, double *psi_t);

    // Retrieve the per time step data
    void get_time_data(double *acc_0, double *acc_n0, int *nt, double *xi_t);

    // Retrieve the points to centers mapping
    int get_n_points_to_centers();
    void get_points_to_centers(int point_ix[], int center_ix[], double logNnk[], double rnk[]);

#if 0
    // Print statistics of the tree (how many points per leaf are there bucketed)
    void print_qtree_leaf_stats();
#endif
    
    // Retrieve centers with a close-by neighbor
    void nearby_centers(int center_list[], double search_radius);

    // Find cache friendly permutation of a set of points
    void find_cache_friendly_permutation(int N, double *x, int *perm_forward, int *perm_reverse);

    // Counts Points to center
    void counters(int list_points[], int list_centers[], double search_radius);

    // Calculate total entropy
    double calc_entropy();

    // Dump the state of the C++ object to a file
    void dump_state(std::ostream &f);
    void dump_state(const char *filename);
};


template<int D>
Light<D>::Light()
    : /*qtree_(NULL),*/ kdtree_(NULL)
{
}


template<int D>
Light<D>::~Light()
{
    // delete qtree_; qtree_ = NULL;
    delete kdtree_; kdtree_ = NULL;
}


template<int D>
void Light<D>::clear_points_to_centers()
{
    for (int ix = 0; ix < (int)centers_.size(); ix++) {
        Center<D> &c = centers_[ix];
        c.center_to_points_.clear();
    }
    for (int ix = 0; ix < (int)points_.size(); ix++) {
        Point<D> &p = points_[ix];
        p.point_to_centers_.clear();
    }
}


template<int D>
void Light<D>::load_points(std::istream &is)
{
    int np;
    is >> np;

    points_.resize(np);
    for (int ix = 0; ix < np; ix++) {
        Point<D> *p = &points_[ix];
        is >> p->x_[0] >> p->x_[1] >> p->sigma2_[0] >> p->sigma2_[1];
        p->init(ix);
    }
}


template<int D>
void Light<D>::load_points(int np, double *x, double *sigma2)
{
    points_.resize(np);
    for (int ix = 0; ix < np; ix++) {
        Point<D> *p = &points_[ix];
        p->x_[0] = x[2*ix+0];
        p->x_[1] = x[2*ix+1];
        p->sigma2_[0] = sigma2[2*ix+0];
        p->sigma2_[1] = sigma2[2*ix+1];
        p->init(ix);
    }
}


template<int D>
void Light<D>::load_points_time(int np, double *x, double *sigma2, int *time)
{
    points_.resize(np);
    for (int ix = 0; ix < np; ix++) {
        Point<D> *p = &points_[ix];
        p->x_[0] = x[2*ix+0];
        p->x_[1] = x[2*ix+1];
        p->sigma2_[0] = sigma2[2*ix+0];
        p->sigma2_[1] = sigma2[2*ix+1];
        p->time_ = time[ix];
        p->init(ix);
    }
}


template<int D>
void Light<D>::load_centers(std::istream &is)
{
    int nk;
    is >> nk;

    centers_.resize(nk);
    for (int ix = 0; ix < nk; ix++) {
        int cix;
        is >> cix;
        Center<D> *center = &centers_[ix];
        Point<D> *point = &points_[cix];
        center->x_[0] = point->x_[0];
        center->x_[1] = point->x_[1];
        center->sigma2_[0] = point->sigma2_[0];
        center->sigma2_[1] = point->sigma2_[1];
        center->init(ix);
    }

    clear_points_to_centers();
}


template<int D>
void Light<D>::load_centers(int nc, double *x, double *sigma2, double *Elog_beta)
{
    centers_.resize(nc);
    for (int ix = 0; ix < nc; ix++) {
        Center<D> *c = &centers_[ix];
        c->x_[0] = x[2*ix+0];
        c->x_[1] = x[2*ix+1];
        c->sigma2_[0] = sigma2[2*ix+0];
        c->sigma2_[1] = sigma2[2*ix+1];
        c->Elog_beta_ = Elog_beta[ix];
        c->init(ix);
    }

    clear_points_to_centers();
}


#if 0
void Light::build_qtree_centers(double x1, double y1, double x2, double y2, double min_cell_size)
{
    // If there is a previous tree, remove it first
    if (qtree_ != NULL) {
        delete qtree_;
    }
    
    qtree_ = new QTree(x1, y1, x2, y2, min_cell_size);

    for (size_t ix = 0; ix < centers_.size(); ix++) {
        qtree_->add_point(&centers_[ix]);
    }
}
#endif


template<int D>
void Light<D>::build_kdtree_centers()
{
    if (kdtree_ != NULL) {
        delete kdtree_;
    }
    
    kdtree_ = new KDTree<D, Center<D> >(centers_);
}


//
// Visitor class
//
template<int D>
class NearbyCenterVisitor : public NeighborPointVisitor<D>
{
public:
    // Point we are looking around
    Point<D> *point_;

    double search_radius_;
    
    // Distance (squared) we are interested in
    double dist2_;
    
    double l_1mpi1_;
    
    double max_logNnk_;
    
    NearbyCenterVisitor(double l_1mpi1, double search_radius)
        : point_(NULL), search_radius_(search_radius), l_1mpi1_(l_1mpi1),
          max_logNnk_(-std::numeric_limits<double>::infinity())
    {
    }

    void set_point(Point<D> *point)
    {
        point_ = point;
        dist2_ = search_radius_ * search_radius_ / 2.0 * (point_->sigma2_[0] + point_->sigma2_[1]);
    }
    
    virtual void visit_point(PointBase<D> &p);

    // Add the noise term at position 0
    void visit_noise(double value0);
};


/*virtual*/
template<int D>
void NearbyCenterVisitor<D>::visit_point(PointBase<D> &p)
{
    Center<D> &center = (Center<D> &)p;
    
    double dx = center.x_[0] - point_->x_[0];
    double dy = center.x_[1] - point_->x_[1];
    double dx2 = dx*dx;
    double dy2 = dy*dy;
    double d2 = dx2 + dy2;

    // Only look at points within 5 standard deviations on average in x/y directions
    if (d2 < dist2_) {
        
        PointToCenter ptc;
        ptc.center_ix = center.ix_;
        ptc.point_ix = point_->ix_;
        double sxi = 0.5 / point_->sigma2_[0];
        double syi = 0.5 / point_->sigma2_[1];
        ptc.logNnk_ = point_->logvar_ - dx2 * sxi - dy2 * syi - center.sigma2_[0] * sxi - center.sigma2_[1] * syi
            + l_1mpi1_ + center.Elog_beta_;

        // Calculate the maximum along the way
        if (ptc.logNnk_ > max_logNnk_) {
            max_logNnk_ = ptc.logNnk_;
        }
        point_->point_to_centers_.push_back(ptc);
	PointToCenter &ptc_new = point_->point_to_centers_[point_->point_to_centers_.size()-1];
        center.center_to_points_.push_back(&ptc_new);
	// std::cout << "INSERTING: " << center.ix_ << ", " << point_->ix_ << std::endl;
    }
}


template<int D>
void NearbyCenterVisitor<D>::visit_noise(double value0)
{
    PointToCenter ptc;

    ptc.center_ix = -1;
    ptc.point_ix = point_->ix_;
    ptc.logNnk_ = value0;
    
    // Calculate the maximum along the way
    if (ptc.logNnk_ > max_logNnk_) {
        max_logNnk_ = ptc.logNnk_;
    }
    point_->point_to_centers_.push_back(ptc);
}



//
// Visitor class to check centers that are close to each other
//
template<int D>
class CenterToCenterVisitor : public NeighborPointVisitor<D>
{
public:
    // Point we are looking around
    double search_radius_;
    Point<D> &orig_;

    CenterToCenterVisitor(double search_radius, Point<D> &orig)
        : search_radius_(search_radius), orig_(orig)
    {
    }

    virtual void visit_point(PointBase<D> &p);
};


/*virtual*/
template<int D>
void CenterToCenterVisitor<D>::visit_point(PointBase<D> &p)
{
    Center<D> &center = (Center<D> &)p;
    
    double dx = center.x_[0] - orig_.x_[0];
    double dy = center.x_[1] - orig_.x_[1];
    double dx2 = dx*dx;
    double dy2 = dy*dy;
    double d2 = dx2 + dy2;

    // Only look at points within 5 standard deviations on average in x/y directions
    if (d2 < search_radius_ * search_radius_) {
        
        PointToCenter ptc;
        ptc.center_ix = center.ix_;
        orig_.point_to_centers_.push_back(ptc);
    }
}


template<int D>
void Light<D>::nearby_centers(int list_ix[], double search_radius)
{
    // It looks for centers that have a neighboor closer than search radius
    //
    // Initialize container with no clusters nearby
    std::vector<int> flag(centers_.size(), 0);
    for (size_t ixc = 0; ixc < centers_.size(); ixc++) {
        if (flag[ixc] == 0){
            // Initialize a point at the center location

            Point<D> p;
            p.x_[0] = centers_[ixc].x_[0];
            p.x_[1] = centers_[ixc].x_[1];

            CenterToCenterVisitor<D> closest_center(search_radius, p);

            // Look at the real centers next
#if 0
            if (qtree_ != NULL) {
                qtree_->visit_neighbor_points(p.x_[0], p.x_[1], closest_center);
            }
#endif
            if (kdtree_ != NULL) {
                double dist = search_radius;
                PointBase<D> min, max;
                min.x_[0] = p.x_[0] - dist;
                min.x_[1] = p.x_[1] - dist;
                max.x_[0] = p.x_[0] + dist;
                max.x_[1] = p.x_[1] + dist;
                kdtree_->find_points(min, max, closest_center);
            }

            size_t nc = p.point_to_centers_.size();
            for (size_t ix = 0; ix < nc; ix++) {
                PointToCenter &ptc = p.point_to_centers_[ix];
                // Make sure we ignore the center being close to itself
                if (ptc.center_ix != (int)ixc) {
                    // Center<D> &c = centers_[ptc.center_ix];
                    list_ix[ixc] = 1;
                    flag[ixc] = 1;
                    flag[ptc.center_ix] = 1;
                    // std::cout << "Origin #" << ixc << ", x:" << p.x_[0] << ", y:" << p.x_[1] << std::endl;
                    // std::cout << "Center #" << ptc.center_ix << ", x:" << c.x_[0] << ", y:" << c.x_[1] << std::endl;
                }
            }
        }
    }
}


template<int D>
void Light<D>::points_to_centers(double l_1mpi1, double value0, double search_radius)
{
    cobs_ = 0.0;
    //std::cout << "l_1mpi1" << l_1mpi1 << ", value0:" << value0 << ", radius:" << search_radius << std::endl;
#pragma omp parallel for schedule(static)
    for (size_t ixp = 0; ixp < points_.size(); ixp++) {
        //int tid = omp_get_thread_num();
        //printf("Running ixp %d in thread: %d\n", ixp, tid);
        
        // if (ixp % 10000 == 0) std::cout << "Point #" << ixp << std::endl;
        Point<D> &p = points_[ixp];

        // std::cout << "Point #" << ixp << ", x:" << p.x_[0] << ", y:" << p.x_[1] << std::endl;
        
        NearbyCenterVisitor<D> closest_center(l_1mpi1, search_radius);
        closest_center.set_point(&p);
        
        // Add the noise virst
        closest_center.visit_noise(value0);

        // Look at the real centers next
#if 0
        if (qtree_ != NULL) {
            qtree_->visit_neighbor_points(p.x_[0], p.x_[1], closest_center);
        }
#endif
        if (kdtree_ != NULL) {
            // std::cout << closest_center.dist2_ <<std::endl;
            double dist = sqrt(closest_center.dist2_);
            PointBase<D> min, max;
            min.x_[0] = p.x_[0] - dist;
            min.x_[1] = p.x_[1] - dist;
            max.x_[0] = p.x_[0] + dist;
            max.x_[1] = p.x_[1] + dist;
            kdtree_->find_points(min, max, closest_center);
        }

        size_t nc = p.point_to_centers_.size();
        // std::cout << nc << std::endl;
        if (nc == 1) {

            // No real centers found, point belongs to noise with p=1
            PointToCenter &ptc = p.point_to_centers_[0];
            ptc.rnk_ = 1.0;
            
        } else {

            // We compute rnk_ (including the noise element 0)
            double max_logNnk = closest_center.max_logNnk_;
            double sum = 0.0;
            for (size_t ix = 0; ix < nc; ix++) {
                PointToCenter &ptc = p.point_to_centers_[ix];
                if (ptc.center_ix >= 0) {
                    Center<D> &c = centers_[ptc.center_ix];
                    c.npoints_++;
                }
                // std::cout << ptc.logNnk_ - max_logNnk << std::endl;
                if ((ptc.logNnk_ - max_logNnk) < -50.0){
                    ptc.rnk_ = 0.0;
                    }
                else{
                    ptc.rnk_ = exp(ptc.logNnk_ - max_logNnk);
                    }

                sum += ptc.rnk_;
            }

            // Normalize the noise rnk_ (since the next loop starts at 1!)
            p.point_to_centers_[0].rnk_ /= sum;
            
            // We compute the new center coordinates/variances.
            // we do not look at the noise (element 0) here
            for (size_t ix = 1; ix < nc; ix++) {
                PointToCenter &ptc = p.point_to_centers_[ix];
                ptc.rnk_ /= sum;
                
                int center_ix = ptc.center_ix;
                Center<D> &center = centers_[center_ix];

                //OMP_LOCK_SET(center.lock);

                center.rk_ += ptc.rnk_;
                double tempx = ptc.rnk_ / p.sigma2_[0];
                double tempy = ptc.rnk_ / p.sigma2_[1];
                center.next_sigma2_[0] += tempx;
                center.next_sigma2_[1] += tempy;
                center.next_x_[0] += tempx * p.x_[0];
                center.next_x_[1] += tempy * p.x_[1];

                //OMP_LOCK_UNSET(center.lock);
            }
        }

        // Accumulate r_{n0} 1.-r_{n0}
        PointToCenter &ptc = p.point_to_centers_[0];
        cobs_ += - 0.5 * (1.0 - ptc.rnk_) * (p.x_[0]*p.x_[0] / p.sigma2_[0] + p.x_[1]*p.x_[1] / p.sigma2_[1]
                //+ log(2 * PI * p.sigma2_[0]) + log(2 * PI * p.sigma2_[1]));
                + p.x2var_logvar_);
        //printf("Finished ixp %d in thread: %d\n", ixp, tid);
    }

}


template<int D>
void Light<D>::points_to_centers_time(int ntime, double *value0, double *value1, double *gamma, double search_radius)
{
    ent_ = 0;
    
    // Set the time accumulators up
    time_data_.resize(ntime);
    for (int ixt = 0; ixt < ntime; ixt++) {
        time_data_[ixt].acc_0 = 0;
        time_data_[ixt].acc_n0 = 0;
        time_data_[ixt].nt = 0;
        time_data_[ixt].xi_t = 0;
    }

    // Loop over time indexes
    for (size_t ixp = 0; ixp < points_.size(); ixp++) {
        
        int ixp0 = ixp;
        Point<D> &p0 = points_[ixp0];
        int time = p0.time_;

        TimeInfo &time_data_t = time_data_[time];
            
        // List of unique center indices seen at this time step
        std::vector<int> time_center_ix;
        
        // Loop over all points at the same time index
        for (; ixp < points_.size() && points_[ixp].time_ == time; ixp++) {

            Point<D> &p = points_[ixp];

            NearbyCenterVisitor<D> closest_center(value1[time], search_radius);
            closest_center.set_point(&p);

            // Add the noise virst
            closest_center.visit_noise(value0[time]);

            // Look at the real centers next
#if 0
            if (qtree_ != NULL) {
                qtree_->visit_neighbor_points(p.x_[0], p.x_[1], closest_center);
            }
#endif
            if (kdtree_ != NULL) {
                double dist = sqrt(closest_center.dist2_);
                PointBase<D> min, max;
                min.x_[0] = p.x_[0] - dist;
                min.x_[1] = p.x_[1] - dist;
                max.x_[0] = p.x_[0] + dist;
                max.x_[1] = p.x_[1] + dist;
                kdtree_->find_points(min, max, closest_center);
            }

            size_t nc = p.point_to_centers_.size();

            if (nc == 1) {
                
                // No real centers found, point belongs to noise with p=1
                PointToCenter &ptc = p.point_to_centers_[0];
                ptc.rnk_ = 1.0;
            
            } else {

                // We compute rnk_ (including the noise element 0)
                double max_logNnk = closest_center.max_logNnk_;
                double sum = 0.0;
                for (size_t ix = 0; ix < nc; ix++) {
                    PointToCenter &ptc = p.point_to_centers_[ix];
                    if (ptc.center_ix >= 0) {
                        Center<D> &c = centers_[ptc.center_ix];
                        c.npoints_++;

                        if (!c.marked_) {
                            c.marked_ = true;
                            time_center_ix.push_back(ptc.center_ix);
                        }
                    }
                    ptc.rnk_ = exp(ptc.logNnk_ - max_logNnk);
                    sum += ptc.rnk_;
                }

                // Normalize the noise rnk_ (since the next loop starts at 1!)
                p.point_to_centers_[0].rnk_ /= sum;
            
                // We compute the new center coordinates/variances.
                // we do not look at the noise (element 0) here
                for (size_t ix = 1; ix < nc; ix++) {
                    PointToCenter &ptc = p.point_to_centers_[ix];
                    ptc.rnk_ /= sum;
                
                    int center_ix = ptc.center_ix;
                    Center<D> &center = centers_[center_ix];
                    center.rk_ += ptc.rnk_;
                    double tempx = ptc.rnk_ / p.sigma2_[0];
                    double tempy = ptc.rnk_ / p.sigma2_[1];
                    center.next_sigma2_[0] += tempx;
                    center.next_sigma2_[1] += tempy;
                    center.next_x_[0] += tempx * p.x_[0];
                    center.next_x_[1] += tempy * p.x_[1];
                    if (ptc.rnk_ > 1e-10) {
                        ent_ += ptc.rnk_ * log(ptc.rnk_);
                    }
                }
            }

            double tmp = p.point_to_centers_[0].rnk_;
            time_data_t.acc_0 += tmp;
            time_data_t.acc_n0 += (1-tmp);
        }

        time_data_t.nt = (int)time_center_ix.size();
        for (size_t ixk = 0; ixk < time_center_ix.size(); ixk++) {
            time_data_t.xi_t += gamma[ time_center_ix[ixk] ];
        }

        for (size_t ixk = 0; ixk < time_center_ix.size(); ixk++) {
            Center<D> &c = centers_[ time_center_ix[ixk] ];
            c.rk2_ += 1 / time_data_t.xi_t;
            c.marked_ = false;
        }
    }
}


template<int D>
void Light<D>::calc_psi_t(double *gamma, double *psi_t)
{
    // Loop over time indexes
    for (size_t ixp = 0; ixp < points_.size(); ixp++) {
        
        int ixp0 = ixp;
        Point<D> &p0 = points_[ixp0];
        int time = p0.time_;

        // List of unique center indices seen at this time step
        std::vector<int> time_center_ix;
        
        // Loop over all points at the same time index
        for (; ixp < points_.size() && points_[ixp].time_ == time; ixp++) {

            Point<D> &p = points_[ixp];
            size_t nc = p.point_to_centers_.size();

            for (size_t ix = 0; ix < nc; ix++) {
                PointToCenter &ptc = p.point_to_centers_[ix];
                if (ptc.center_ix >= 0) {
                    Center<D> &c = centers_[ptc.center_ix];
                    
                    if (!c.marked_) {
                        c.marked_ = true;
                        time_center_ix.push_back(ptc.center_ix);
                        psi_t[time] += gamma[ptc.center_ix];
                    }
                }
            }
        }
        
        for (size_t ixk = 0; ixk < time_center_ix.size(); ixk++) {
            Center<D> &c = centers_[ time_center_ix[ixk] ];
            c.marked_ = false;
        }
    }
}


template<int D>
int Light<D>::get_n_points_to_centers()
{
    int n = 0;
    for (size_t ix = 0; ix < points_.size(); ix++) {
        n += points_[ix].point_to_centers_.size();
    }
    return n;
}


template<int D>
void Light<D>::get_next_center_data(double *next_mu, double *next_sigma, double *rk, double *rn0, double &cobs)
{
    int ix2 = 0;
    for (int ix = 0; ix < (int)centers_.size(); ix++, ix2 += 2) {
        next_mu[ix2+0] = centers_[ix].next_x_[0];
        next_mu[ix2+1] = centers_[ix].next_x_[1];
        next_sigma[ix2+0] = centers_[ix].next_sigma2_[0];
        next_sigma[ix2+1] = centers_[ix].next_sigma2_[1];
        rk[ix] = centers_[ix].rk_;
    }

    for (int ix = 0; ix < (int)points_.size(); ix++) {
      rn0[ix] = points_[ix].point_to_centers_[0].rnk_;
    }
	
    cobs = cobs_;
}


template<int D>
void Light<D>::get_next_center_data_time(double *next_mu, double *next_sigma, double *rk, double *rk2)
{
    int ix2 = 0;
    for (int ix = 0; ix < (int)centers_.size(); ix++, ix2 += 2) {
        next_mu[ix2+0] = centers_[ix].next_x_[0];
        next_mu[ix2+1] = centers_[ix].next_x_[1];
        next_sigma[ix2+0] = centers_[ix].next_sigma2_[0];
        next_sigma[ix2+1] = centers_[ix].next_sigma2_[1];
        rk[ix] = centers_[ix].rk_;
        rk2[ix] = centers_[ix].rk2_;
    }
}


template<int D>
void Light<D>::get_time_data(double *acc_0, double *acc_n0, int *nt, double *xi_t)
{
    for (int ixt = 0; ixt < (int)time_data_.size(); ixt++) {
        TimeInfo &time_data_t = time_data_[ixt];
        acc_0[ixt] = time_data_t.acc_0;
        acc_n0[ixt] = time_data_t.acc_n0;
        nt[ixt] = time_data_t.nt;
        xi_t[ixt] = time_data_t.xi_t;
    }
}


template<int D>
void Light<D>::get_points_to_centers(int point_ix[], int center_ix[], double logNnk[], double rnk[])
{
    int n = 0;
    for (size_t ix = 0; ix < points_.size(); ix++) {
        for(size_t ix2 = 0; ix2 < points_[ix].point_to_centers_.size(); ix2++) {
            PointToCenter &ptc = points_[ix].point_to_centers_[ix2];
            point_ix[n] = ptc.point_ix;
            center_ix[n] = ptc.center_ix;
            logNnk[n] = ptc.logNnk_;
            rnk[n] = ptc.rnk_;
            n++;
        }
    }
}


#if 0
void Light::print_qtree_leaf_stats()
{
    CounterIntEvent counter(CounterIntEvent::SCALE_LOG2);
    qtree_->points_per_leaf_stats(counter);
    std::cout << "    Centers-per-leaf stats: " << counter << std::endl;
}
#endif


template<int D>
void Light<D>::find_cache_friendly_permutation(int N, double *x, int *perm_forward, int *perm_reverse)
{
    std::vector< Point<D> > data(N);

    for (int i = 0; i < N; i++) {
        data[i].x_[0] = x[2*i];
        data[i].x_[1] = x[2*i+1];
        data[i].ix_ = i;
    }

    KDTree<D, Point<D> > tree(data);
    std::vector< Point<D> * > &data2 = tree.data();

    for (int i = 0; i < N; i++) {
        perm_forward[i] = data2[i]->ix_;
        perm_reverse[ data2[i]->ix_ ] = i;
    }
}


template<int D>
void Light<D>::counters(int list_points[], int list_centers[], double search_radius)
{
    for (size_t ixp = 0; ixp < points_.size(); ixp++) {
            // Initialize a point at the center location
            Point<D> p;
            p.x_[0] = points_[ixp].x_[0];
            p.x_[1] = points_[ixp].x_[1];

            CenterToCenterVisitor<D> closest_center(search_radius, p);

            // Look at the real centers next
#if 0
            if (qtree_ != NULL) {
                qtree_->visit_neighbor_points(p.x_[0], p.x_[1], closest_center);
            }
#endif
            if (kdtree_ != NULL) {
                double dist = search_radius;
                PointBase<D> min, max;
                min.x_[0] = p.x_[0] - dist;
                min.x_[1] = p.x_[1] - dist;
                max.x_[0] = p.x_[0] + dist;
                max.x_[1] = p.x_[1] + dist;
                kdtree_->find_points(min, max, closest_center);
            }

            size_t nc = p.point_to_centers_.size();
            list_points[ixp] = nc;
            for (size_t ix = 0; ix < nc; ix++) {
                PointToCenter &ptc = p.point_to_centers_[ix];
                    list_centers[ptc.center_ix] += 1;
            }
    }
}


template<int D>
double Light<D>::calc_entropy()
{
    double h = 0.0;
    
    for (int ixp = 0; ixp < (int)points_.size(); ixp++) {
	const Point<D> &p = points_[ixp];
	for (int ixpc = 0; ixpc < (int)p.point_to_centers_.size(); ixpc++) {
	    double r = p.point_to_centers_[ixpc].rnk_;
	    if (r >= 1e-10) {
		h += r*log(r);
	    }
	}
    }

    return h;
}


template<int D>
void Light<D>::dump_state(std::ostream &f)
{
    f << std::setprecision(4);
    f << "Points[" << points_.size() << "]" << std::endl;
    for(auto it = points_.begin(); it != points_.end(); ++it) {
	f << "    Point: ix:" << it->ix_
	  << ", x:[" << it->x_[0] << ", " << it->x_[1]
	  << "], sigma2:[" << it->sigma2_[0] << ", " << it->sigma2_[1]
	  << "], logvar:" << it->logvar_
	  << ", x2var_logvar:" << it->x2var_logvar_
	  << std::endl;
	for (auto it2 = it->point_to_centers_.begin(); it2 != it->point_to_centers_.end(); ++it2) {
	    f << "        P2C: p:" << it2->point_ix
	      << ", c:" << it2->center_ix
	      << ", logNnk:" << it2->logNnk_
	      << ", rnk:" << it2->rnk_
	      << std::endl;
	}
    }

    f << "Centers[" << centers_.size() << "]" << std::endl;
    for(auto it = centers_.begin(); it != centers_.end(); ++it) {
	f << "    Center: ix:" << it->ix_
	  << ", x:[" << it->x_[0] << ", " << it->x_[1]
	  << "], sigma2:[" << it->sigma2_[0] << ", " << it->sigma2_[1]
	  << "], Elog_beta:" << it->Elog_beta_
	  << ", rk:" << it->rk_
	  << ", rk2:" << it->rk2_
	  << ", next_x:[" << it->next_x_[0] << ", " << it->next_x_[1]
	  << "], next_sigma2:[" << it->next_sigma2_[0] << ", " << it->next_sigma2_[1]
	  << "]" << std::endl;
	for (auto it2 = it->center_to_points_.begin(); it2 != it->center_to_points_.end(); ++it2) {
	    f << "        P2C: p:" << (*it2)->point_ix
	      << ", c:" << (*it2)->center_ix
	      << ", logNnk:" << (*it2)->logNnk_
	      << ", rnk:" << (*it2)->rnk_
	      << std::endl;
	}
    }
}


template<int D>
void Light<D>::dump_state(const char *filename)
{
    std::ofstream f(filename);
    dump_state(f);
}


typedef Light<2> Light2D;

#endif
