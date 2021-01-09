#ifndef __KDTREE_LEVEL_HH
#define __KDTREE_LEVEL_HH

#include "PointBase.hh"
#include "NeighborVisitor.hh"
#include <vector>
#include <algorithm>
#include <ostream>


template<int D, class TPoint>
class KDTreeLevel
{
    typedef std::vector<TPoint *> TVector;
    typedef typename TVector::iterator TVectorIt;
    
    // Slice of the array this node represents
    TVectorIt it_begin_;
    TVectorIt it_end_;

    // Dimension this array is split on
    int split_dim_;

    // Value we split at
    double split_value_;

    // Children
    KDTreeLevel<D, TPoint> *children_[2];
    
    class SortCompare
    {
        int sort_dim_;
        
    public:
        SortCompare(int sort_dim):
            sort_dim_(sort_dim)
        {}

        bool operator()(const TPoint *a, const TPoint *b)
        {
            return a->x_[sort_dim_] < b->x_[sort_dim_];
        }
    };
        
public:
    KDTreeLevel(TVectorIt it_begin, TVectorIt it_end, int split_dim)
        : it_begin_(it_begin), it_end_(it_end), split_dim_(split_dim)
    {
        children_[0] = NULL;
        children_[1] = NULL;
    }

    
    void create()
    {
        // If there are too few points, we don't subdivide
        int n = it_end_ - it_begin_;
        if (n < 8) {
            split_dim_ = -1;
            return;
        }

        // Sort the points by the correct dimension
        SortCompare comp(split_dim_);
        std::sort(it_begin_, it_end_, comp);

        // Mid point of split
        TVectorIt it_mid = it_begin_ + n/2;

        // Value we split at
        split_value_ = (*it_mid)->x_[split_dim_];

        // The child gets split on the next dimension
        int child_split_dim_ = (split_dim_ == D-1 ? 0 : split_dim_ + 1);

        children_[0] = new KDTreeLevel(it_begin_, it_mid, child_split_dim_);
        children_[1] = new KDTreeLevel(it_mid, it_end_, child_split_dim_);

        children_[0]->create();
        children_[1]->create();
    }


    void print(std::ostream &os, int indent)
    {        
        if (split_dim_ < 0) {
            for (TVectorIt it = it_begin_; it != it_end_; ++it) {
                TPoint &p = *it;
                for (int ix = 0; ix < indent; ix++) {
                    os << "  ";
                }
                for (int ix = 0; ix < D; ix++) {
                    if (ix == 0) {
                        os << "Point: ";
                    } else {
                        os << ", ";
                    }
                    os << p.x_[ix];
                }
                os << std::endl;
            }

        } else {

            for (int ix = 0; ix < indent; ix++) {
                os << "  ";
            }
            os << "Split-Dim:" << split_dim_ << ",  Spit-Value:" << split_value_ << std::endl;
            children_[0]->print(os, indent+1);
            children_[1]->print(os, indent+1);
            
        }
    }
            
            
    void find_points(const PointBase<D> &min, const PointBase<D> &max, NeighborPointVisitor<D> &visitor)
    {
        if (split_dim_ < 0) {

            // If we are at a leaf, look through the points:
            for (TVectorIt it = it_begin_; it != it_end_; ++it) {
                visitor.visit_point(**it);
            }
            
        } else {

            // Otherwise see which of the children we need to visit
            if (min.x_[split_dim_] <= split_value_) {
                children_[0]->find_points(min, max, visitor);
            }

            if (max.x_[split_dim_] >= split_value_) {
                children_[1]->find_points(min, max, visitor);
            }

        }
                
    }
};


#endif
