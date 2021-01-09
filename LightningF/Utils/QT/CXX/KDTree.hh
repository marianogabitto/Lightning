#ifndef __KDTREE_HH
#define __KDTREE_HH

#include "KDTreeLevel.hh"


template<int D, class TPoint>
class KDTree
{
    typedef std::vector<TPoint *> TVector;
    typedef typename TVector::iterator TVectorIt;

    // Storage for the whole data - no copies are made, only iterators point into the array
    TVector data_;

    // Root of the tree
    KDTreeLevel<D, TPoint> *root_;

public:
    KDTree(std::vector<TPoint> &data)
    {
	for (auto it = data.begin(); it != data.end(); ++it) {
	    data_.push_back(&*it);
	}
	
        root_ = new KDTreeLevel<D, TPoint>(data_.begin(), data_.end(), 0);
        root_->create();
    }

    TVector &data() { return data_; }
    
    void print(std::ostream &os, int indent)
    {
        root_->print(os, indent);
    }
    
    void find_points(const PointBase<D> &min, const PointBase<D> &max, NeighborPointVisitor<D> &visitor)
    {
        root_->find_points(min, max, visitor);
    }

};


#endif
