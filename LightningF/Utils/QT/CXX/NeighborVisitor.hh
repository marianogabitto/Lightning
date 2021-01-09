#ifndef __NEIGHBOR_VISITOR_HH
#define __NEIGHBOR_VISITOR_HH

#include "PointBase.hh"


template<int D>
class NeighborPointVisitor
{
public:
    virtual void visit_point(PointBase<D> &point) = 0;
};

#endif
