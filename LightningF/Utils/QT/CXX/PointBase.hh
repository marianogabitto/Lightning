#ifndef __POINT_BASE_HH
#define __POINT_BASE_HH


//
// Base class for points that go into trees
//
template<int D>
struct PointBase
{
    // Coordinates
    double x_[D];

    // Next point in this tree leaf
    PointBase *next_point_leaf_;
};


#endif
