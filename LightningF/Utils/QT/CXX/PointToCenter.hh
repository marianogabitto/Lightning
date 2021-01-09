#ifndef _POINT_TO_CENTER_HH
#define _POINT_TO_CENTER_HH

//
// Mapping between points and centers
//
struct PointToCenter
{
    // Index of point and center in points/centers array
    int point_ix;
    int center_ix;

    // Computed quantities
    double logNnk_;
    double rnk_;
};


#endif
