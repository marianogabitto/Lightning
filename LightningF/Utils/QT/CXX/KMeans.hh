#ifndef __KMEANS_HH
#define __KMEANS_HH

#include "Point.hh"
#include "Center.hh"
#include "QTree.hh"
#include <vector>
#include <iostream>


class KMeansData
{
protected:
    std::vector<Point> points_;
    std::vector<Center> centers_;

public:
    void assign_point_center(Point *point, Center *center);
    
public:
    void load(std::istream &is);
};


class KMeansNaive : public KMeansData
{
public:
    void assign_points_to_centers();
};
    

class KMeansTree : public KMeansData
{
    QTree *qtree_;
    
public:
    QTree *qtree() { return qtree_; }
    
    void build_tree_points();
    void build_tree_centers();
    
    void assign_points_to_centers();

    void find_nearest_point();
};


#endif
