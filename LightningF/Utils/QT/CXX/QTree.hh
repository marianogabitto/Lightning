#ifndef __QTREE_HH
#define __QTREE_HH

#include <iostream>
#include "QTreeNode.hh"
#include "PointBase.hh"


class QTree
{
    QTreeParam *param_;

    // Root node of the tree
    QTreeNode *root_;
    
public:
    QTree(double x1, double y1, double x2, double y2, double min_cell_size);
    ~QTree();

    QTreeNode *root() { return root_; }

    // Dump the tree structure to a stream
    void dump(std::ostream &os, int fmt = 0) const;

    // Find the leaf node that a point belongs to
    QTreeNode *find_leaf(double x, double y)
    { return root_->find_leaf(x, y); }

    // Add a new point
    void add_point(PointBase<2> *point, bool should_subdivide = true)
    { root_->add_point(point, should_subdivide); }
    
    // Visit neighboring points
    void visit_neighbor_points(double x, double y, NeighborPointVisitor<2> &visitor);

    // Statistics on how many points are per leaf
    void points_per_leaf_stats(CounterIntEvent &counter)
    { root_->points_per_leaf_stats(counter); }
};


#endif
