#ifndef __QTREE_NODE_HH
#define __QTREE_NODE_HH

#include <iostream>
#include "PointBase.hh"
#include "NeighborVisitor.hh"

class CounterIntEvent;
class QTreeParam;


// Children by index
enum TChildIndex
{
    /* 0 */ CHILD_TOP_LEFT,
    /* 1 */ CHILD_TOP_RIGHT,
    /* 2 */ CHILD_BOTTOM_LEFT,
    /* 3 */ CHILD_BOTTOM_RIGHT,
};

// Neighbors by index
enum TNeighborIndex
{
    /* 0 */ NEIGHBOR_TOP_LEFT,
    /* 1 */ NEIGHBOR_TOP,
    /* 2 */ NEIGHBOR_TOP_RIGHT,
    /* 3 */ NEIGHBOR_RIGHT,
    /* 4 */ NEIGHBOR_BOTTOM_RIGHT,
    /* 5 */ NEIGHBOR_BOTTOM,
    /* 6 */ NEIGHBOR_BOTTOM_LEFT,
    /* 7 */ NEIGHBOR_LEFT,
    NEIGHBOR_MAX,
};





class QTreeNode
{
    // Tree-wide parameters
    QTreeParam *param_;
    
    // Parent of this node (NULL for root)
    QTreeNode *parent_;

    // Which child of the parent are we?
    int parent_child_ix_;
    
    // Children of this node (NULL if no children)
    QTreeNode *children_[4];

    // Neighbors of this node (4 edge and 4 corner neighbors)
    QTreeNode *neighbor_[8];

    // Id of this node
    int node_id_;
    
    // Subdivision level of the node
    int level_;

    // Maximum depth reachable from this node
    int depth_;
    
    // top left and bottom right corners
    double x1_, y1_, x2_, y2_;

    // Points in this node
    PointBase<2> *points_innode_;

    // Number of points in this node
    int n_points_innode_;
    
    void verify_coordinates(const QTreeNode *parent, double x1, double y1, double x2, double y2) const;

    // Verify neighbor relations to all 8 neighbors
    void verify_neighbor_top() const;
    void verify_neighbor_bottom() const;
    void verify_neighbor_left() const;
    void verify_neighbor_right() const;
    void verify_neighbor_top_left() const;
    void verify_neighbor_top_right() const;
    void verify_neighbor_bottom_left() const;
    void verify_neighbor_bottom_right() const;

    void verify_parent() const;
    void verify_children() const;
    
    void set_adjacent_neighbor(QTreeNode *child, int node_neighbor, int parent_neighbor, int neighbor_child);

    // Make sure a point is inside the area of the node
    void verify_inside(double x, double y);
                       
    void visit_points(NeighborPointVisitor<2> &visitor);

public:
    QTreeNode(QTreeNode *parent, int parent_child_ix, QTreeParam *param, double x1, double y1, double x2, double y2);
    ~QTreeNode();

    int get_level() const { return level_; }

    // Signal to the parent that a new node with a specific level has been created
    void signal_level(int level);

    //@@@ Perhaps this shouldn't be here
    QTreeNode *child(int child_id) { return children_[child_id]; }
    
    // Add a new point
    void add_point(PointBase<2> *point, bool should_subdivide = true);

    // Subdivide this node into 4 children
    void subdivide();

    // Find which child quadrant a point belongs to
    int find_child_quadrant(double x, double y);
    
    // Find the leaf that a point is in
    QTreeNode *find_leaf(double x, double y);
    
    // Visit points in neigborhood
    void visit_neighbor_points(NeighborPointVisitor<2> &visitor);
        
    // Dump the tree structure to a stream
    void dump(std::ostream &os, int fmt) const;
    
    // Verify the structures (debugging tool)
    void verify() const;

    // Statistics on how many points are per leaf
    void points_per_leaf_stats(CounterIntEvent &counter);
};


#endif

