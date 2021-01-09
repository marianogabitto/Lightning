#ifndef __QTREE_PARAM__
#define __QTREE_PARAM__


enum QTreeDumpFormat
{
    QTREE_DUMP_READABLE,
    QTREE_DUMP_STRUCTURE,
};


class QTreeParam
{
    // Domain
    double x1_, y1_, x2_, y2_;

    // Next node ID
    int next_node_id_;

    // Max level to subdivide to
    int max_subdiv_level_;

    // Maximum points per node before we subdivide
    int max_points_subdiv_;

    // Minimum cell size (dx or dy), i.e. we won't subdivide unless node is 2*min_cell_size_
    double min_cell_size_;
    
    // Epsilon for the purposes of comparisons
    double eps_;
    
public:
    QTreeParam(double x1, double y1, double x2, double y2, double min_cell_size)
        : x1_(x1), y1_(y1), x2_(x2), y2_(y2),
          next_node_id_(0),
          max_subdiv_level_(20),
          max_points_subdiv_(2),
          min_cell_size_(min_cell_size),
          eps_(1e-12)
    {
    }

    double x1() const { return x1_; }
    double x2() const { return x2_; }
    double y1() const { return y1_; }
    double y2() const { return y2_; }

    int max_subdiv_level() const { return max_subdiv_level_; }
    int max_points_subdiv() const { return max_points_subdiv_; }
    double min_cell_size() const { return min_cell_size_; }
    
    double eps() const { return eps_; }
    
    // Allocate a new node ID
    int new_node_id() { return next_node_id_++; }
    
};
    

#endif
