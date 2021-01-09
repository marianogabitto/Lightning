#include "QTree.hh"

#include "QTreeParam.hh"
#include "QTreeNode.hh"


QTree::QTree(double x1, double y1, double x2, double y2, double min_cell_size)
{
    param_ = new QTreeParam(x1, y1, x2, y2, min_cell_size);
    root_ = new QTreeNode(NULL /*parent*/, -1 /*parent_child_ix*/, param_, x1, y1, x2, y2);
}


QTree::~QTree()
{
    delete root_;
    delete param_;
}


void QTree::dump(std::ostream &os, int fmt) const
{
    root_->dump(os, fmt);
}


void QTree::visit_neighbor_points(double x, double y, NeighborPointVisitor<2> &visitor)
{
    QTreeNode *node = find_leaf(x, y);
    node->visit_neighbor_points(visitor);
}

