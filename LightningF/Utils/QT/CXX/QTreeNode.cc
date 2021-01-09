#include "QTreeNode.hh"

#include "CounterIntEvent.hh"
#include "QTreeParam.hh"
#include "PointBase.hh"
#include <cmath>


static const int neighbor_inverse[8] = {
    /* 0: NEIGHBOR_TOP_LEFT */       NEIGHBOR_BOTTOM_RIGHT,
    /* 1: NEIGHBOR_TOP */            NEIGHBOR_BOTTOM,
    /* 2: NEIGHBOR_TOP_RIGHT */      NEIGHBOR_BOTTOM_LEFT,
    /* 3: NEIGHBOR_RIGHT */          NEIGHBOR_LEFT,
    /* 4: NEIGHBOR_BOTTOM_RIGHT */   NEIGHBOR_TOP_LEFT,
    /* 5: NEIGHBOR_BOTTOM */         NEIGHBOR_TOP,
    /* 6: NEIGHBOR_BOTTOM_LEFT */    NEIGHBOR_TOP_RIGHT,
    /* 7: NEIGHBOR_LEFT */           NEIGHBOR_RIGHT,
};


QTreeNode::QTreeNode(QTreeNode *parent, int parent_child_ix, QTreeParam *param, double x1, double y1, double x2, double y2)
    : param_(param),
      parent_(parent),
      parent_child_ix_(parent_child_ix),
      children_{NULL, NULL, NULL, NULL},
      neighbor_{NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
      node_id_(param->new_node_id()),
      level_(0), depth_(0),
      x1_(x1), y1_(y1), x2_(x2), y2_(y2),
      points_innode_(NULL), n_points_innode_(0)
{
    if (parent_ != NULL) {
        level_ = parent_->get_level() + 1;
        depth_ = level_;
        parent_->signal_level(level_);
    }
}


QTreeNode::~QTreeNode()
{
    // Delete the children if there are any
    for (int i = 0; i < 4; i++) {
        delete children_[i];
    }
}


void QTreeNode::verify_inside(double x, double y)
{
    double eps = param_->eps();
    if (x < x1_ - eps || x > x2_ + eps
        || y < y1_ - eps || y > y2_ + eps) {

        std::cout << "ERROR: point(" << x << ", " << y
                  << "), is outside node [ (" << x1_ << ", " << y1_ << "), " << x2_ << ", " << y2_ << ") ]"
                  << std::endl;
    }
}


void QTreeNode::signal_level(int level)
{
    if (level > depth_) {
        depth_ = level;
    }
    if (parent_ != NULL) {
        parent_->signal_level(level);
    }
}


void QTreeNode::add_point(PointBase<2> *point, bool should_subdivide)
{
    // Make sure the point is in this node
    verify_inside(point->x_[0], point->x_[1]);

    if (children_[0] != NULL) {
        
        // If the node is subdivided, add to one of the children
        int child_quadrant = find_child_quadrant(point->x_[0], point->x_[1]);
        children_[child_quadrant]->add_point(point, should_subdivide);

    } else {
    
        // See if we have too many points, so we need to subdivide
        if (should_subdivide && (n_points_innode_ > param_->max_points_subdiv())
            && level_ < param_->max_subdiv_level() && 2*param_->min_cell_size() <= x2_-x1_ && 2*param_->min_cell_size() <= y2_-y1_) {
            subdivide();
            
            int child_quadrant = find_child_quadrant(point->x_[0], point->x_[1]);
            children_[child_quadrant]->add_point(point, should_subdivide);

        } else {
            // Add to the linked list
            point->next_point_leaf_ = points_innode_;
            points_innode_ = point;
            
            n_points_innode_++;
        }
    }
}


void QTreeNode::set_adjacent_neighbor(QTreeNode *child, int node_neighbor, int parent_neighbor, int neighbor_child)
{
    // Parent neighbor of parent node
    QTreeNode *parent_neighbor_node = neighbor_[parent_neighbor];

    if (parent_neighbor_node == NULL) {

        // We are at the edge
        child->neighbor_[node_neighbor] = NULL;

    } else {

        // We are not on the edge - look at the parent's neighbor
        QTreeNode *neighbor_child_node = parent_neighbor_node->children_[neighbor_child];

#if 0
        if (node_id_ == 2) {
            int neighbor_child_node_id = (neighbor_child_node == NULL ? -1 : neighbor_child_node->node_id_);
            int child_neighbor = neighbor_inverse[node_neighbor];
            std::cout << "Node-2-child: child:" << child->node_id_
                      << ", node_neighbor:" << node_neighbor
                      << ", parent_neighbor:" << parent_neighbor
                      << ", neighbor_child:" << neighbor_child
                      << ", neighbor_child_node:" << neighbor_child_node_id
                      << ", child_neighbor:" << child_neighbor
                      << std::endl;
        }
#endif
        
        if (neighbor_child_node == NULL) {

            // The neighbor has not been subdivided, so the neighbor is the courser parent
            child->neighbor_[node_neighbor] =  parent_neighbor_node;
            
        } else {

            // Set this node's neighbor
            child->neighbor_[node_neighbor] = neighbor_child_node;
            
            // Make the reverse neighbor connection of the neighbor's child:
            int child_neighbor = neighbor_inverse[node_neighbor];
            neighbor_child_node->neighbor_[child_neighbor] = child;
        }
    }
}


void QTreeNode::subdivide()
{
    // If it has already been subdivided ...
    if (children_[0] != NULL) {
        return;
    }
    
    double xm = (x1_ + x2_) / 2;
    double ym = (y1_ + y2_) / 2;

    //std::cout << "Subdividing node(" << node_id_ << ")" << std::endl;

    // Ensure that any courser neighbors are subdivided to our level first
    // This makes the new children satisfy the condition that
    // neighbors can't be more than one refinement level apart
    for (int ix = 0; ix < 8; ix++) {
        QTreeNode *neighbor = neighbor_[ix];
        if (neighbor != NULL) {
            if (neighbor->level_ < level_) {
                neighbor->subdivide();
            }
        }
    }

    // There is another case of courser neighbors:
    // when our parent has coarser neighbors that aren't subdivided
    // (since their neighbor will be this cell)
    if (parent_ != NULL) {
        for (int ix = 0; ix < 8; ix++) {
            QTreeNode *neighbor = parent_->neighbor_[ix];
            if (neighbor != NULL) {
                if (neighbor->children_[0] == NULL) {
                    neighbor->subdivide();
                }
            }
        }
    }
    
    // Create the children
    children_[CHILD_TOP_LEFT]     = new QTreeNode(this /*parent*/, CHILD_TOP_LEFT,     param_, x1_, ym , xm , y2_);
    children_[CHILD_TOP_RIGHT]    = new QTreeNode(this /*parent*/, CHILD_TOP_RIGHT,    param_, xm , ym , x2_, y2_);
    children_[CHILD_BOTTOM_LEFT]  = new QTreeNode(this /*parent*/, CHILD_BOTTOM_LEFT,  param_, x1_, y1_, xm , ym );
    children_[CHILD_BOTTOM_RIGHT] = new QTreeNode(this /*parent*/, CHILD_BOTTOM_RIGHT, param_, xm , y1_, x2_, ym );

    // Neighbors of CHILD_TOP_LEFT
    set_adjacent_neighbor(children_[CHILD_TOP_LEFT], NEIGHBOR_TOP_LEFT,     NEIGHBOR_TOP_LEFT,   CHILD_BOTTOM_RIGHT);
    set_adjacent_neighbor(children_[CHILD_TOP_LEFT], NEIGHBOR_TOP,          NEIGHBOR_TOP,        CHILD_BOTTOM_LEFT);
    set_adjacent_neighbor(children_[CHILD_TOP_LEFT], NEIGHBOR_TOP_RIGHT,    NEIGHBOR_TOP,        CHILD_BOTTOM_RIGHT);
    children_[CHILD_TOP_LEFT]->neighbor_[NEIGHBOR_RIGHT]        = children_[CHILD_TOP_RIGHT];
    children_[CHILD_TOP_LEFT]->neighbor_[NEIGHBOR_BOTTOM_RIGHT] = children_[CHILD_BOTTOM_RIGHT];
    children_[CHILD_TOP_LEFT]->neighbor_[NEIGHBOR_BOTTOM]       = children_[CHILD_BOTTOM_LEFT];
    set_adjacent_neighbor(children_[CHILD_TOP_LEFT], NEIGHBOR_BOTTOM_LEFT,  NEIGHBOR_LEFT,       CHILD_BOTTOM_RIGHT);
    set_adjacent_neighbor(children_[CHILD_TOP_LEFT], NEIGHBOR_LEFT,         NEIGHBOR_LEFT,       CHILD_TOP_RIGHT);

    // Neighbor of CHILD_TOP_RIGHT
    set_adjacent_neighbor(children_[CHILD_TOP_RIGHT], NEIGHBOR_TOP_LEFT,     NEIGHBOR_TOP,        CHILD_BOTTOM_LEFT);
    set_adjacent_neighbor(children_[CHILD_TOP_RIGHT], NEIGHBOR_TOP,          NEIGHBOR_TOP,        CHILD_BOTTOM_RIGHT);
    set_adjacent_neighbor(children_[CHILD_TOP_RIGHT], NEIGHBOR_TOP_RIGHT,    NEIGHBOR_TOP_RIGHT,  CHILD_BOTTOM_LEFT);
    set_adjacent_neighbor(children_[CHILD_TOP_RIGHT], NEIGHBOR_RIGHT,        NEIGHBOR_RIGHT,      CHILD_TOP_LEFT);
    set_adjacent_neighbor(children_[CHILD_TOP_RIGHT], NEIGHBOR_BOTTOM_RIGHT, NEIGHBOR_RIGHT,      CHILD_BOTTOM_LEFT);
    children_[CHILD_TOP_RIGHT]->neighbor_[NEIGHBOR_BOTTOM]      = children_[CHILD_BOTTOM_RIGHT];
    children_[CHILD_TOP_RIGHT]->neighbor_[NEIGHBOR_BOTTOM_LEFT] = children_[CHILD_BOTTOM_LEFT];
    children_[CHILD_TOP_RIGHT]->neighbor_[NEIGHBOR_LEFT]        = children_[CHILD_TOP_LEFT];
    
    // Neighbor of CHILD_BOTTOM_RIGHT
    children_[CHILD_BOTTOM_RIGHT]->neighbor_[NEIGHBOR_TOP_LEFT] = children_[CHILD_TOP_LEFT];
    children_[CHILD_BOTTOM_RIGHT]->neighbor_[NEIGHBOR_TOP]      = children_[CHILD_TOP_RIGHT];
    set_adjacent_neighbor(children_[CHILD_BOTTOM_RIGHT], NEIGHBOR_TOP_RIGHT,    NEIGHBOR_RIGHT,        CHILD_TOP_LEFT);
    set_adjacent_neighbor(children_[CHILD_BOTTOM_RIGHT], NEIGHBOR_RIGHT,        NEIGHBOR_RIGHT,        CHILD_BOTTOM_LEFT);
    set_adjacent_neighbor(children_[CHILD_BOTTOM_RIGHT], NEIGHBOR_BOTTOM_RIGHT, NEIGHBOR_BOTTOM_RIGHT, CHILD_TOP_LEFT);
    set_adjacent_neighbor(children_[CHILD_BOTTOM_RIGHT], NEIGHBOR_BOTTOM,       NEIGHBOR_BOTTOM,       CHILD_TOP_RIGHT);
    set_adjacent_neighbor(children_[CHILD_BOTTOM_RIGHT], NEIGHBOR_BOTTOM_LEFT,  NEIGHBOR_BOTTOM,       CHILD_TOP_LEFT);
    children_[CHILD_BOTTOM_RIGHT]->neighbor_[NEIGHBOR_LEFT]        = children_[CHILD_BOTTOM_LEFT];
    
    // Neighbor of CHILD_BOTTOM_LEFT
    set_adjacent_neighbor(children_[CHILD_BOTTOM_LEFT], NEIGHBOR_TOP_LEFT,    NEIGHBOR_LEFT,        CHILD_TOP_RIGHT);
    children_[CHILD_BOTTOM_LEFT]->neighbor_[NEIGHBOR_TOP]       = children_[CHILD_TOP_LEFT];
    children_[CHILD_BOTTOM_LEFT]->neighbor_[NEIGHBOR_TOP_RIGHT] = children_[CHILD_TOP_RIGHT];
    children_[CHILD_BOTTOM_LEFT]->neighbor_[NEIGHBOR_RIGHT]     = children_[CHILD_BOTTOM_RIGHT];
    set_adjacent_neighbor(children_[CHILD_BOTTOM_LEFT], NEIGHBOR_BOTTOM_RIGHT, NEIGHBOR_BOTTOM,       CHILD_TOP_RIGHT);
    set_adjacent_neighbor(children_[CHILD_BOTTOM_LEFT], NEIGHBOR_BOTTOM,       NEIGHBOR_BOTTOM,       CHILD_TOP_LEFT);
    set_adjacent_neighbor(children_[CHILD_BOTTOM_LEFT], NEIGHBOR_BOTTOM_LEFT,  NEIGHBOR_BOTTOM_LEFT,  CHILD_TOP_RIGHT);
    set_adjacent_neighbor(children_[CHILD_BOTTOM_LEFT], NEIGHBOR_LEFT,         NEIGHBOR_LEFT,         CHILD_BOTTOM_RIGHT);

    // Move the points to the child nodes
    while (points_innode_ != NULL) {
        PointBase<2> *point = points_innode_;
        points_innode_ = point->next_point_leaf_;

        // Add this point to one of the child nodes that were just created
        int child_quadrant = find_child_quadrant(point->x_[0], point->x_[1]);
        children_[child_quadrant]->add_point(point);
    }
    n_points_innode_ = 0;
}


int QTreeNode::find_child_quadrant(double x, double y)
{
    double xm = (x1_ + x2_) / 2;
    double ym = (y1_ + y2_) / 2;
    if (x <= xm) {
        if (y < ym) {
            return CHILD_BOTTOM_LEFT;
        } else {
            return CHILD_TOP_LEFT;
        }
    } else {
        if (y < ym) {
            return CHILD_BOTTOM_RIGHT;
        } else {
            return CHILD_TOP_RIGHT;
        }
    }
}


QTreeNode *QTreeNode::find_leaf(double x, double y)
{
    if (children_[0] == NULL) {
        return this;
    }

    int child_quadrant = find_child_quadrant(x, y);
    return children_[child_quadrant]->find_leaf(x, y);
}


void QTreeNode::visit_points(NeighborPointVisitor<2> &visitor)
{
    PointBase<2> *point = points_innode_;
    while (point != NULL) {
        visitor.visit_point(*point);
        point = point->next_point_leaf_;
    }
}


void QTreeNode::visit_neighbor_points(NeighborPointVisitor<2> &visitor)
{
    // Visit the points in this leaf
    visit_points(visitor);

    // Need to make sure we don't double visit neighbors
    // (when a finer leaf visits coarser neighbors)
    QTreeNode *prev_neighbor = neighbor_[NEIGHBOR_MAX-1];
    
    // Go through all the neighbors of the leaf
    for (int ix = 0; ix < NEIGHBOR_MAX; ix++) {

        // Visit points in neighbor node
        QTreeNode *neighbor = neighbor_[ix];
        if (neighbor != NULL and neighbor != prev_neighbor) {
            neighbor->visit_points(visitor);
        }

        prev_neighbor = neighbor;
    }
}



void QTreeNode::dump(std::ostream &os, int fmt) const
{
    int parent_id = -1;
    if (parent_) {
        parent_id = parent_->node_id_;
    }

    switch (fmt) {
    case QTREE_DUMP_READABLE:
        for (int ix = 0; ix < level_; ix++) {
            os << "    ";
        }
    
        os << "QTreeNode(" << node_id_
           << ", [" << x1_ << ", " << y1_
           << "], [" << x2_ << ", " << y2_
           << "]), parent:" << parent_id
           << ", level:" << level_
           << ", depth:" << depth_
           << ", neighbor:[";
        
        for (int ix = 0; ix < 8; ix++) {
            if (ix > 0) {
                os << ", ";
            }
            if (neighbor_[ix] == NULL) {
                os << -1;
        } else {
                os << neighbor_[ix]->node_id_;
            }
        }
        os << "]" << std::endl;
        break;

    case QTREE_DUMP_STRUCTURE:

        os << node_id_
           << " " << parent_id
           << " " << level_
           << " " << depth_
           << "    " << x1_
           << " " << y1_
           << " " << x2_
           << " " << y2_
           << std::endl;
        break;

    default:
        os << "QTreeNode::dump: UNKNOWN FORMAT" << std::endl;
        break;
    }
    
    for (int i = 0; i < 4; i++) {
        if (children_[i] != NULL) {
            children_[i]->dump(os, fmt);
        }
    }
}


bool equals(double v1, double v2, double eps = 1e-12)
{
    return fabs(v1 - v2) < eps * (fabs(v1) + fabs(v2) + 1.0);
}


void QTreeNode::verify_coordinates(const QTreeNode *parent, double x1, double y1, double x2, double y2) const
{
    if (parent_ != parent) {
        std::cout << "ERROR: child parent is not self" << std::endl;
    }
    
    if (!equals(x1_, x1) || !equals(x2_, x2)
        || !equals(y1_, y1) || !equals(y2_, y2) ) {
        std::cout << "ERROR: coordinates mismatch for node("
                  << node_id_ << ")"
                  << std::endl;
    }

}


void QTreeNode::verify_neighbor_top() const
{
    // NEIGHBOR_TOP:
    const QTreeNode *neighbor = neighbor_[NEIGHBOR_TOP];
    if (neighbor == NULL) {

        // No neighbor - we must be on the top boundary
        if (!equals(y2_, param_->y2())) {
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP is NULL, but not on the top edge"
                      << " [y2:" << y2_ << ", param->y2:" << param_->y2() << "]"
                      << std::endl;
        }
        
    } else {

        // We have a neighbor
        // Neighbor needs to be exactly above
        if (!equals(y2_, neighbor->y1_)) {
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP is not adjacent above"
                      << " [y2:" << y2_ << ", neighbor->y1:" << neighbor->y1_ << "]"
                      << std::endl;
        }
        
        // Two cases: same level, or coarser level
        if (level_ == neighbor->level_) {
            // Same level - x coordinates must match
            if (!equals(x1_, neighbor->x1_) || !equals(x2_, neighbor->x2_)) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP same level - x-mismatch"
                          << " [x1:" << x1_ << ", neighbor->x1:" << neighbor->x1_ << ", "
                          << " [x2:" << x2_ << ", neighbor->x2:" << neighbor->x2_ << "]"
                          << std::endl;
            }
            // Same level neighbor and we are a leaf - should have depth at most one more
            if (children_[0] == NULL && depth_ != neighbor->depth_ && depth_ != neighbor->depth_-1) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP same level - depth mismatch on neighbor"
                          << " [depth:" << depth_ << ", neighbor->depth:" << neighbor->depth_ << "]"
                          << std::endl;
            }
        } else if (level_ == neighbor->level_ + 1) {
            // Coarser level - one of the x coordinates must match
            if (!equals(x1_, neighbor->x1_) && !equals(x2_, neighbor->x2_)) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP coarser level - x-mismatch"
                          << " [x1:" << x1_ << ", neighbor->x1:" << neighbor->x1_ << ", "
                          << " [x2:" << x2_ << ", neighbor->x2:" << neighbor->x2_ << "]"
                          << std::endl;
            }
            // The coarser neighbor should not have children
            if (neighbor->children_[0] != NULL) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP coarser level neighbor has children" << std::endl;
            }
        } else {
            // This shouldn't happen - incorrect neighbor subdivision
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP level mismatch"
                      << " [level:" << level_ << ", neighbor->level:" << neighbor->level_ << "]"
                      << std::endl;
        }
            
    }
}


void QTreeNode::verify_neighbor_bottom() const
{
    // NEIGHBOR_BOTTOM:
    const QTreeNode *neighbor = neighbor_[NEIGHBOR_BOTTOM];
    if (neighbor == NULL) {

        // No neighbor - we must be on the bottom boundary
        if (!equals(y1_, param_->y1())) {
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM is NULL, but not on the bottom edge"
                      << " [y1:" << y1_ << ", param->y1:" << param_->y1() << "]"
                      << std::endl;
        }
        
    } else {

        // We have a neighbor
        // Neighbor needs to be exactly below
        if (!equals(y1_, neighbor->y2_)) {
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM is not adjacent below"
                      << " [y1:" << y1_ << ", neighbor->y2:" << neighbor->y2_ << "]"
                      << std::endl;
        }
        
        // Two cases: same level, or coarser level
        if (level_ == neighbor->level_) {
            // Same level - x coordinates must match
            if (!equals(x1_, neighbor->x1_) || !equals(x2_, neighbor->x2_)) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM same level - x-mismatch"
                          << " [x1:" << x1_ << ", neighbor->x1:" << neighbor->x1_ << ", "
                          << " [x2:" << x2_ << ", neighbor->x2:" << neighbor->x2_ << "]"
                          << std::endl;
            }
            // Same level neighbor and we are a leaf - should have depth at most one more
            if (children_[0] == NULL && depth_ != neighbor->depth_ && depth_ != neighbor->depth_-1) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM same level - depth mismatch on neighbor"
                          << " [depth:" << depth_ << ", neighbor->depth:" << neighbor->depth_ << "]"
                          << std::endl;
            }
        } else if (level_ == neighbor->level_ + 1) {
            // Coarser level - one of the x coordinates must match
            if (!equals(x1_, neighbor->x1_) && !equals(x2_, neighbor->x2_)) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM coarser level - x-mismatch"
                          << " [x1:" << x1_ << ", neighbor->x1:" << neighbor->x1_ << ", "
                          << " [x2:" << x2_ << ", neighbor->x2:" << neighbor->x2_ << "]"
                          << std::endl;
            }
            // The coarser neighbor should not have children
            if (neighbor->children_[0] != NULL) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM coarser level neighbor has children" << std::endl;
            }
        } else {
            // This shouldn't happen - incorrect neighbor subdivision
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM level mismatch"
                      << " [level:" << level_ << ", neighbor->level:" << neighbor->level_ << "]"
                      << std::endl;
        }
            
    }
}


void QTreeNode::verify_neighbor_left() const
{
    // NEIGHBOR_LEFT:
    const QTreeNode *neighbor = neighbor_[NEIGHBOR_LEFT];
    if (neighbor == NULL) {

        // No neighbor - we must be on the left boundary
        if (!equals(x1_, param_->x1())) {
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_LEFT is NULL, but not on the left edge"
                      << " [x1:" << x1_ << ", param->x1:" << param_->x1() << "]"
                      << std::endl;
        }
        
    } else {

        // We have a neighbor
        // Neighbor needs to be exactly to the left
        if (!equals(x1_, neighbor->x2_)) {
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_LEFT is not adjacent to the left"
                      << " [x1:" << x1_ << ", neighbor->x2:" << neighbor->x2_ << "]"
                      << std::endl;
        }
        
        // Two cases: same level, or coarser level
        if (level_ == neighbor->level_) {
            // Same level - y coordinates must match
            if (!equals(y1_, neighbor->y1_) || !equals(y2_, neighbor->y2_)) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_LEFT same level - y-mismatch"
                          << " [y1:" << y1_ << ", neighbor->y1:" << neighbor->y1_ << ", "
                          << " [y2:" << y2_ << ", neighbor->y2:" << neighbor->y2_ << "]"
                          << std::endl;
            }
            // Same level neighbor and we are a leaf - should have depth at most one more
            if (children_[0] == NULL && depth_ != neighbor->depth_ && depth_ != neighbor->depth_-1) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_LEFT same level - depth mismatch on neighbor"
                          << " [depth:" << depth_ << ", neighbor->depth:" << neighbor->depth_ << "]"
                          << std::endl;
            }
        } else if (level_ == neighbor->level_ + 1) {
            // Coarser level - one of the y coordinates must match
            if (!equals(y1_, neighbor->y1_) && !equals(y2_, neighbor->y2_)) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_LEFT coarser level - y-mismatch"
                          << " [y1:" << y1_ << ", neighbor->y1:" << neighbor->y1_ << ", "
                          << " [y2:" << y2_ << ", neighbor->y2:" << neighbor->y2_ << "]"
                          << std::endl;
            }
            // The coarser neighbor should not have children
            if (neighbor->children_[0] != NULL) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_LEFT coarser level neighbor has children" << std::endl;
            }
        } else {
            // This shouldn't happen - incorrect neighbor subdivision
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_LEFT level mismatch"
                      << " [level:" << level_ << ", neighbor->level:" << neighbor->level_ << "]"
                      << std::endl;
        }
            
    }
}


void QTreeNode::verify_neighbor_right() const
{
    // NEIGHBOR_RIGHT:
    const QTreeNode *neighbor = neighbor_[NEIGHBOR_RIGHT];
    if (neighbor == NULL) {

        // No neighbor - we must be on the right boundary
        if (!equals(x2_, param_->x2())) {
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_RIGHT is NULL, but not on the right edge"
                      << " [x1:" << x1_ << ", param->x1:" << param_->x1() << "]"
                      << std::endl;
        }
        
    } else {

        // We have a neighbor
        // Neighbor needs to be exactly to the right
        if (!equals(x2_, neighbor->x1_)) {
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_RIGHT is not adjacent to the right"
                      << " [x2:" << x2_ << ", neighbor->x1:" << neighbor->x1_ << "]"
                      << std::endl;
        }
        
        // Two cases: same level, or coarser level
        if (level_ == neighbor->level_) {
            // Same level - y coordinates must match
            if (!equals(y1_, neighbor->y1_) || !equals(y2_, neighbor->y2_)) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_RIGHT same level - y-mismatch"
                          << " [y1:" << y1_ << ", neighbor->y1:" << neighbor->y1_ << ", "
                          << " [y2:" << y2_ << ", neighbor->y2:" << neighbor->y2_ << "]"
                          << std::endl;
            }
            // Same level neighbor and we are a leaf - should have depth at most one more
            if (children_[0] == NULL && depth_ != neighbor->depth_ && depth_ != neighbor->depth_-1) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_RIGHT same level - depth mismatch on neighbor"
                          << " [depth:" << depth_ << ", neighbor->depth:" << neighbor->depth_ << "]"
                          << std::endl;
            }
        } else if (level_ == neighbor->level_ + 1) {
            // Coarser level - one of the y coordinates must match
            if (!equals(y1_, neighbor->y1_) && !equals(y2_, neighbor->y2_)) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_RIGHT coarser level - y-mismatch"
                          << " [y1:" << y1_ << ", neighbor->y1:" << neighbor->y1_ << ", "
                          << " [y2:" << y2_ << ", neighbor->y2:" << neighbor->y2_ << "]"
                          << std::endl;
            }
            // The coarser neighbor should not have children
            if (neighbor->children_[0] != NULL) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_RIGHT coarser level neighbor has children" << std::endl;
            }
        } else {
            // This shouldn't happen - incorrect neighbor subdivision
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_RIGHT level mismatch"
                      << " [level:" << level_ << ", neighbor->level:" << neighbor->level_ << "]"
                      << std::endl;
        }
            
    }
}


void QTreeNode::verify_neighbor_top_left() const
{
    // NEIGHBOR_TOP_LEFT:
    const QTreeNode *neighbor = neighbor_[NEIGHBOR_TOP_LEFT];
    if (neighbor == NULL) {

        // No neighbor - we must be on the left or top boundary
        if (!equals(x1_, param_->x1()) && !equals(y2_, param_->y2())) {
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP_LEFT is NULL, but not on the left or top edge"
                      << " [x1:" << x1_ << ", param->x1:" << param_->x1() << ", "
                      << " [y2:" << y2_ << ", param->y2:" << param_->y2() << "]"
                      << std::endl;
        }
        
    } else {

        // Two cases: same level, or coarser level
        if (level_ == neighbor->level_) {
            
            // Same level - Neighbor needs to be exactly on the left and top
            if (!equals(x1_, neighbor->x2_) || !equals(y2_, neighbor->y1_)) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP_LEFT is not adjacent top-left"
                          << " [x1:" << x1_ << ", neighbor->x2:" << neighbor->x2_
                          << ", y2:" << y2_ << ", neighbor->y1:" << neighbor->y1_ << "]"
                          << std::endl;
            }
            // Same level neighbor and we are a leaf - should have depth at most one more
            if (children_[0] == NULL && depth_ != neighbor->depth_ && depth_ != neighbor->depth_-1) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP_LEFT same level - depth mismatch on neighbor"
                          << " [depth:" << depth_ << ", neighbor->depth:" << neighbor->depth_ << "]"
                          << std::endl;
            }
        } else if (level_ == neighbor->level_ + 1) {
            // Coarser level - more complicated case, since the top left corner can be in the middle of the coarser edge
            if ((equals(x1_, neighbor->x2_) && equals(y2_, neighbor->y1_))
                 || (equals(x1_, neighbor->x2_) && equals(y1_, neighbor->y1_))
                 || (equals(x2_, neighbor->x2_) && equals(y2_, neighbor->y1_))) {
                // GOOD
            } else {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP_LEFT coarser level - corner mismatch"
                          << " [x1:" << x1_ << ", y1:" << y1_
                          << ", x2:" << x2_ << ", y2:" << y2_
                          << " neighbor->x2:" << neighbor->x2_ << ", neighbor->y1:" << neighbor->y1_ << "]"
                          << std::endl;
            }
            // The coarser neighbor should not have children
            if (neighbor->children_[0] != NULL) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP_LEFT coarser level neighbor has children" << std::endl;
            }
        } else {
            // This shouldn't happen - incorrect neighbor subdivision
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP_LEFT level mismatch"
                      << " [level:" << level_ << ", neighbor->level:" << neighbor->level_ << "]"
                      << std::endl;
        }

    }
}


void QTreeNode::verify_neighbor_top_right() const
{
    // NEIGHBOR_TOP_RIGHT:
    const QTreeNode *neighbor = neighbor_[NEIGHBOR_TOP_RIGHT];
    if (neighbor == NULL) {

        // No neighbor - we must be on the right or top boundary
        if (!equals(x2_, param_->x2()) && !equals(y2_, param_->y2())) {
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP_RIGHT is NULL, but not on the right or top edge"
                      << " [x2:" << x2_ << ", param->x2:" << param_->x2() << ", "
                      << " [y2:" << y2_ << ", param->y2:" << param_->y2() << "]"
                      << std::endl;
        }
        
    } else {

        // Two cases: same level, or coarser level
        if (level_ == neighbor->level_) {
            
            // Same level - Neighbor needs to be exactly on the right and top
            if (!equals(x2_, neighbor->x1_) || !equals(y2_, neighbor->y1_)) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP_RIGHT is not adjacent top-right"
                          << " [x2:" << x2_ << ", neighbor->x1:" << neighbor->x1_
                          << ", y2:" << y2_ << ", neighbor->y1:" << neighbor->y1_ << "]"
                          << std::endl;
            }
            // Same level neighbor and we are a leaf - should have depth at most one more
            if (children_[0] == NULL && depth_ != neighbor->depth_ && depth_ != neighbor->depth_-1) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP_RIGHT same level - depth mismatch on neighbor"
                          << " [depth:" << depth_ << ", neighbor->depth:" << neighbor->depth_ << "]"
                          << std::endl;
            }
        } else if (level_ == neighbor->level_ + 1) {
            // Coarser level - more complicated case, since the top right corner can be in the middle of the coarser edge
            if ((equals(x2_, neighbor->x1_) && equals(y2_, neighbor->y1_))
                 || (equals(x2_, neighbor->x1_) && equals(y1_, neighbor->y1_))
                 || (equals(x1_, neighbor->x1_) && equals(y2_, neighbor->y1_))) {
                // GOOD
            } else {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP_RIGHT coarser level - corner mismatch"
                          << " [x1:" << x1_ << ", y1:" << y1_
                          << ", x2:" << x2_ << ", y2:" << y2_
                          << " neighbor->x1:" << neighbor->x1_ << ", neighbor->y1:" << neighbor->y1_ << "]"
                          << std::endl;
            }
            // The coarser neighbor should not have children
            if (neighbor->children_[0] != NULL) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP_RIGHT coarser level neighbor has children" << std::endl;
            }
        } else {
            // This shouldn't happen - incorrect neighbor subdivision
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_TOP_RIGHT level mismatch"
                      << " [level:" << level_ << ", neighbor->level:" << neighbor->level_ << "]"
                      << std::endl;
        }

    }
}


void QTreeNode::verify_neighbor_bottom_right() const
{
    // NEIGHBOR_BOTTOM_RIGHT:
    const QTreeNode *neighbor = neighbor_[NEIGHBOR_BOTTOM_RIGHT];
    if (neighbor == NULL) {

        // No neighbor - we must be on the right or bottom boundary
        if (!equals(x2_, param_->x2()) && !equals(y1_, param_->y1())) {
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM_RIGHT is NULL, but not on the right or bottom edge"
                      << " [x2:" << x2_ << ", param->x2:" << param_->x2() << ", "
                      << " [y1:" << y1_ << ", param->y1:" << param_->y1() << "]"
                      << std::endl;
        }
        
    } else {

        // Two cases: same level, or coarser level
        if (level_ == neighbor->level_) {
            
            // Same level - Neighbor needs to be exactly on the right and bottom
            if (!equals(x2_, neighbor->x1_) || !equals(y1_, neighbor->y2_)) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM_RIGHT is not adjacent bottom-right"
                          << " [x2:" << x2_ << ", neighbor->x1:" << neighbor->x1_
                          << ", y1:" << y1_ << ", neighbor->y2:" << neighbor->y2_ << "]"
                          << std::endl;
            }
            // Same level neighbor and we are a leaf - should have depth at most one more
            if (children_[0] == NULL && depth_ != neighbor->depth_ && depth_ != neighbor->depth_-1) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM_RIGHT same level - depth mismatch on neighbor"
                          << " [depth:" << depth_ << ", neighbor->depth:" << neighbor->depth_ << "]"
                          << std::endl;
            }
        } else if (level_ == neighbor->level_ + 1) {
            // Coarser level - more complicated case, since the top right corner can be in the middle of the coarser edge
            if ((equals(x2_, neighbor->x1_) && equals(y1_, neighbor->y2_))
                 || (equals(x2_, neighbor->x1_) && equals(y2_, neighbor->y2_))
                 || (equals(x1_, neighbor->x1_) && equals(y1_, neighbor->y2_))) {
                // GOOD
            } else {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM_RIGHT coarser level - corner mismatch"
                          << " [x1:" << x1_ << ", y1:" << y1_
                          << ", x2:" << x2_ << ", y2:" << y2_
                          << " neighbor->x1:" << neighbor->x1_ << ", neighbor->y2:" << neighbor->y2_ << "]"
                          << std::endl;
            }
            // The coarser neighbor should not have children
            if (neighbor->children_[0] != NULL) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM_RIGHT coarser level neighbor has children" << std::endl;
            }
        } else {
            // This shouldn't happen - incorrect neighbor subdivision
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM_RIGHT level mismatch"
                      << " [level:" << level_ << ", neighbor->level:" << neighbor->level_ << "]"
                      << std::endl;
        }

    }
}


void QTreeNode::verify_neighbor_bottom_left() const
{
    // NEIGHBOR_BOTTOM_LEFT:
    const QTreeNode *neighbor = neighbor_[NEIGHBOR_BOTTOM_LEFT];
    if (neighbor == NULL) {

        // No neighbor - we must be on the left or bottom boundary
        if (!equals(x1_, param_->x1()) && !equals(y1_, param_->y1())) {
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM_LEFT is NULL, but not on the right or bottom edge"
                      << " [x1:" << x1_ << ", param->x1:" << param_->x1() << ", "
                      << " [y1:" << y1_ << ", param->y1:" << param_->y1() << "]"
                      << std::endl;
        }
        
    } else {

        // Two cases: same level, or coarser level
        if (level_ == neighbor->level_) {
            
            // Same level - Neighbor needs to be exactly on the left and bottom
            if (!equals(x1_, neighbor->x2_) || !equals(y1_, neighbor->y2_)) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM_LEFT is not adjacent bottom-left"
                          << " [x1:" << x1_ << ", neighbor->x2:" << neighbor->x2_
                          << ", y1:" << y1_ << ", neighbor->y2:" << neighbor->y2_ << "]"
                          << std::endl;
            }
            // Same level neighbor and we are a leaf - should have depth at most one more
            if (children_[0] == NULL && depth_ != neighbor->depth_ && depth_ != neighbor->depth_-1) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM_LEFT same level - depth mismatch on neighbor"
                          << " [depth:" << depth_ << ", neighbor->depth:" << neighbor->depth_ << "]"
                          << std::endl;
            }
        } else if (level_ == neighbor->level_ + 1) {
            // Coarser level - more complicated case, since the top right corner can be in the middle of the coarser edge
            if ((equals(x1_, neighbor->x2_) && equals(y1_, neighbor->y2_))
                 || (equals(x1_, neighbor->x2_) && equals(y2_, neighbor->y2_))
                 || (equals(x2_, neighbor->x2_) && equals(y1_, neighbor->y2_))) {
                // GOOD
            } else {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM_LEFT coarser level - corner mismatch"
                          << " [x1:" << x1_ << ", y1:" << y1_
                          << ", x2:" << x2_ << ", y2:" << y2_
                          << " neighbor->x2:" << neighbor->x2_ << ", neighbor->y2:" << neighbor->y2_ << "]"
                          << std::endl;
            }
            // The coarser neighbor should not have children
            if (neighbor->children_[0] != NULL) {
                std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM_LEFT coarser level neighbor has children" << std::endl;
            }
        } else {
            // This shouldn't happen - incorrect neighbor subdivision
            std::cout << "ERROR: Node(" << node_id_ << "): NEIGHBOR_BOTTOM_LEFT level mismatch"
                      << " [level:" << level_ << ", neighbor->level:" << neighbor->level_ << "]"
                      << std::endl;
        }

    }
}


void QTreeNode::verify_parent() const
{
    // Root node
    if (node_id_ == 0) {
        if (parent_ != NULL) {
            std::cout << "ERROR: root node has parent" << std::endl;
        }
        verify_coordinates(NULL, param_->x1(), param_->y1(), param_->x2(), param_->y2());
    }

    // Verify info from parent:
    if (parent_ != NULL) {

        if (parent_->children_[parent_child_ix_] != this) {
            std::cout << "ERROR: we are not the right child ren of the parent" << std::endl;
        }
        
        if (parent_->param_ != param_) {
            std::cout << "ERROR: param pointer mismatch" << std::endl;
        }

        if (parent_->level_ != level_-1) {
            std::cout << "ERROR: level incorrect" << std::endl;
        }
        if (parent_->depth_ < depth_) {
            std::cout << "ERROR: Node(" << node_id_ << "): parent depth incorrect" << std::endl;
        }
    }

}


void QTreeNode::verify_children() const
{
    // No children
    if (children_[0] == NULL && children_[1] == NULL && children_[2] == NULL && children_[3] == NULL) {
        // No children - ok
        return;
    }

    // Some children exist
    if (children_[0] == NULL || children_[1] == NULL || children_[2] == NULL || children_[3] == NULL) {
        std::cout << "ERROR: Only some children exist" << std::endl;
        return;
    }

    // All children exist - verify coordinates & parent
    double xm = (x1_ + x2_) / 2;
    double ym = (y1_ + y2_) / 2;
    children_[CHILD_TOP_LEFT]    ->verify_coordinates(this, x1_, ym , xm , y2_);
    children_[CHILD_TOP_RIGHT]   ->verify_coordinates(this, xm , ym , x2_, y2_);
    children_[CHILD_BOTTOM_LEFT] ->verify_coordinates(this, x1_, y1_, xm , ym );
    children_[CHILD_BOTTOM_RIGHT]->verify_coordinates(this, xm , y1_, x2_, ym );

    // Verify children
    for (int ix = 0; ix < 4; ix++) {
        children_[ix]->verify();
    }
}


void QTreeNode::verify() const
{
    // Make sure the parent relationship is ok
    verify_parent();
    
    // Verify neighbors
    verify_neighbor_top();
    verify_neighbor_bottom();
    verify_neighbor_left();
    verify_neighbor_right();
    verify_neighbor_top_left();
    verify_neighbor_top_right();
    verify_neighbor_bottom_right();
    verify_neighbor_bottom_left();
    
    // Verify children
    verify_children();
}


void QTreeNode::points_per_leaf_stats(CounterIntEvent &counter)
{
    if (children_[0] != NULL) {
        for (int ix = 0; ix < 4; ix++) {
            children_[ix]->points_per_leaf_stats(counter);
        }
    } else {
        counter.add(n_points_innode_);
    }
}


