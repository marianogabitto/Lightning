#include "KMeans.hh"
#include <limits>
#include <cmath>


void KMeansData::load(std::istream &is)
{
    int nk, np;

    is >> nk >> np;

    centers_.resize(nk);
    for (int ix = 0; ix < nk; ix++) {
        Center *c = &centers_[ix];
        is >> c->x_ >> c->y_;
        c->next_point_leaf_ = NULL;
        c->points_ = NULL;
    }

    points_.resize(np);
    for (int ix = 0; ix < np; ix++) {
        Point *p = &points_[ix];
        is >> p->x_ >> p->y_;
        p->center_ = NULL;
        p->next_point_center_ = NULL;
        p->next_point_leaf_ = NULL;
    }
}


void KMeansData::assign_point_center(Point *point, Center *center)
{
    // Mark this point's center
    point->center_ = center;

    // Add point to center's linked list of points
    point->next_point_center_ = center->points_;
    center->points_ = point;
}


void KMeansNaive::assign_points_to_centers()
{
    for (size_t ixp = 0; ixp < points_.size(); ixp++) {

        Point *point = &points_[ixp];

        Center *min_center = NULL;
        double min_d2 = std::numeric_limits<double>::infinity();
        
        for (size_t ixc = 0; ixc < centers_.size(); ixc++) {

            Center *center = &centers_[ixc];

            double dx = center->x_ - point->x_;
            double dy = center->y_ - point->y_;
            //double d2 = dx*dx + dy*dy;
            double d2 = std::exp((dx*dx + dy*dy) / 50);
            if (d2 < min_d2) {
                min_center = center;
                min_d2 = d2;
            }

        }

        if (min_center != NULL) {
            assign_point_center(point, min_center);
        }
    }
}


void KMeansTree::build_tree_points()
{
    //@@@ Do not hard code
    qtree_ = new QTree(0, 0, 30000, 30000);

    for (size_t ix = 0; ix < points_.size(); ix++) {
        qtree_->add_point(&points_[ix]);
    }
}


void KMeansTree::build_tree_centers()
{
    //@@@ Do not hard code
    qtree_ = new QTree(0, 0, 1000, 1000);

    for (size_t ix = 0; ix < centers_.size(); ix++) {
        qtree_->add_point(&centers_[ix]);
    }
}



class ClosestCenterVisitor : public QTreeNeighborPointVisitor
{
    // Point we look to find center closest to
    Point *point_;

    // Closest center found so far
    Center *center_;
    
    // Distance to closest center found
    double dist2_;
    
public:
    ClosestCenterVisitor(Point *point)
        : point_(point),
          center_(NULL),
          dist2_(std::numeric_limits<double>::infinity())
    {
    }

    virtual void visit_point(PointBase *p);

    Center *closest_center() const { return center_; }
    double closest_dist2() const { return dist2_; }
};


/*virtual*/
void ClosestCenterVisitor::visit_point(PointBase *p)
{
    Center *center = (Center *)p;
    
    double dx = center->x_ - point_->x_;
    double dy = center->y_ - point_->y_;
    double d2 = dx*dx + dy*dy;
    if (d2 < dist2_) {
        center_ = center;
        dist2_ = d2;
    }
}


void KMeansTree::assign_points_to_centers()
{
    for (size_t ix = 0; ix < points_.size(); ix++) {

        Point *p = &points_[ix];
        
        ClosestCenterVisitor closest_center(p);
        qtree_->visit_neighbor_points(p->x_, p->y_, closest_center);

        p->center_ = closest_center.closest_center();
        p->center_dist2_ = closest_center.closest_dist2();
    }
    
}



class ClosestPointVisitor : public QTreeNeighborPointVisitor
{
    // Point we look to find point closest to
    Point *point_;

    // Closest point found so far
    PointBase *closest_point_;
    
    // Distance to closest point found
    double dist2_;
    
public:
    ClosestPointVisitor(Point *point)
        : point_(point),
          closest_point_(NULL),
          dist2_(std::numeric_limits<double>::infinity())
    {
    }

    virtual void visit_point(PointBase *point);

    Point *closest_point() const { return point_; }
    double closest_dist2() const { return dist2_; }
};


/*virtual*/
void ClosestPointVisitor::visit_point(PointBase *point)
{
    double dx = point->x_ - point_->x_;
    double dy = point->y_ - point_->y_;
    double d2 = dx*dx + dy*dy;
    if (d2 < dist2_) {
        closest_point_ = point;
        dist2_ = d2;
    }
}


void KMeansTree::find_nearest_point()
{
    for (size_t ix = 0; ix < points_.size(); ix++) {

        if (ix % 10000 == 0) {
            std::cout << "IX:" << ix << std::endl;
        }
        
        Point *p = &points_[ix];
        
        ClosestPointVisitor closest_point(p);
        qtree_->visit_neighbor_points(p->x_, p->y_, closest_point);

        //p->point_ = closest_point.closest_point();
        //p->point_dist2_ = closest_point.closest_dist2();
    }
    
}


