#include "KDTree.hh"
#include <fstream>
#include <stdlib.h>
#include <iostream>

struct MyPoint : public PointBase<2>
{
    int mark_;
};


MyPoint make_point2d(double x1, double x2)
{
    MyPoint p;
    p.x_[0] = x1;
    p.x_[1] = x2;
    p.mark_ = 0;
    return p;
}


class MyVisitor : public NeighborPointVisitor<2>
{
    virtual void visit_point(PointBase<2> &point)
    {
        MyPoint &p = (MyPoint &)point;
        std::cout << "    Visit: " << point.x_[0] << ", " << point.x_[1] << ", " << &p << std::endl;
        p.mark_ = 1;
    }
};


void test1()
{
    std::cout << "Points:" << std::endl;
    
    std::vector<MyPoint> D;
    for (int ix = 0; ix < 2000; ix++) {
        MyPoint p = make_point2d(100 * drand48(), 100 * drand48());
        std::cout << "    " << p.x_[0] << ", " << p.x_[1] << std::endl;
        D.push_back(p);
    }

    KDTree<2, MyPoint> tree(D);

    std::cout << "Tree:" << std::endl;
    tree.print(std::cout, 1);
    
    PointBase<2> min = make_point2d(20, 20);
    PointBase<2> max = make_point2d(40, 40);
    MyVisitor visitor;
    tree.find_points(min, max, visitor);

    // Verify that all the relevant points have been visited
    for (auto it = tree.data().begin(); it != tree.data().end(); ++it) {
        MyPoint &p = *it;
        if (min.x_[0] <= p.x_[0] && min.x_[1] <= p.x_[1]
            && max.x_[0] >= p.x_[0] && max.x_[1] >= p.x_[1]) {
            if (!p.mark_) {
                std::cout << "ERROR: Point not visited: " << p.x_[0] << ", " << p.x_[1] << ", " << &p << std::endl;
            }
        }
    }
}


int main()
{
    test1();
    
    return 0;
}

