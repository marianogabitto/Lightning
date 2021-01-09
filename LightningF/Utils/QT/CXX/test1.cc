#include "QTree.hh"
#include "QTreeNode.hh"
#include "QTreeParam.hh"
#include <fstream>


void test1()
{
    const double x1 = 0.0;
    const double y1 = 0.0;
    const double x2 = 1.0;
    const double y2 = 1.0;
    
    QTree *tree = new QTree(x1, y1, x2, y2);

    tree->root()->subdivide();
    tree->root()->child(0)->subdivide();
    tree->root()->child(1)->subdivide();
    //tree->root()->child(2)->subdivide();
    //tree->root()->child(3)->subdivide();
    tree->root()->child(0)->child(3)->subdivide();
    tree->root()->child(0)->child(3)->child(3)->subdivide();
    tree->root()->child(0)->child(3)->child(3)->child(3)->subdivide();

    tree->dump(std::cout);

    std::ofstream f("test1-tree.txt");
    tree->dump(f, QTREE_DUMP_STRUCTURE);
    
    tree->root()->verify();
    
    delete tree;
    
}


int main()
{
    test1();
    
    return 0;
}

