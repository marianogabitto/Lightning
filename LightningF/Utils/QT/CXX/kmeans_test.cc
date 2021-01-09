#include "KMeans.hh"
#include "Timer.hh"
#include "CounterIntEvent.hh"
#include <fstream>


void kmeans_test_tree()
{
    KMeansTree km;

    std::cout << "Loading data from file" << std::endl;
    std::ifstream f("input2.txt");

    {
        Timer t(true);
        km.load(f);
        std::cout << "    Complete in " << t.elapsed_msec() << "ms" << std::endl;
    }
    
    std::cout << "Building tree of centers" << std::endl;
    {
        Timer t(true);
        km.build_tree_centers();
        std::cout << "    Complete in " << t.elapsed_msec() << "ms" << std::endl;
    }
    
    std::cout << "Assigning points to closest center" << std::endl;
    {
        Timer t(true);
        km.assign_points_to_centers();
        std::cout << "    Complete in " << t.elapsed_msec() << "ms" << std::endl;
    }
}


void kmeans_test_naive()
{
    KMeansNaive km;

    std::cout << "Loading data from file" << std::endl;
    std::ifstream f("input2.txt");

    {
        Timer t(true);
        km.load(f);
        std::cout << "    Complete in " << t.elapsed_msec() << "ms" << std::endl;
    }
    
    std::cout << "Assigning points to closest center" << std::endl;
    {
        Timer t(true);
        km.assign_points_to_centers();
        std::cout << "    Complete in " << t.elapsed_msec() << "ms" << std::endl;
    }
}



void nearest_point_tree()
{
    KMeansTree km;

    std::cout << "Loading data from file" << std::endl;
    std::ifstream f("input-large.txt");

    {
        Timer t(true);
        km.load(f);
        std::cout << "    Complete in " << t.elapsed_msec() << "ms" << std::endl;
    }

    std::cout << "Building tree of points" << std::endl;
    {
        Timer t(true);
        km.build_tree_points();
        std::cout << "    Complete in " << t.elapsed_msec() << "ms" << std::endl;
    }

    CounterIntEvent counter(CounterIntEvent::SCALE_LOG2);
    km.qtree()->points_per_leaf_stats(counter);
    std::cout << "Points-per-leaf stats: " << counter << std::endl;
    
    std::cout << "Find closest point to each point" << std::endl;
    {
        Timer t(true);
        km.find_nearest_point();
        std::cout << "    Complete in " << t.elapsed_msec() << "ms" << std::endl;
    }
}


int main()
{
    //kmeans_test_naive();
    //kmeans_test_tree();
    nearest_point_tree();
    
    return 0;
}
