#include "Light.hh"
#include "Timer.hh"
#include "CounterIntEvent.hh"
#include <iostream>
#include <fstream>


void light_test()
{
    Light app;

    {
        std::cout << "Loading points" << std::endl;
        Timer t(true);
        std::ifstream f("../../data/points.txt");
        app.load_points(f);
        std::cout << "    Complete in " << t.elapsed_msec() << "ms" << std::endl;
    }

    {
        std::cout << "Loading centers" << std::endl;
        Timer t(true);
        std::ifstream f("../../data/centers.txt");
        app.load_centers(f);
        std::cout << "    Complete in " << t.elapsed_msec() << "ms" << std::endl;
    }

    {
        std::cout << "Building centers tree" << std::endl;
        Timer t(true);
        app.build_tree_centers();
        std::cout << "    Complete in " << t.elapsed_msec() << "ms" << std::endl;

        CounterIntEvent counter(CounterIntEvent::SCALE_LOG2);
        app.qtree()->points_per_leaf_stats(counter);
        std::cout << "    Centers-per-leaf stats: " << counter << std::endl;
    }

    {
        std::cout << "Looping over points - looking up centers" << std::endl;
        Timer t(true);
        app.points_to_centers();
        std::cout << "    Complete in " << t.elapsed_msec() << "ms" << std::endl;
    }
}


int main()
{
    light_test();
    
    return 0;
}
