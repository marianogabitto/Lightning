#include "Timer.hh"
#include <time.h>
#include <iostream>


/*static*/
int64_t Timer::get_time_ns()
{
    timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    int64_t t = (int64_t)tp.tv_sec*1000000000l + (int64_t)tp.tv_nsec;
    //std::cout << "Time: " << t << std::endl;
    return t;
}


Timer::Timer(bool should_start)
    : total_time_(0), start_time_(0)
{
    if (should_start) {
        start();
    }
}


void Timer::clear()
{
    total_time_ = 0;
    start_time_ = 0;
}


void Timer::start()
{
    if (start_time_ == 0) {
        start_time_ = get_time_ns();
    }
}


void Timer::stop()
{
    int64_t t = get_time_ns();
    total_time_ += (t - start_time_);
    start_time_ = -1;
}


void Timer::update()
{
    if (is_started()) {
        int64_t t = get_time_ns();
        total_time_ += (t - start_time_);
        start_time_ = t;
    }
}
