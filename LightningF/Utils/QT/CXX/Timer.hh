#ifndef _TIMER_HH
#define _TIMER_HH

#include <stdint.h>


class Timer
{
    int64_t total_time_;
    int64_t start_time_;

public:
    Timer(bool should_start = false);

    // Return the time as an integer number of nanoseconds since epoch
    static int64_t get_time_ns();

    // Clear the timer (stopped with zero time)
    void clear();
    
    // Start the timer
    void start();

    // Stop the timer
    void stop();

    // Update the timer, but don't stop it
    void update();
    
    // Is the timer running
    bool is_started() const { return start_time_ > 0;}
    
    int64_t elapsed_nsec() { update(); return total_time_; }
    double elapsed_usec() { update(); return total_time_ * 1.0e-03; }
    double elapsed_msec() { update(); return total_time_ * 1.0e-06; }
    double elapsed_sec() { update(); return total_time_ * 1.0e-09; }
};


#endif
