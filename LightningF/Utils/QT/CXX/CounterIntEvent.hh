#ifndef COUNTER_INT_EVENT_HH
#define COUNTER_INT_EVENT_HH

#include <vector>
#include <iostream>


class CounterIntEvent
{
public:
    enum Scale {
        SCALE_LINEAR,
        SCALE_LOG2,
    };
    
private:
    const Scale scale_;
    const int max_index_;
    
    std::vector<long> counts_;
    
    long calc_index(long value) const
    {
        switch (scale_) {
        case SCALE_LINEAR:
            return value;
        case SCALE_LOG2: {
            long ix = 0;
            while (value > 0) {
                ix++;
                value >>= 1;
            }
            return ix;
        }
        default:
            return value;
        }
    }

    
public:
    CounterIntEvent(Scale scale, int max_index = 1000)
        : scale_(scale), max_index_(max_index)
    {}

    void add(long value)
    {
        long ix = calc_index(value);
        if (ix < 0 || ix > max_index_) {
            std::cerr << "CounterIntEvent::add() value out of range [" << value << "]" << std::endl;
            return;
        }
        while ((long)counts_.size() <= ix) {
            counts_.push_back(0);
        }
        counts_[ix]++;
    }

    Scale scale() const { return scale_; }
    size_t ncounts() const { return counts_.size(); }
    int count(int ix) const { return counts_[ix]; }
};


inline std::ostream &operator<<(std::ostream &os, const CounterIntEvent &ev)
{
    os << "[ ";
    
    switch (ev.scale()) {
    case CounterIntEvent::SCALE_LINEAR:
        for (size_t ix = 0; ix < ev.ncounts(); ix++) {
            if (ix > 0) {
                os << ", ";
            }
            os << ix << ":" << ev.count(ix);
        }
        break;
    case CounterIntEvent::SCALE_LOG2:
        long v = 1;
        for (size_t ix = 0; ix < ev.ncounts(); ix++) {
            if (ix > 0) {
                os << ", ";
            }
            os << "<" << v << ":" << ev.count(ix);
            v <<= 1;
        }
        break;
    }
        
    os << " ]";
    return os;
}



#endif
