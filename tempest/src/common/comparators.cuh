#ifndef COMPARATORS_H
#define COMPARATORS_H

#include "../data/structs.cuh"
#include "macros.cuh"

struct TimestampComparator {
    int64_t* timestamps;

    HOST DEVICE explicit TimestampComparator(int64_t* ts) : timestamps(ts) {}

    HOST DEVICE bool operator()(int a, int b) const {
        return timestamps[a] < timestamps[b];
    }
};

#endif // COMPARATORS_H
