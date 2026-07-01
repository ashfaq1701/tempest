#ifndef STEP_CUH
#define STEP_CUH

#include <cstdint>

// edge_id is EMPTY_EDGE_ID for the first hop.
struct Step {
    int node;
    int64_t timestamp;
    int64_t edge_id;

    HOST DEVICE Step() : node(-1), timestamp(-1), edge_id(EMPTY_EDGE_ID) {}

    HOST DEVICE Step(const int n, const int64_t ts, const int64_t e_id = EMPTY_EDGE_ID)
        : node(n), timestamp(ts), edge_id(e_id) {}

    HOST DEVICE bool operator==(const Step& other) const {
        return node == other.node && timestamp == other.timestamp && edge_id == other.edge_id;
    }
};

#endif // STEP_CUH
