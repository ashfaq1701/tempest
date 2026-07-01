#ifndef WALK_ITERATOR_CUH
#define WALK_ITERATOR_CUH

#include <cstddef>

#include "step.cuh"
#include "../../common/const.cuh"
#include "../../common/macros.cuh"

class WalkIterator {
private:
    const int* nodes_;
    const int64_t* timestamps_;
    const int64_t* edge_ids_; // per-walk slice, length = max(0, walk_len - 1)
    size_t index_;
    size_t length_;

public:
    HOST WalkIterator(
        const int* nodes,
        const int64_t* timestamps,
        const int64_t* edge_ids,
        const size_t index,
        const size_t length)
        : nodes_(nodes), timestamps_(timestamps), edge_ids_(edge_ids), index_(index), length_(length) {}

    HOST DEVICE Step operator*() const {
        const int64_t incoming_edge_id = (index_ == 0) ? EMPTY_EDGE_ID : edge_ids_[index_ - 1];
        return {nodes_[index_], timestamps_[index_], incoming_edge_id};
    }

    HOST DEVICE WalkIterator& operator++() {
        ++index_;
        return *this;
    }

    HOST DEVICE WalkIterator operator++(int) {
        const WalkIterator temp = *this;
        ++index_;
        return temp;
    }

    HOST DEVICE bool operator==(const WalkIterator& other) const {
        return index_ == other.index_;
    }

    HOST DEVICE bool operator!=(const WalkIterator& other) const {
        return index_ != other.index_;
    }
};

#endif // WALK_ITERATOR_CUH
