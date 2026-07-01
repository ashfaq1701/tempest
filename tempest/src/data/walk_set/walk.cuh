#ifndef WALK_CUH
#define WALK_CUH

#include <stdexcept>
#include <cstddef>

#include "step.cuh"
#include "walk_iterator.cuh"
#include "../../common/const.cuh"
#include "../../common/macros.cuh"

// non-owning view of a single walk.
class Walk {
private:
    const int* nodes_;
    const int64_t* timestamps_;
    const int64_t* edge_ids_; // per-walk slice, length = max(0, length_ - 1)
    size_t length_;

public:
    HOST Walk(
        const int* nodes,
        const int64_t* timestamps,
        const int64_t* edge_ids,
        const size_t length)
        : nodes_(nodes), timestamps_(timestamps), edge_ids_(edge_ids), length_(length) {}

    HOST Step operator[](const size_t index) const {
        if (index >= length_) {
            throw std::out_of_range("Walk index out of range");
        }
        const int64_t incoming_edge_id = (index == 0) ? EMPTY_EDGE_ID : edge_ids_[index - 1];
        return {nodes_[index], timestamps_[index], incoming_edge_id};
    }

    HOST size_t size() const {
        return length_;
    }

    HOST bool empty() const {
        return length_ == 0;
    }

    HOST WalkIterator begin() const {
        return {nodes_, timestamps_, edge_ids_, 0, length_};
    }

    HOST WalkIterator end() const {
        return {nodes_, timestamps_, edge_ids_, length_, length_};
    }

    HOST const int* nodes() const {
        return nodes_;
    }

    HOST const int64_t* timestamps() const {
        return timestamps_;
    }

    HOST const int64_t* edge_ids() const {
        return edge_ids_;
    }

    HOST Step back() const {
        if (length_ == 0) {
            throw std::out_of_range("Walk is empty");
        }
        const size_t idx = length_ - 1;
        const int64_t incoming_edge_id = (idx == 0) ? EMPTY_EDGE_ID : edge_ids_[idx - 1];
        return {nodes_[idx], timestamps_[idx], incoming_edge_id};
    }

    HOST Step front() const {
        if (length_ == 0) {
            throw std::out_of_range("Walk is empty");
        }
        return {nodes_[0], timestamps_[0], EMPTY_EDGE_ID};
    }
};

#endif // WALK_CUH
