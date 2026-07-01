#ifndef WALKS_ITERATOR_IMPL_CUH
#define WALKS_ITERATOR_IMPL_CUH

#include "walks_iterator.cuh"

HOST inline void WalksIterator::find_next_non_empty() {
    while (walk_index_ < max_walks_ && walk_lens_[walk_index_] == 0) {
        ++walk_index_;
    }
}

HOST inline WalksIterator::WalksIterator(
    const int*     nodes,
    const int64_t* timestamps,
    const size_t*  walk_lens,
    const int64_t* edge_ids,
    const size_t   num_walks,
    const size_t   max_len,
    const size_t   start_index)
    : nodes_(nodes),
      timestamps_(timestamps),
      walk_lens_(walk_lens),
      edge_ids_(edge_ids),
      max_len_(max_len),
      max_walks_(num_walks),
      walk_index_(start_index) {
    find_next_non_empty();
}

HOST inline Walk WalksIterator::operator*() const {
    const size_t len = walk_lens_[walk_index_];
    const size_t node_offset = walk_index_ * max_len_;
    const size_t edge_stride = (max_len_ > 0) ? (max_len_ - 1) : 0;
    const size_t edge_offset = walk_index_ * edge_stride;

    return {
        nodes_ ? nodes_ + node_offset : nullptr,
        timestamps_ ? timestamps_ + node_offset : nullptr,
        edge_ids_ ? edge_ids_ + edge_offset : nullptr,
        len
    };
}

HOST inline WalksIterator& WalksIterator::operator++() {
    ++walk_index_;
    find_next_non_empty();
    return *this;
}

HOST inline WalksIterator WalksIterator::operator++(int) {
    const WalksIterator temp = *this;
    ++(*this);
    return temp;
}

HOST inline bool WalksIterator::operator==(const WalksIterator& other) const {
    return walk_index_ == other.walk_index_;
}

HOST inline bool WalksIterator::operator!=(const WalksIterator& other) const {
    return walk_index_ != other.walk_index_;
}

#endif // WALKS_ITERATOR_IMPL_CUH
