#include "walk_set_device.cuh"
#include "walk_set_host.cuh"

#include <cstddef>
#include <cstdint>
#include <cstring>

WalkSetDevice::WalkSetDevice(const size_t num_walks,
                             const size_t max_len,
                             const int walk_padding_value)
    : nodes_(true), timestamps_(true), walk_lens_(true), edge_ids_(true),
      cutoffs_(true),
      num_walks_(num_walks), max_len_(max_len),
      walk_padding_value_(walk_padding_value) {

    const size_t total       = num_walks * max_len;
    const size_t edge_total  = num_walks * (max_len == 0 ? 0 : max_len - 1);

    nodes_.resize(total);
    timestamps_.resize(total);
    walk_lens_.resize(num_walks);
    edge_ids_.resize(edge_total);
    cutoffs_.resize(num_walks);

    nodes_.fill(walk_padding_value);
    timestamps_.fill(EMPTY_TIMESTAMP_VALUE);
    walk_lens_.fill(static_cast<size_t>(0));
    edge_ids_.fill(EMPTY_EDGE_ID);
    cutoffs_.fill(NO_WALK_CUTOFF);
}

WalkSetView WalkSetDevice::make_view() {
    WalkSetView v{};
    v.nodes              = nodes_.data();
    v.timestamps         = timestamps_.data();
    v.walk_lens          = walk_lens_.data();
    v.edge_ids           = edge_ids_.data();
    v.cutoffs            = cutoffs_.data();
    v.num_walks          = num_walks_;
    v.max_len            = max_len_;
    v.walk_padding_value = walk_padding_value_;
    return v;
}

WalkSetHost WalkSetDevice::download_to_host() && {
    WalkSetHost host;
    host.overwrite_from_device_buffers(
        num_walks_, max_len_, walk_padding_value_,
        nodes_, timestamps_, walk_lens_, edge_ids_);

    num_walks_ = 0;
    max_len_   = 0;
    return host;
}
