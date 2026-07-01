#ifndef DATA_WALK_SET_WALK_SET_VIEW_CUH
#define DATA_WALK_SET_WALK_SET_VIEW_CUH

#include <cstddef>
#include <cstdint>

#include "../../common/macros.cuh"
#include "../../common/const.cuh"

// non-owning POD view of WalkSetDevice buffers for pass-by-value to kernels.
struct WalkSetView {
    int*     nodes;
    int64_t* timestamps;
    size_t*  walk_lens;
    int64_t* edge_ids;

    // Per-walk exclusive start-time cutoff (length num_walks). NO_WALK_CUTOFF
    // = unbounded. Read by the edge selectors at every hop; it is in-progress
    // walk state (the walk's time horizon), not part of the released output.
    int64_t* cutoffs;

    size_t num_walks;
    size_t max_len;
    int    walk_padding_value;

    // edge_id is the transition INTO this hop; skipped when hop_pos == 0.
    HOST DEVICE void add_hop(
        const int walk_idx,
        const int node,
        const int64_t timestamp,
        const int64_t edge_id = EMPTY_EDGE_ID) const {

        const size_t hop_pos = walk_lens[walk_idx];
        const size_t hop_offset =
            static_cast<size_t>(walk_idx) * max_len + hop_pos;

        nodes[hop_offset] = node;
        timestamps[hop_offset] = timestamp;

        if (edge_id != EMPTY_EDGE_ID && hop_pos > 0) {
            const size_t edge_base =
                static_cast<size_t>(walk_idx) * (max_len - 1);
            edge_ids[edge_base + (hop_pos - 1)] = edge_id;
        }

        walk_lens[walk_idx] = hop_pos + 1;
    }

    // backward-in-time walks need this so callers see chronological order.
    HOST DEVICE void reverse_walk(const int walk_idx) const {
        const size_t walk_length = walk_lens[walk_idx];
        if (walk_length <= 1) return;

        const size_t node_start = static_cast<size_t>(walk_idx) * max_len;
        const size_t node_end   = node_start + walk_length - 1;

        for (size_t i = 0; i < walk_length / 2; ++i) {
            const size_t l = node_start + i;
            const size_t r = node_end   - i;

            const int tmp_n = nodes[l]; nodes[l] = nodes[r]; nodes[r] = tmp_n;
            const int64_t tmp_t = timestamps[l];
            timestamps[l] = timestamps[r];
            timestamps[r] = tmp_t;
        }

        const size_t edge_count = walk_length - 1;
        if (edge_count > 1) {
            const size_t edge_start =
                static_cast<size_t>(walk_idx) * (max_len - 1);
            const size_t edge_end = edge_start + edge_count - 1;
            for (size_t i = 0; i < edge_count / 2; ++i) {
                const size_t l = edge_start + i;
                const size_t r = edge_end   - i;
                const int64_t tmp_e = edge_ids[l];
                edge_ids[l] = edge_ids[r];
                edge_ids[r] = tmp_e;
            }
        }
    }
};

#endif // DATA_WALK_SET_WALK_SET_VIEW_CUH
