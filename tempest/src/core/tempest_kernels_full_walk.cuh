#ifndef TEMPEST_KERNELS_FULL_WALK_CUH
#define TEMPEST_KERNELS_FULL_WALK_CUH

#include "../data/walk_set/walk_set_view.cuh"
#include "../data/temporal_graph_view.cuh"
#include "../graph/temporal_graph.cuh"
#include "../utils/random.cuh"
#include "../utils/utils.cuh"
#include "helpers.cuh"
#include "../graph/edge_selectors.cuh"
#include "../common/picker_dispatch.cuh"

namespace tempest {

    #ifdef HAS_CUDA

    template<bool IsDirected, bool Forward, RandomPickerType EdgePickerType, RandomPickerType StartPickerType>
    __global__ void generate_random_walks_kernel(
        WalkSetView walk_set,
        TemporalGraphView view,
        const int *__restrict__ start_node_ids,
        const int max_walk_len,
        const size_t num_walks,
        const uint64_t base_seed) {

        const size_t walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (walk_idx >= num_walks) return;

        if (max_walk_len == 0) return;

        const int walk_padding_value = walk_set.walk_padding_value;

        PhiloxState rng;
        init_philox_state(rng, base_seed, static_cast<uint64_t>(walk_idx));

        const double r0 = draw_u01_philox(rng);
        const double r1 = draw_u01_philox(rng);

        const int64_t cutoff = walk_set.cutoffs[walk_idx];

        const auto padding_value = walk_set.nodes[walk_idx * max_walk_len];
        InternalEdge start_edge;
        if (start_node_ids[walk_idx] == -1) {
            start_edge = temporal_graph::get_edge_at_device<Forward, StartPickerType>(
                view,
                -1,
                r0,
                r1);
        } else {
            walk_set.nodes[walk_idx * max_walk_len] = start_node_ids[walk_idx];

            start_edge = temporal_graph::get_node_edge_at_device<Forward, StartPickerType, IsDirected>(
                view,
                start_node_ids[walk_idx],
                /*timestamp=*/-1,
                cutoff,
                /*prev_node=*/-1,
                r0,
                r1);
        }

        if (start_edge.i == -1) {
            walk_set.nodes[walk_idx * max_walk_len] = padding_value;
            return;
        }

        int current_node;
        int prev_node = -1;
        int64_t current_timestamp = Forward ? INT64_MIN : INT64_MAX;
        const int start_src = start_edge.u;
        const int start_dst = start_edge.i;
        const int64_t start_ts = start_edge.ts;

        // prev_node seeds from the walk's first node (slot 0), so the first
        // node2vec transition (slot 1 -> slot 2) is biased by slot 0 — matching
        // the NODE_GROUPED solo/coop paths, which read prev from walk history.
        // Leaving it at -1 would skip the beta term on that first hop entirely.
        // (No effect for non-node2vec pickers, which ignore prev_node.)
        if constexpr (IsDirected) {
            if constexpr (Forward) {
                walk_set.add_hop(walk_idx, start_src, current_timestamp);
                prev_node = start_src;
                current_node = start_dst;
            } else {
                walk_set.add_hop(walk_idx, start_dst, current_timestamp);
                prev_node = start_dst;
                current_node = start_src;
            }
        } else {
            const double r2 = draw_u01_philox(rng);

            const int picked_node = (start_node_ids[walk_idx] != -1)
                                        ? start_node_ids[walk_idx]
                                        : pick_random_number(start_src, start_dst, r2);

            walk_set.add_hop(walk_idx, picked_node, current_timestamp);
            prev_node = picked_node;
            current_node = pick_other_number(start_src, start_dst, picked_node);
        }

        current_timestamp = start_ts;
        int64_t current_edge_id = start_edge.edge_id;

        int walk_len = 1;
        while (walk_len < max_walk_len && current_node != walk_padding_value) {
            const double r_step0 = draw_u01_philox(rng);
            const double r_step1 = draw_u01_philox(rng);

            walk_set.add_hop(walk_idx, current_node, current_timestamp, current_edge_id);

            InternalEdge next_edge = temporal_graph::get_node_edge_at_device<Forward, EdgePickerType, IsDirected>(
                view,
                current_node,
                current_timestamp,
                cutoff,
                prev_node,
                r_step0,
                r_step1);

            if (next_edge.ts == -1) {
                current_node = walk_padding_value;
                continue;
            }

            if constexpr (IsDirected) {
                prev_node = current_node;
                current_node = Forward ? next_edge.i : next_edge.u;
            } else {
                prev_node = current_node;
                current_node = pick_other_number(next_edge.u, next_edge.i, current_node);
            }

            current_timestamp = next_edge.ts;
            current_edge_id = next_edge.edge_id;
            walk_len++;
        }

        if constexpr (!Forward) {
            walk_set.reverse_walk(walk_idx);
        }
    }

    inline void launch_random_walk_kernel_full_walk(
        const TemporalGraphView& view,
        const bool is_directed,
        WalkSetView walk_set_view,
        const int max_walk_len,
        const int *start_node_ids,
        const size_t num_walks,
        const RandomPickerType edge_picker_type,
        const RandomPickerType start_picker_type,
        const WalkDirection walk_direction,
        const uint64_t base_seed,
        const dim3 &grid_dim,
        const dim3 &block_dim,
        const cudaStream_t stream = 0) {
        dim3 grid = grid_dim;
        if (grid.x == 0) {
            grid.x = (num_walks + block_dim.x - 1) / block_dim.x;
        }

        const bool should_walk_forward = get_should_walk_forward(walk_direction);

        const int num_walks_int = static_cast<int>(num_walks);

        dispatch_bool(is_directed, [&](auto dir_tag) {
            constexpr bool kDir = decltype(dir_tag)::value;
            dispatch_bool(should_walk_forward, [&](auto fwd_tag) {
                constexpr bool kFwd = decltype(fwd_tag)::value;
                dispatch_picker_type(edge_picker_type, [&](auto edge_tag) {
                    constexpr auto kEdge = decltype(edge_tag)::value;
                    dispatch_picker_type(start_picker_type, [&](auto start_tag) {
                        constexpr auto kStart = decltype(start_tag)::value;
                        generate_random_walks_kernel<kDir, kFwd, kEdge, kStart>
                            <<<grid, block_dim, 0, stream>>>(
                                walk_set_view, view, start_node_ids,
                                max_walk_len, num_walks_int, base_seed);
                    });
                });
            });
        });
    }

    #endif

}

#endif //TEMPEST_KERNELS_FULL_WALK_CUH
