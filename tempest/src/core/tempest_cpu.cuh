#ifndef TEMPEST_CPU_CUH
#define TEMPEST_CPU_CUH

#include <climits>
#include <cstdint>

#include "../common/macros.cuh"
#include "../data/enums.cuh"
#include "../data/temporal_graph_view.cuh"
#include "../data/walk_set/walk_set_view.cuh"
#include "../utils/random.cuh"
#include "../utils/utils.cuh"
#include "helpers.cuh"
#include "../graph/edge_selectors.cuh"
#include "../common/picker_dispatch.cuh"

namespace tempest {

    template<bool IsDirected, bool Forward, RandomPickerType EdgePickerType, RandomPickerType StartPickerType>
    HOST void generate_random_walk_and_time_std_new(
        const TemporalGraphView& view,
        const int walk_idx,
        WalkSetView walk_set,
        const int max_walk_len,
        const int* start_node_ids,
        const double* rand_nums) {

        if (max_walk_len == 0) return;

        const int walk_padding_value = walk_set.walk_padding_value;

        const size_t rand_nums_start_idx_for_walk = static_cast<size_t>(walk_idx)
            + static_cast<size_t>(walk_idx) * static_cast<size_t>(max_walk_len) * 2;

        const int64_t cutoff = walk_set.cutoffs[walk_idx];

        const auto padding_value = walk_set.nodes[walk_idx * max_walk_len];
        InternalEdge start_edge;
        if (start_node_ids[walk_idx] == -1) {
            start_edge = temporal_graph::get_edge_at_host<Forward, StartPickerType>(
                view,
                -1,
                rand_nums[rand_nums_start_idx_for_walk],
                rand_nums[rand_nums_start_idx_for_walk + 1]);
        } else {
            walk_set.nodes[walk_idx * max_walk_len] = start_node_ids[walk_idx];

            start_edge = temporal_graph::get_node_edge_at_host<Forward, StartPickerType, IsDirected>(
                view,
                start_node_ids[walk_idx],
                /*timestamp=*/-1,
                cutoff,
                /*prev_node=*/-1,
                rand_nums[rand_nums_start_idx_for_walk],
                rand_nums[rand_nums_start_idx_for_walk + 1]
            );
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

        if constexpr (IsDirected) {
            if constexpr (Forward) {
                walk_set.add_hop(walk_idx, start_src, current_timestamp);
                current_node = start_dst;
            } else {
                walk_set.add_hop(walk_idx, start_dst, current_timestamp);
                current_node = start_src;
            }
        } else {
            const int picked_node = (start_node_ids[walk_idx] != -1)
                                        ? start_node_ids[walk_idx]
                                        : pick_random_number(start_src, start_dst,
                                                             rand_nums[rand_nums_start_idx_for_walk + 2]);
            walk_set.add_hop(walk_idx, picked_node, current_timestamp);
            current_node = pick_other_number(start_src, start_dst, picked_node);
        }

        current_timestamp = start_ts;
        int64_t current_edge_id = start_edge.edge_id;

        int walk_len = 1;
        while (walk_len < max_walk_len && current_node != walk_padding_value) {
            const auto step_start_idx = rand_nums_start_idx_for_walk + walk_len * 2 + 1;
            const auto group_selector_rand_num = rand_nums[step_start_idx];
            const auto edge_selector_rand_num = rand_nums[step_start_idx + 1];

            walk_set.add_hop(walk_idx, current_node, current_timestamp, current_edge_id);

            InternalEdge next_edge = temporal_graph::get_node_edge_at_host<Forward, EdgePickerType, IsDirected>(
                view,
                current_node,
                current_timestamp,
                cutoff,
                prev_node,
                group_selector_rand_num,
                edge_selector_rand_num
            );

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

    template<bool IsDirected, bool Forward>
    HOST void dispatch_walk_generation_new(
        const TemporalGraphView& view,
        const int walk_idx,
        WalkSetView walk_set,
        const int max_walk_len,
        const int* start_node_ids,
        const double* rand_nums,
        const RandomPickerType edge_picker_type,
        const RandomPickerType start_picker_type) {
        dispatch_picker_type(edge_picker_type, [&](auto edge_tag) {
            constexpr auto kEdge = decltype(edge_tag)::value;
            dispatch_picker_type(start_picker_type, [&](auto start_tag) {
                constexpr auto kStart = decltype(start_tag)::value;
                generate_random_walk_and_time_std_new<
                    IsDirected, Forward, kEdge, kStart>(
                    view, walk_idx, walk_set, max_walk_len,
                    start_node_ids, rand_nums);
            });
        });
    }

    HOST inline void launch_random_walk_cpu_new(
        const TemporalGraphView& view,
        const bool is_directed,
        WalkSetView walk_set,
        const int max_walk_len,
        const int* start_node_ids,
        const size_t num_walks,
        const RandomPickerType edge_picker_type,
        const RandomPickerType start_picker_type,
        const WalkDirection walk_direction,
        const double* rand_nums) {
        const bool should_walk_forward = get_should_walk_forward(walk_direction);

        const RandomPickerType normalized_start_picker_type =
            start_picker_type == RandomPickerType::TemporalNode2Vec
                ? RandomPickerType::ExponentialWeight
                : start_picker_type;

        #pragma omp parallel for
        for (int walk_idx = 0; walk_idx < static_cast<int>(num_walks); walk_idx++) {
            if (is_directed) {
                if (should_walk_forward) {
                    dispatch_walk_generation_new<true, true>(
                        view, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums,
                        edge_picker_type, normalized_start_picker_type);
                } else {
                    dispatch_walk_generation_new<true, false>(
                        view, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums,
                        edge_picker_type, normalized_start_picker_type);
                }
            } else {
                if (should_walk_forward) {
                    dispatch_walk_generation_new<false, true>(
                        view, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums,
                        edge_picker_type, normalized_start_picker_type);
                } else {
                    dispatch_walk_generation_new<false, false>(
                        view, walk_idx, walk_set, max_walk_len, start_node_ids, rand_nums,
                        edge_picker_type, normalized_start_picker_type);
                }
            }
        }
    }

}

#endif // TEMPEST_CPU_CUH
