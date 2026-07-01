#ifndef EDGE_SELECTORS_CUH
#define EDGE_SELECTORS_CUH

#include <algorithm>
#include "../common/macros.cuh"
#include "../data/structs.cuh"
#include "../data/temporal_graph_view.cuh"
#include "../random/pickers.cuh"
#include "../utils/random.cuh"
#include "temporal_node2vec_helpers.cuh"

#ifdef HAS_CUDA
#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/upper_bound.h>
#endif

namespace temporal_graph {

    // view-based mirrors of edge_data:: query helpers (host)
    HOST inline size_t find_group_after_timestamp_host(
        const TemporalGraphView& view, const int64_t timestamp) {
        if (view.num_groups == 0) return 0;
        const int64_t* begin = view.unique_timestamps;
        const int64_t* end   = begin + view.num_groups;
        const auto it = std::upper_bound(begin, end, timestamp);
        return it - begin;
    }

    HOST inline size_t find_group_before_timestamp_host(
        const TemporalGraphView& view, const int64_t timestamp) {
        if (view.num_groups == 0) return 0;
        const int64_t* begin = view.unique_timestamps;
        const int64_t* end   = begin + view.num_groups;
        const auto it = std::lower_bound(begin, end, timestamp);
        return (it - begin) - 1;
    }

    HOST inline SizeRange get_timestamp_group_range_host(
        const TemporalGraphView& view, const size_t group_idx) {
        if (group_idx >= view.num_groups) return SizeRange{0, 0};
        return SizeRange{view.timestamp_group_offsets[group_idx],
                         view.timestamp_group_offsets[group_idx + 1]};
    }

    HOST inline bool is_node_active_host(
        const TemporalGraphView& view, const int node_id) {
        return node_id >= 0
            && static_cast<size_t>(node_id) < view.active_node_ids_size
            && view.active_node_ids[node_id] == 1;
    }

    // slice-parameterized stage-1 helpers; positions are local to the slice [0, G).
    // first_ts non-null = single-indirect probe (cooperative kernels preload smem);
    // null = double-indirect fallback used by solo/host callers.
    // Restrict a node's timestamp groups [0, G) to the slice a walk may sample.
    //   timestamp : the previous hop's time (monotonic walk bound). -1 = none
    //               (start hop). Forward keeps groups with ts > timestamp;
    //               Backward keeps groups with ts < timestamp.
    //   cutoff    : per-walk EXCLUSIVE upper bound on edge time (NO_WALK_CUTOFF
    //               = unbounded). Applies in BOTH directions — it always drops
    //               groups with ts >= cutoff, i.e. trims out_valid_end. This is
    //               what realises "walk this node as of time t": forward it caps
    //               every hop below t; backward it only bites on the start hop
    //               (later hops are already earlier than t by monotonicity).
    template <bool Forward>
    HOST DEVICE inline void filter_valid_groups_by_timestamp_slice(
        const size_t*  group_offsets,
        const int64_t* first_ts,
        const size_t*  node_ts_sorted_indices,
        const int64_t* view_timestamps,
        const int      G,
        const int64_t  timestamp,
        const int64_t  cutoff,
        int&           out_valid_begin,
        int&           out_valid_end) {

        out_valid_begin = 0;
        out_valid_end   = G;

        if (timestamp == -1 && cutoff == NO_WALK_CUTOFF) return;

        auto ts_of = [&](const int p) -> int64_t {
            if (first_ts != nullptr) return first_ts[p];
            return view_timestamps[node_ts_sorted_indices[group_offsets[p]]];
        };

        // Directional monotonic bound from the previous hop's timestamp.
        if (timestamp != -1) {
            if constexpr (Forward) {
                int lo = 0, hi = G;
                while (lo < hi) {
                    const int mid = lo + ((hi - lo) >> 1);
                    if (timestamp < ts_of(mid)) hi = mid;
                    else                         lo = mid + 1;
                }
                out_valid_begin = lo;
            } else {
                int lo = 0, hi = G;
                while (lo < hi) {
                    const int mid = lo + ((hi - lo) >> 1);
                    if (ts_of(mid) < timestamp) lo = mid + 1;
                    else                          hi = mid;
                }
                out_valid_end = lo;
            }
        }

        // Universal cutoff: exclude every group with ts >= cutoff. Search the
        // first such group within the surviving slice and pull out_valid_end
        // down to it (a no-op when the slice is already entirely below cutoff).
        if (cutoff != NO_WALK_CUTOFF) {
            int lo = out_valid_begin, hi = out_valid_end;
            while (lo < hi) {
                const int mid = lo + ((hi - lo) >> 1);
                if (ts_of(mid) < cutoff) lo = mid + 1;
                else                      hi = mid;
            }
            out_valid_end = lo;
        }
    }

    // returns the picked group's local index in [0, G), or -1.
    // cum_weights is the full per-graph array; slice_global_begin shifts
    // local <-> global for the weighted picker. s_cum_weights, when set,
    // redirects the picker's loads to a smem-resident copy of the slice.
    template <bool Forward, RandomPickerType PickerType>
    HOST DEVICE inline long find_group_pos_slice(
        const size_t*  group_offsets,
        const int64_t* first_ts,
        const size_t*  node_ts_sorted_indices,
        const int64_t* view_timestamps,
        const double*  cum_weights,
        const size_t   cum_weights_size,
        const size_t   slice_global_begin,
        const int      G,
        const int64_t  timestamp,
        const int64_t  cutoff,
        const double   r_group,
        const double*  s_cum_weights = nullptr) {

        int valid_begin = 0;
        int valid_end   = G;
        filter_valid_groups_by_timestamp_slice<Forward>(
            group_offsets, first_ts,
            node_ts_sorted_indices, view_timestamps,
            G, timestamp, cutoff,
            valid_begin, valid_end);
        if (valid_begin >= valid_end) return -1;

        if constexpr (random_pickers::is_index_based_picker(PickerType)) {
            const int available = valid_end - valid_begin;
            const int index = random_pickers::pick_using_index_based_picker(
                PickerType, 0, available, !Forward, r_group);
            if (index == -1 || index >= available) return -1;
            return Forward
                ? static_cast<long>(valid_begin + index)
                : static_cast<long>(valid_end - 1 - (available - index - 1));
        } else {
            const size_t g_begin =
                slice_global_begin + static_cast<size_t>(valid_begin);
            const size_t g_end   =
                slice_global_begin + static_cast<size_t>(valid_end);
            // per-node piecewise CDF; pass slice_global_begin so g_begin ==
            // slice_global_begin means "prefix 0" (don't read prev node's total)
            const int global_pos = random_pickers::pick_using_weight_based_picker(
                PickerType, cum_weights, cum_weights_size,
                g_begin, g_end, r_group,
                /*slice_start=*/slice_global_begin,
                /*smem_weights=*/s_cum_weights);
            if (global_pos == -1) return -1;
            return static_cast<long>(global_pos) -
                   static_cast<long>(slice_global_begin);
        }
    }

    template<bool Forward, RandomPickerType PickerType>
    HOST InternalEdge get_edge_at_host(
        const TemporalGraphView& view,
        const int64_t timestamp,
        const double group_selector_rand_num,
        const double edge_selector_rand_num) {
        if (view.num_edges == 0) return InternalEdge{-1, -1, -1, -1};

        const size_t num_groups = view.num_groups;
        if (num_groups == 0) return InternalEdge{-1, -1, -1, -1};

        long group_idx;
        if (timestamp != -1) {
            if constexpr (Forward) {
                const size_t first_group = find_group_after_timestamp_host(view, timestamp);
                const size_t available_groups = num_groups - first_group;
                if (available_groups == 0) return InternalEdge{-1, -1, -1, -1};

                if constexpr (random_pickers::is_index_based_picker(PickerType)) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available_groups),
                        false, group_selector_rand_num);
                    if (index == -1) return InternalEdge{-1, -1, -1, -1};

                    if (index >= available_groups) return InternalEdge{-1, -1, -1, -1};
                    group_idx = static_cast<long>(first_group + index);
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker(
                        PickerType,
                        view.forward_cumulative_weights_exponential,
                        view.forward_cumulative_weights_exponential_size,
                        first_group, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                }
            } else {
                const size_t last_group = find_group_before_timestamp_host(view, timestamp);
                if (last_group == static_cast<size_t>(-1)) return InternalEdge{-1, -1, -1, -1};

                const size_t available_groups = last_group + 1;
                if constexpr (random_pickers::is_index_based_picker(PickerType)) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available_groups),
                        true, group_selector_rand_num);
                    if (index == -1) return InternalEdge{-1, -1, -1, -1};

                    if (index >= available_groups) return InternalEdge{-1, -1, -1, -1};
                    group_idx = static_cast<long>(last_group) - static_cast<long>(available_groups - index - 1);
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker(
                        PickerType,
                        view.backward_cumulative_weights_exponential,
                        view.backward_cumulative_weights_exponential_size,
                        0, last_group + 1, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                }
            }
        } else {
            if constexpr (random_pickers::is_index_based_picker(PickerType)) {
                const auto index = random_pickers::pick_using_index_based_picker(
                    PickerType, 0, static_cast<int>(num_groups), !Forward, group_selector_rand_num);
                if (index == -1) return InternalEdge{-1, -1, -1, -1};

                if (index >= num_groups) return InternalEdge{-1, -1, -1, -1};
                group_idx = index;
            } else {
                if constexpr (Forward) {
                    group_idx = random_pickers::pick_using_weight_based_picker(
                        PickerType,
                        view.forward_cumulative_weights_exponential,
                        view.forward_cumulative_weights_exponential_size,
                        0, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker(
                        PickerType,
                        view.backward_cumulative_weights_exponential,
                        view.backward_cumulative_weights_exponential_size,
                        0, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                }
            }
        }

        const SizeRange group_range = get_timestamp_group_range_host(view, group_idx);
        if (group_range.from == group_range.to) {
            return InternalEdge{-1, -1, -1, -1};
        }

        const size_t random_idx = group_range.from +
                                  generate_random_number_bounded_by(
                                      static_cast<int>(group_range.to - group_range.from),
                                      edge_selector_rand_num);

        return InternalEdge{
            view.sources[random_idx],
            view.targets[random_idx],
            view.timestamps[random_idx],
            static_cast<int64_t>(random_idx)
        };
    }

    template<bool Forward, RandomPickerType PickerType, bool IsDirected>
    HOST InternalEdge get_node_edge_at_host(
        const TemporalGraphView& view,
        const int node_id,
        const int64_t timestamp,
        const int64_t cutoff,
        const int prev_node,
        const double group_selector_rand_num,
        const double edge_selector_rand_num) {
        if (!is_node_active_host(view, node_id)) {
            return InternalEdge{-1, -1, -1, -1};
        }

        const size_t* count_ts_group_per_node =
                Forward
                    ? view.count_ts_group_per_node_outbound
                    : (IsDirected
                           ? view.count_ts_group_per_node_inbound
                           : view.count_ts_group_per_node_outbound);

        const size_t* node_ts_groups_offsets =
                Forward
                    ? view.node_ts_group_outbound_offsets
                    : (IsDirected
                           ? view.node_ts_group_inbound_offsets
                           : view.node_ts_group_outbound_offsets);

        const size_t* node_ts_sorted_indices =
                Forward
                    ? view.node_ts_sorted_outbound_indices
                    : (IsDirected
                           ? view.node_ts_sorted_inbound_indices
                           : view.node_ts_sorted_outbound_indices);

        const size_t node_ts_sorted_indices_size =
                Forward
                    ? view.node_ts_sorted_outbound_indices_size
                    : (IsDirected
                           ? view.node_ts_sorted_inbound_indices_size
                           : view.node_ts_sorted_outbound_indices_size);

        const size_t* node_edge_offsets =
                Forward
                    ? view.node_group_outbound_offsets
                    : (IsDirected
                           ? view.node_group_inbound_offsets
                           : view.node_group_outbound_offsets);

        const double* weights =
                Forward
                    ? view.outbound_forward_cumulative_weights_exponential
                    : (IsDirected
                           ? view.inbound_backward_cumulative_weights_exponential
                           : view.outbound_backward_cumulative_weights_exponential);

        const size_t weights_size =
                Forward
                    ? view.outbound_forward_cumulative_weights_exponential_size
                    : (IsDirected
                           ? view.inbound_backward_cumulative_weights_exponential_size
                           : view.outbound_backward_cumulative_weights_exponential_size);

        const size_t node_group_begin = count_ts_group_per_node[node_id];
        const size_t node_group_end = count_ts_group_per_node[node_id + 1];

        if (node_group_begin == node_group_end) {
            return InternalEdge{-1, -1, -1, -1};
        }

        // node2vec stays inline since its picker depends on prev_node
        const int G = static_cast<int>(node_group_end - node_group_begin);
        long group_pos = -1;

        if constexpr (PickerType == RandomPickerType::TemporalNode2Vec) {
            int local_begin = 0;
            int local_end   = G;
            filter_valid_groups_by_timestamp_slice<Forward>(
                node_ts_groups_offsets + node_group_begin,
                /*first_ts=*/nullptr,
                node_ts_sorted_indices,
                view.timestamps,
                G, timestamp, cutoff,
                local_begin, local_end);
            if (local_begin >= local_end) {
                return InternalEdge{-1, -1, -1, -1};
            }
            const size_t valid_begin = node_group_begin + static_cast<size_t>(local_begin);
            const size_t valid_end   = node_group_begin + static_cast<size_t>(local_end);

            if (prev_node == -1) {
                // First-hop: no β term — fall through to plain ExponentialWeight.
                group_pos = random_pickers::pick_using_weight_based_picker(
                    RandomPickerType::ExponentialWeight,
                    weights, weights_size,
                    valid_begin, valid_end,
                    group_selector_rand_num,
                    /*slice_start=*/node_group_begin);
                if (group_pos == -1) {
                    return InternalEdge{-1, -1, -1, -1};
                }
            } else {
                // Stage-2 rejection sampling: returns final edge_idx directly,
                // bypassing the group+edge split used for non-n2v pickers.
                const long n2v_edge_idx =
                    pick_random_temporal_node2vec_host<Forward, IsDirected>(
                        view, node_id, prev_node,
                        valid_begin, valid_end,
                        node_group_begin, node_group_end,
                        node_ts_groups_offsets,
                        node_ts_sorted_indices,
                        weights, weights_size,
                        group_selector_rand_num,
                        edge_selector_rand_num);
                if (n2v_edge_idx == -1) {
                    return InternalEdge{-1, -1, -1, -1};
                }
                return InternalEdge{
                    view.sources[n2v_edge_idx],
                    view.targets[n2v_edge_idx],
                    view.timestamps[n2v_edge_idx],
                    n2v_edge_idx
                };
            }
        } else {
            const long local_pos = find_group_pos_slice<Forward, PickerType>(
                node_ts_groups_offsets + node_group_begin,
                /*first_ts=*/nullptr,
                node_ts_sorted_indices,
                view.timestamps,
                weights, weights_size,
                node_group_begin,
                G, timestamp, cutoff,
                group_selector_rand_num);
            if (local_pos == -1) {
                return InternalEdge{-1, -1, -1, -1};
            }
            group_pos = local_pos + static_cast<long>(node_group_begin);
        }

        const size_t valid_edge_start = node_ts_groups_offsets[group_pos];
        const size_t valid_edge_end =
                (static_cast<size_t>(group_pos) + 1 < node_group_end)
                    ? node_ts_groups_offsets[group_pos + 1]
                    : node_edge_offsets[node_id + 1];

        if (valid_edge_start >= valid_edge_end ||
            valid_edge_start >= node_ts_sorted_indices_size ||
            valid_edge_end > node_ts_sorted_indices_size) {
            return InternalEdge{-1, -1, -1, -1};
        }

        // TN2V with prev_node != -1 returned early via the rejection sampler.
        // Remaining paths (non-n2v, or first-hop n2v) sample uniformly within
        // the chosen group.
        const long edge_idx = static_cast<long>(node_ts_sorted_indices[
            valid_edge_start +
            generate_random_number_bounded_by(
                static_cast<int>(valid_edge_end - valid_edge_start),
                edge_selector_rand_num)]);

        return InternalEdge{
            view.sources[edge_idx],
            view.targets[edge_idx],
            view.timestamps[edge_idx],
            edge_idx
        };
    }

    #ifdef HAS_CUDA

    DEVICE inline size_t find_group_after_timestamp(
        const TemporalGraphView& view, const int64_t timestamp) {
        if (view.num_groups == 0) return 0;
        const int64_t* begin = view.unique_timestamps;
        const int64_t* end   = begin + view.num_groups;
        const auto it = cuda::std::upper_bound(begin, end, timestamp);
        return it - begin;
    }

    DEVICE inline size_t find_group_before_timestamp(
        const TemporalGraphView& view, const int64_t timestamp) {
        if (view.num_groups == 0) return 0;
        const int64_t* begin = view.unique_timestamps;
        const int64_t* end   = begin + view.num_groups;
        const auto it = cuda::std::lower_bound(begin, end, timestamp);
        return (it - begin) - 1;
    }

    DEVICE inline SizeRange get_timestamp_group_range(
        const TemporalGraphView& view, size_t group_idx) {
        if (group_idx >= view.num_groups) return SizeRange{0, 0};
        return SizeRange{view.timestamp_group_offsets[group_idx],
                         view.timestamp_group_offsets[group_idx + 1]};
    }

    DEVICE inline bool is_node_active(const TemporalGraphView& view,
                                       const int node_id) {
        return node_id >= 0
            && static_cast<size_t>(node_id) < view.active_node_ids_size
            && view.active_node_ids[node_id] == 1;
    }

    template<bool Forward, RandomPickerType PickerType>
    DEVICE InternalEdge get_edge_at_device(
        const TemporalGraphView& view,
        const int64_t timestamp,
        const double group_selector_rand_num,
        const double edge_selector_rand_num) {
        if (view.num_edges == 0) return InternalEdge{-1, -1, -1, -1};

        const size_t num_groups = view.num_groups;
        if (num_groups == 0) return InternalEdge{-1, -1, -1, -1};

        long group_idx;
        if (timestamp != -1) {
            if constexpr (Forward) {
                const size_t first_group = find_group_after_timestamp(view, timestamp);
                const size_t available_groups = num_groups - first_group;
                if (available_groups == 0) return InternalEdge{-1, -1, -1, -1};

                if constexpr (random_pickers::is_index_based_picker(PickerType)) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available_groups),
                        false, group_selector_rand_num);
                    if (index == -1) return InternalEdge{-1, -1, -1, -1};

                    if (index >= available_groups) return InternalEdge{-1, -1, -1, -1};
                    group_idx = static_cast<long>(first_group + index);
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker(
                        PickerType,
                        view.forward_cumulative_weights_exponential,
                        view.forward_cumulative_weights_exponential_size,
                        first_group, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                }
            } else {
                const size_t last_group = find_group_before_timestamp(view, timestamp);
                if (last_group == static_cast<size_t>(-1)) return InternalEdge{-1, -1, -1, -1};

                const size_t available_groups = last_group + 1;
                if constexpr (random_pickers::is_index_based_picker(PickerType)) {
                    const auto index = random_pickers::pick_using_index_based_picker(
                        PickerType, 0, static_cast<int>(available_groups),
                        true, group_selector_rand_num);
                    if (index == -1) return InternalEdge{-1, -1, -1, -1};

                    if (index >= available_groups) return InternalEdge{-1, -1, -1, -1};
                    group_idx = static_cast<long>(last_group) - static_cast<long>(available_groups - index - 1);
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker(
                        PickerType,
                        view.backward_cumulative_weights_exponential,
                        view.backward_cumulative_weights_exponential_size,
                        0, last_group + 1, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                }
            }
        } else {
            if constexpr (random_pickers::is_index_based_picker(PickerType)) {
                const auto index = random_pickers::pick_using_index_based_picker(
                    PickerType, 0, static_cast<int>(num_groups), !Forward, group_selector_rand_num);
                if (index == -1) return InternalEdge{-1, -1, -1, -1};

                if (index >= num_groups) return InternalEdge{-1, -1, -1, -1};
                group_idx = index;
            } else {
                if constexpr (Forward) {
                    group_idx = random_pickers::pick_using_weight_based_picker(
                        PickerType,
                        view.forward_cumulative_weights_exponential,
                        view.forward_cumulative_weights_exponential_size,
                        0, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                } else {
                    group_idx = random_pickers::pick_using_weight_based_picker(
                        PickerType,
                        view.backward_cumulative_weights_exponential,
                        view.backward_cumulative_weights_exponential_size,
                        0, num_groups, group_selector_rand_num);
                    if (group_idx == -1) return InternalEdge{-1, -1, -1, -1};
                }
            }
        }

        const SizeRange group_range = get_timestamp_group_range(view, group_idx);
        if (group_range.from == group_range.to) {
            return InternalEdge{-1, -1, -1, -1};
        }

        const size_t random_idx = group_range.from +
                                  generate_random_number_bounded_by(
                                      static_cast<int>(group_range.to - group_range.from),
                                      edge_selector_rand_num);

        return InternalEdge{
            view.sources[random_idx],
            view.targets[random_idx],
            view.timestamps[random_idx],
            static_cast<int64_t>(random_idx),
        };
    }

    template<bool Forward, RandomPickerType PickerType, bool IsDirected>
    DEVICE InternalEdge get_node_edge_at_device(
        const TemporalGraphView& view,
        const int node_id,
        const int64_t timestamp,
        const int64_t cutoff,
        const int prev_node,
        const double group_selector_rand_num,
        const double edge_selector_rand_num) {
        if (!is_node_active(view, node_id)) {
            return InternalEdge{-1, -1, -1, -1};
        }

        const size_t* count_ts_group_per_node =
                Forward
                    ? view.count_ts_group_per_node_outbound
                    : (IsDirected
                           ? view.count_ts_group_per_node_inbound
                           : view.count_ts_group_per_node_outbound);

        const size_t* node_ts_groups_offsets =
                Forward
                    ? view.node_ts_group_outbound_offsets
                    : (IsDirected
                           ? view.node_ts_group_inbound_offsets
                           : view.node_ts_group_outbound_offsets);

        const size_t* node_ts_sorted_indices =
                Forward
                    ? view.node_ts_sorted_outbound_indices
                    : (IsDirected
                           ? view.node_ts_sorted_inbound_indices
                           : view.node_ts_sorted_outbound_indices);

        const size_t node_ts_sorted_indices_size =
                Forward
                    ? view.node_ts_sorted_outbound_indices_size
                    : (IsDirected
                           ? view.node_ts_sorted_inbound_indices_size
                           : view.node_ts_sorted_outbound_indices_size);

        const size_t* node_edge_offsets =
                Forward
                    ? view.node_group_outbound_offsets
                    : (IsDirected
                           ? view.node_group_inbound_offsets
                           : view.node_group_outbound_offsets);

        const double* weights =
                Forward
                    ? view.outbound_forward_cumulative_weights_exponential
                    : (IsDirected
                           ? view.inbound_backward_cumulative_weights_exponential
                           : view.outbound_backward_cumulative_weights_exponential);

        const size_t weights_size =
                Forward
                    ? view.outbound_forward_cumulative_weights_exponential_size
                    : (IsDirected
                           ? view.inbound_backward_cumulative_weights_exponential_size
                           : view.outbound_backward_cumulative_weights_exponential_size);

        const size_t node_group_begin = count_ts_group_per_node[node_id];
        const size_t node_group_end = count_ts_group_per_node[node_id + 1];

        if (node_group_begin == node_group_end) {
            return InternalEdge{-1, -1, -1, -1};
        }

        // inline binary search; routing through slice helpers added a
        // 3-deep dependent-load chain in the comparator (vs 2-deep here)
        // and a per-probe runtime branch on first_ts. solo/host hot path.
        size_t valid_begin = node_group_begin;
        size_t valid_end = node_group_end;
        const int64_t* view_timestamps = view.timestamps;

        if (timestamp != -1) {
            if constexpr (Forward) {
                const auto it = cuda::std::upper_bound(
                    node_ts_groups_offsets + static_cast<long>(node_group_begin),
                    node_ts_groups_offsets + static_cast<long>(node_group_end),
                    timestamp,
                    [view_timestamps, node_ts_sorted_indices](const int64_t ts, const size_t pos) {
                        return ts < view_timestamps[node_ts_sorted_indices[pos]];
                    });

                valid_begin = static_cast<size_t>(it - node_ts_groups_offsets);
            } else {
                const auto it = cuda::std::lower_bound(
                    node_ts_groups_offsets + static_cast<long>(node_group_begin),
                    node_ts_groups_offsets + static_cast<long>(node_group_end),
                    timestamp,
                    [view_timestamps, node_ts_sorted_indices](const size_t pos, const int64_t ts) {
                        return view_timestamps[node_ts_sorted_indices[pos]] < ts;
                    });

                valid_end = static_cast<size_t>(it - node_ts_groups_offsets);
            }
        }

        // Universal cutoff: drop groups with ts >= cutoff (exclusive upper
        // bound). lower_bound over the surviving slice → first group at/after
        // the cutoff; pull valid_end down to it.
        if (cutoff != NO_WALK_CUTOFF) {
            const auto it = cuda::std::lower_bound(
                node_ts_groups_offsets + static_cast<long>(valid_begin),
                node_ts_groups_offsets + static_cast<long>(valid_end),
                cutoff,
                [view_timestamps, node_ts_sorted_indices](const size_t pos, const int64_t ts) {
                    return view_timestamps[node_ts_sorted_indices[pos]] < ts;
                });

            valid_end = static_cast<size_t>(it - node_ts_groups_offsets);
        }

        if (valid_begin >= valid_end) {
            return InternalEdge{-1, -1, -1, -1};
        }

        long group_pos = -1;

        if constexpr (random_pickers::is_index_based_picker(PickerType)) {
            const size_t available = valid_end - valid_begin;

            const auto index = random_pickers::pick_using_index_based_picker(
                PickerType,
                0,
                static_cast<int>(available),
                !Forward,
                group_selector_rand_num);

            if (index == -1 || index >= available) {
                return InternalEdge{-1, -1, -1, -1};
            }

            group_pos = Forward
                            ? static_cast<long>(valid_begin + index)
                            : static_cast<long>(valid_end - 1 - (available - index - 1));
        } else {
            if constexpr (PickerType == RandomPickerType::TemporalNode2Vec) {
                if (prev_node == -1) {
                    // First-hop: no β — plain ExponentialWeight.
                    group_pos = random_pickers::pick_using_weight_based_picker(
                        RandomPickerType::ExponentialWeight,
                        weights, weights_size,
                        valid_begin, valid_end,
                        group_selector_rand_num,
                        /*slice_start=*/node_group_begin);
                } else {
                    // Stage-2 rejection sampler returns final edge_idx directly.
                    const long n2v_edge_idx =
                        pick_random_temporal_node2vec_device<Forward, IsDirected>(
                            view, node_id, prev_node,
                            valid_begin, valid_end,
                            node_group_begin, node_group_end,
                            node_ts_groups_offsets,
                            node_ts_sorted_indices,
                            weights, weights_size,
                            group_selector_rand_num,
                            edge_selector_rand_num);
                    if (n2v_edge_idx == -1) {
                        return InternalEdge{-1, -1, -1, -1};
                    }
                    return InternalEdge{
                        view.sources[n2v_edge_idx],
                        view.targets[n2v_edge_idx],
                        view.timestamps[n2v_edge_idx],
                        n2v_edge_idx
                    };
                }
            } else {
                group_pos = random_pickers::pick_using_weight_based_picker(
                    PickerType,
                    weights, weights_size,
                    valid_begin, valid_end,
                    group_selector_rand_num,
                    /*slice_start=*/node_group_begin);
            }

            if (group_pos == -1) {
                return InternalEdge{-1, -1, -1, -1};
            }
        }

        const size_t valid_edge_start = node_ts_groups_offsets[group_pos];
        const size_t valid_edge_end =
                (static_cast<size_t>(group_pos) + 1 < node_group_end)
                    ? node_ts_groups_offsets[group_pos + 1]
                    : node_edge_offsets[node_id + 1];

        if (valid_edge_start >= valid_edge_end ||
            valid_edge_start >= node_ts_sorted_indices_size ||
            valid_edge_end > node_ts_sorted_indices_size) {
            return InternalEdge{-1, -1, -1, -1};
        }

        // TN2V with prev_node != -1 returned early via the rejection sampler.
        const long edge_idx = static_cast<long>(node_ts_sorted_indices[
            valid_edge_start +
            generate_random_number_bounded_by(
                static_cast<int>(valid_edge_end - valid_edge_start),
                edge_selector_rand_num)]);

        return InternalEdge{
            view.sources[edge_idx],
            view.targets[edge_idx],
            view.timestamps[edge_idx],
            edge_idx
        };
    }

    #endif
}

#endif // EDGE_SELECTORS_CUH
