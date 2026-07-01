#ifndef TEMPORAL_NODE2VEC_HELPERS_CUH
#define TEMPORAL_NODE2VEC_HELPERS_CUH

#include <algorithm>
#include <cstdint>
#include <cstring>
#include "../common/const.cuh"
#include "../common/macros.cuh"
#include "../data/temporal_graph_view.cuh"
#include "../random/pickers.cuh"
#include "walk_step_helpers.cuh"

#ifdef HAS_CUDA
#include <cuda/std/__algorithm/lower_bound.h>
#endif

namespace temporal_graph {

    // ------------------------------------------------------------------------
    // Temporal Node2Vec via rejection sampling on the static-exp proposal
    // (paper: TEA §2.3 III, Alg 2 L19–22).
    //
    // Per hop:
    //   1. Sample a candidate edge from the same per-edge distribution
    //      ExponentialWeight uses (group via cumulative-weight ITS,
    //      then uniform within the group — Tempest's weight tables
    //      already encode group_size · exp((t_g − t_max_u) · scale) for
    //      backward / group_size · exp((t_min_u − t_g) · scale) for
    //      forward, so this yields a per-edge probability ∝ exp(t_g · scale)
    //      after the per-node pivot constant cancels in normalization).
    //   2. Compute β(prev, dest(candidate)) — one binary search on the
    //      sorted adjacency list of prev.
    //   3. Accept with probability β / β_max.  If reject, retry.
    //
    // Per-hop cost: O(retries · log D_prev) β-evaluations vs the
    // previous β-weighted ITS's O(L · log D_prev).  For typical p, q
    // with β_max ≤ 2 · E[β], expected retries ≤ 2.
    //
    // NODE2VEC_MAX_RETRIES (const.cuh) caps the rejection loop;
    // on overflow we accept the last proposal defensively.

    // Stateless retry-random derivation.  Each retry consumes three
    // uniforms (group, edge, accept); we hash the seed doubles + retry
    // index via splitmix64 so the picker keeps its existing two-double
    // signature.  Quality is sufficient for the small number of retries.
    HOST DEVICE inline uint64_t double_to_bits(double x) {
        uint64_t u;
        // memcpy is HOST DEVICE friendly and avoids strict-aliasing UB.
        #if defined(__CUDA_ARCH__)
        u = __double_as_longlong(x);
        #else
        std::memcpy(&u, &x, sizeof(u));
        #endif
        return u;
    }

    HOST DEVICE inline double derive_uniform(
        double seed_a, double seed_b, int counter) {
        uint64_t s = double_to_bits(seed_a) * 0x9E3779B97F4A7C15ULL;
        s ^= double_to_bits(seed_b) + 0xBF58476D1CE4E5B9ULL;
        s += static_cast<uint64_t>(counter) * 0x94D049BB133111EBULL;
        // splitmix64 finalizer
        s ^= s >> 30;
        s *= 0xBF58476D1CE4E5B9ULL;
        s ^= s >> 27;
        s *= 0x94D049BB133111EBULL;
        s ^= s >> 31;
        // High 53 bits → uniform [0, 1)
        return static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0);
    }

    // β_max = max(1/p, 1, 1/q).  Tempest already stores inv_p and inv_q
    // on the view; this is one comparison chain.
    HOST DEVICE inline double compute_beta_max(double inv_p, double inv_q) {
        double m = 1.0;
        if (inv_p > m) m = inv_p;
        if (inv_q > m) m = inv_q;
        return m;
    }

    HOST inline bool is_node_adjacent_to_host(
        const TemporalGraphView& view,
        const int prev_node,
        const int candidate_node) {

        if (!view.enable_temporal_node2vec || view.node_adj_offsets == nullptr) {
            return false;
        }

        if (prev_node < 0
            || static_cast<size_t>(prev_node) + 1 >= view.node_adj_offsets_size) {
            return false;
        }

        const size_t start = view.node_adj_offsets[prev_node];
        const size_t end   = view.node_adj_offsets[prev_node + 1];

        if (start >= end) {
            return false;
        }

        return std::binary_search(
            view.node_adj_neighbors + start,
            view.node_adj_neighbors + end,
            candidate_node
        );
    }

    HOST inline double compute_node2vec_beta_host(
        const TemporalGraphView& view,
        const int prev_node,
        const int w) {
        if (w == prev_node) {
            return view.inv_p;
        }

        if (is_node_adjacent_to_host(view, prev_node, w)) {
            return 1.0;
        }

        return view.inv_q;
    }

    // Combined group+edge picker via rejection sampling on the static-exp
    // proposal.  Replaces the legacy two-stage β-weighted ITS.  Returns the
    // chosen edge_idx (global edge index into the CSR), or -1 on dead/empty.
    template<bool Forward, bool IsDirected>
    HOST long pick_random_temporal_node2vec_host(
        const TemporalGraphView& view,
        const int node_id,
        const int prev_node,
        const size_t valid_node_ts_group_begin,
        const size_t valid_node_ts_group_end,
        const size_t node_group_begin,
        const size_t node_group_end,
        const size_t* node_ts_groups_offsets,
        const size_t* node_ts_sorted_indices,
        const double* weights,
        const size_t weights_size,
        const double group_selector_rand_num,
        const double edge_selector_rand_num) {

        if (valid_node_ts_group_begin >= valid_node_ts_group_end || prev_node == -1) {
            return -1;
        }

        const double beta_max     = compute_beta_max(view.inv_p, view.inv_q);
        const double inv_beta_max = 1.0 / beta_max;

        long last_edge_idx = -1;

        for (int retry = 0; retry < NODE2VEC_MAX_RETRIES; ++retry) {
            const double g_rand = derive_uniform(
                group_selector_rand_num, edge_selector_rand_num, retry * 3 + 0);
            const double e_rand = derive_uniform(
                group_selector_rand_num, edge_selector_rand_num, retry * 3 + 1);
            const double a_rand = derive_uniform(
                group_selector_rand_num, edge_selector_rand_num, retry * 3 + 2);

            // Propose: static-exp group via the existing ExponentialWeight ITS.
            const long group_pos = random_pickers::pick_using_weight_based_picker(
                RandomPickerType::ExponentialWeight,
                weights, weights_size,
                valid_node_ts_group_begin, valid_node_ts_group_end,
                g_rand,
                /*slice_start=*/node_group_begin);
            if (group_pos == -1) return -1;

            const size_t edge_start = node_ts_groups_offsets[group_pos];
            const size_t edge_end = get_node_group_edge_end<Forward, IsDirected>(
                view, node_id, node_ts_groups_offsets,
                static_cast<size_t>(group_pos), node_group_end);
            if (edge_start >= edge_end) continue;

            // Uniform within the proposed group → per-edge probability
            // becomes ∝ group_size · exp((t_g − t_max_u) · scale) / group_size,
            // which after per-node normalization matches the un-pivoted
            // softmax over edge timestamps.
            const size_t local_pick = static_cast<size_t>(
                e_rand * static_cast<double>(edge_end - edge_start));
            const size_t pick_idx = edge_start +
                (local_pick < edge_end - edge_start ? local_pick : (edge_end - edge_start - 1));
            const long edge_idx = static_cast<long>(node_ts_sorted_indices[pick_idx]);

            // Compute β only for the proposed edge.
            const int candidate_v = get_candidate_node<Forward, IsDirected>(
                view, node_id, static_cast<size_t>(edge_idx));
            const double beta = compute_node2vec_beta_host(view, prev_node, candidate_v);

            last_edge_idx = edge_idx;
            if (a_rand < beta * inv_beta_max) {
                return edge_idx;
            }
        }

        // Defensive: retry cap exceeded (only reachable for degenerate p, q).
        // Accept the last proposal — its bias is bounded by the cap.
        return last_edge_idx;
    }

    #ifdef HAS_CUDA

    DEVICE inline bool is_node_adjacent_to_device(
        const TemporalGraphView& view,
        const int prev_node,
        const int candidate_node) {
        if (!view.enable_temporal_node2vec || view.node_adj_offsets == nullptr) {
            return false;
        }

        if (prev_node < 0
            || static_cast<size_t>(prev_node) + 1 >= view.node_adj_offsets_size) {
            return false;
        }

        const size_t start = view.node_adj_offsets[prev_node];
        const size_t end   = view.node_adj_offsets[prev_node + 1];

        if (start >= end) {
            return false;
        }

        const int* begin  = view.node_adj_neighbors + start;
        const int* finish = view.node_adj_neighbors + end;

        const int* it = cuda::std::lower_bound(begin, finish, candidate_node);
        return (it != finish && *it == candidate_node);
    }

    DEVICE inline double compute_node2vec_beta_device(
        const TemporalGraphView& view,
        const int prev_node,
        const int w) {
        if (w == prev_node) {
            return view.inv_p;
        }

        if (is_node_adjacent_to_device(view, prev_node, w)) {
            return 1.0;
        }

        return view.inv_q;
    }

    // Device variant of the rejection-sampling group+edge picker.  Returns
    // chosen edge_idx (global CSR index) or -1 on dead/empty.
    template<bool Forward, bool IsDirected>
    DEVICE long pick_random_temporal_node2vec_device(
        const TemporalGraphView& view,
        const int node_id,
        const int prev_node,
        const size_t valid_node_ts_group_begin,
        const size_t valid_node_ts_group_end,
        const size_t node_group_begin,
        const size_t node_group_end,
        const size_t* node_ts_groups_offsets,
        const size_t* node_ts_sorted_indices,
        const double* weights,
        const size_t weights_size,
        const double group_selector_rand_num,
        const double edge_selector_rand_num) {

        if (valid_node_ts_group_begin >= valid_node_ts_group_end || prev_node == -1) {
            return -1;
        }

        const double beta_max     = compute_beta_max(view.inv_p, view.inv_q);
        const double inv_beta_max = 1.0 / beta_max;

        long last_edge_idx = -1;

        for (int retry = 0; retry < NODE2VEC_MAX_RETRIES; ++retry) {
            const double g_rand = derive_uniform(
                group_selector_rand_num, edge_selector_rand_num, retry * 3 + 0);
            const double e_rand = derive_uniform(
                group_selector_rand_num, edge_selector_rand_num, retry * 3 + 1);
            const double a_rand = derive_uniform(
                group_selector_rand_num, edge_selector_rand_num, retry * 3 + 2);

            const long group_pos = random_pickers::pick_using_weight_based_picker(
                RandomPickerType::ExponentialWeight,
                weights, weights_size,
                valid_node_ts_group_begin, valid_node_ts_group_end,
                g_rand,
                /*slice_start=*/node_group_begin);
            if (group_pos == -1) return -1;

            const size_t edge_start = node_ts_groups_offsets[group_pos];
            const size_t edge_end = get_node_group_edge_end<Forward, IsDirected>(
                view, node_id, node_ts_groups_offsets,
                static_cast<size_t>(group_pos), node_group_end);
            if (edge_start >= edge_end) continue;

            const size_t local_pick = static_cast<size_t>(
                e_rand * static_cast<double>(edge_end - edge_start));
            const size_t pick_idx = edge_start +
                (local_pick < edge_end - edge_start ? local_pick : (edge_end - edge_start - 1));
            const long edge_idx = static_cast<long>(node_ts_sorted_indices[pick_idx]);

            const int candidate_v = get_candidate_node<Forward, IsDirected>(
                view, node_id, static_cast<size_t>(edge_idx));
            const double beta = compute_node2vec_beta_device(view, prev_node, candidate_v);

            last_edge_idx = edge_idx;
            if (a_rand < beta * inv_beta_max) {
                return edge_idx;
            }
        }

        return last_edge_idx;
    }

    #endif

}

#endif // TEMPORAL_NODE2VEC_HELPERS_CUH
