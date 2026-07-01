// Tests for the ExponentialWeightInverseDegree random picker.
//
// ExponentialWeightInverseDegree mirrors ExponentialWeight (same recency-first
// exp(scaled_t) factor, same weighted-ITS machinery, same cumulative-array
// plumbing) but replaces each timestamp group's count factor (group_size) with
//     Σ_{edge e in group} 1/(deg(src_e)^2 · deg(tgt_e)^2)
// so that, holding recency fixed, low-degree endpoints get MORE mass — the
// mirror of ExponentialWeight, whose edge-count selection favours high-degree
// nodes. The whole behaviour lives in the precomputed cumulative arrays, so the
// tests below assert directly on those arrays (deterministic, exact) plus one
// end-to-end sampling test that drives the full per-node selection dispatch.
//
// Per CPU_GPU_PAIRING.md every suite is a TYPED_TEST over GPU_USAGE_TYPES, so
// each case runs on both the CPU (_std) and GPU (_cuda) weight-build paths.

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <vector>

#include "../src/common/random_gen.cuh"
#include "../src/core/tempest.cuh"
#include "../src/graph/edge_data.cuh"
#include "../src/graph/node_edge_index.cuh"
#include "test_temporal_graph_utils.h"

template<typename T>
class ExpWeightInverseDegreeTest : public ::testing::Test {
protected:
    // A cumulative array (global, or one node's contiguous segment) must be
    // non-negative, non-decreasing, and end at 1.0 after normalization.
    static void verify_cumulative_segment(
        const std::vector<double>& cum, const size_t start, const size_t end) {
        ASSERT_GT(end, start);
        for (size_t i = start; i < end; ++i) {
            EXPECT_GE(cum[i], 0.0);
            if (i > start) EXPECT_GE(cum[i], cum[i - 1]);
        }
        EXPECT_NEAR(cum[end - 1], 1.0, 1e-6);
    }

    // De-cumulate one node's [start,end) segment of a per-node cumulative array
    // into per-group individual weights (the first group's weight is the raw
    // cumulative value, since the segment restarts the running sum per node).
    static std::vector<double> segment_individual(
        const std::vector<double>& cum, const size_t start, const size_t end) {
        std::vector<double> w;
        for (size_t i = start; i < end; ++i)
            w.push_back(i == start ? cum[i] : cum[i] - cum[i - 1]);
        return w;
    }

    static double node_degree(
        const node_edge_index::NodeEdgeIndexSnapshot& idx, const int n) {
        const double out = static_cast<double>(
            idx.node_group_outbound_offsets[n + 1] - idx.node_group_outbound_offsets[n]);
        double in = 0.0;
        if (idx.node_group_inbound_offsets.size() > static_cast<size_t>(n + 1)) {
            in = static_cast<double>(
                idx.node_group_inbound_offsets[n + 1] - idx.node_group_inbound_offsets[n]);
        }
        const double d = out + in;
        return d == 0.0 ? 1.0 : d;
    }
};

#ifdef HAS_CUDA
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>,
    std::integral_constant<bool, true>
>;
#else
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>
>;
#endif

TYPED_TEST_SUITE(ExpWeightInverseDegreeTest, GPU_USAGE_TYPES);

// ── 1. Degree-regular graph ⇒ identical to ExponentialWeight ───────────────
// On a graph where every node has equal total degree the per-edge discount
// 1/(deg^2·deg^2) is a constant shared by every group, so it cancels in the
// per-group / per-node normalization. The inverse-degree cumulative arrays must
// then be bit-for-bit (to fp tolerance) the ExponentialWeight arrays. This is
// the strongest invariant: it proves the ONLY thing the new picker changes is
// the relative degree weighting, nothing about the recency machinery.
TYPED_TEST(ExpWeightInverseDegreeTest, ReducesToExponentialWhenDegreesEqual) {
    // Directed circulant on 5 nodes: i->(i+1) and i->(i+2). Every node has
    // out-degree 2 and in-degree 2 ⇒ total degree 4, uniform across the graph.
    constexpr int n = 5;
    std::vector<Edge> edges;
    int64_t ts = 10;
    for (int i = 0; i < n; ++i) {
        edges.push_back(Edge{i, (i + 1) % n, ts}); ts += 10;
        edges.push_back(Edge{i, (i + 2) % n, ts}); ts += 10;
    }

    core::Tempest graph(
        /*is_directed=*/true, TypeParam::value, -1, /*enable_weight_computation=*/true,
        /*enable_temporal_node2vec=*/false, -1);
    test_util::add_edges(graph, edges);

    const auto e = edge_data::snapshot(graph.data());
    ASSERT_EQ(e.forward_cumulative_weights_inverse_degree.size(),
              e.forward_cumulative_weights_exponential.size());
    ASSERT_FALSE(e.forward_cumulative_weights_inverse_degree.empty());
    for (size_t i = 0; i < e.forward_cumulative_weights_exponential.size(); ++i) {
        EXPECT_NEAR(e.forward_cumulative_weights_inverse_degree[i],
                    e.forward_cumulative_weights_exponential[i], 1e-9);
        EXPECT_NEAR(e.backward_cumulative_weights_inverse_degree[i],
                    e.backward_cumulative_weights_exponential[i], 1e-9);
    }

    const auto idx = node_edge_index::snapshot(graph.data());
    ASSERT_EQ(idx.outbound_forward_cumulative_weights_inverse_degree.size(),
              idx.outbound_forward_cumulative_weights_exponential.size());
    ASSERT_FALSE(idx.outbound_forward_cumulative_weights_inverse_degree.empty());
    for (size_t i = 0; i < idx.outbound_forward_cumulative_weights_exponential.size(); ++i) {
        EXPECT_NEAR(idx.outbound_forward_cumulative_weights_inverse_degree[i],
                    idx.outbound_forward_cumulative_weights_exponential[i], 1e-9);
        EXPECT_NEAR(idx.outbound_backward_cumulative_weights_inverse_degree[i],
                    idx.outbound_backward_cumulative_weights_exponential[i], 1e-9);
    }
    // directed ⇒ inbound arrays exist and must also match.
    ASSERT_FALSE(idx.inbound_backward_cumulative_weights_inverse_degree.empty());
    for (size_t i = 0; i < idx.inbound_backward_cumulative_weights_exponential.size(); ++i) {
        EXPECT_NEAR(idx.inbound_backward_cumulative_weights_inverse_degree[i],
                    idx.inbound_backward_cumulative_weights_exponential[i], 1e-9);
    }
}

// Shared fixture graph for the discrimination tests: node 0 fans out to a
// low-degree target (node 1) and a high-degree target (node 2) at nearly the
// same time; nodes 3/4/5 pile extra inbound edges onto node 2 to inflate its
// degree without touching node 0's groups.
namespace {
    inline std::vector<Edge> low_vs_high_degree_edges() {
        return {
            Edge{0, 1, 100},   // node 0 -> low-degree target (node 1) @ 100
            Edge{0, 2, 101},   // node 0 -> high-degree target (node 2) @ 101
            Edge{3, 2, 50},    // inflate deg(node 2)
            Edge{4, 2, 60},
            Edge{5, 2, 70},
        };
    }
}

// ── 2. Recency-isolated proof that low degree is favoured ──────────────────
// Node 0's two outbound groups differ in BOTH recency and target degree, so a
// raw weight comparison confounds the two. Dividing the inverse-degree share
// ratio by the ExponentialWeight share ratio cancels the shared exp() recency
// factor EXACTLY, leaving only the degree effect, which must equal
// (deg_high / deg_low)^2. This is the precise, recency-free statement of the
// formula 1/(deg(src)^2·deg(tgt)^2) (the constant deg(node0)^2 cancels too).
TYPED_TEST(ExpWeightInverseDegreeTest, FavorsLowDegreeIsolatingRecency) {
    core::Tempest graph(
        /*is_directed=*/true, TypeParam::value, -1, true, false, -1);
    test_util::add_edges(graph, low_vs_high_degree_edges());

    const auto idx = node_edge_index::snapshot(graph.data());
    const size_t start = idx.count_ts_group_per_node_outbound[0];
    const size_t end   = idx.count_ts_group_per_node_outbound[1];
    ASSERT_EQ(end - start, 2u) << "node 0 should have two outbound timestamp groups";

    const auto exp_w = this->segment_individual(
        idx.outbound_forward_cumulative_weights_exponential, start, end);
    const auto id_w = this->segment_individual(
        idx.outbound_forward_cumulative_weights_inverse_degree, start, end);

    // group 0 = (0->1 @100, low degree), group 1 = (0->2 @101, high degree).
    const double ratio_exp = exp_w[0] / exp_w[1];
    const double ratio_id  = id_w[0]  / id_w[1];

    // Inverse-degree must tilt FURTHER toward the low-degree group than exp does.
    EXPECT_GT(ratio_id, ratio_exp);

    const double deg_low  = this->node_degree(idx, 1);
    const double deg_high = this->node_degree(idx, 2);
    ASSERT_GT(deg_high, deg_low);
    const double expected_factor = (deg_high / deg_low) * (deg_high / deg_low);

    // The recency factor cancels in the ratio-of-ratios; only degree^2 survives.
    EXPECT_NEAR(ratio_id / ratio_exp, expected_factor, expected_factor * 1e-4);
}

// ── 3. Cumulative-array sanity on a non-trivial graph ──────────────────────
TYPED_TEST(ExpWeightInverseDegreeTest, CumulativeArraysWellFormed) {
    core::Tempest graph(
        /*is_directed=*/true, TypeParam::value, -1, true, false, -1);
    test_util::add_edges(graph, low_vs_high_degree_edges());

    const auto e = edge_data::snapshot(graph.data());
    this->verify_cumulative_segment(
        e.forward_cumulative_weights_inverse_degree, 0,
        e.forward_cumulative_weights_inverse_degree.size());
    this->verify_cumulative_segment(
        e.backward_cumulative_weights_inverse_degree, 0,
        e.backward_cumulative_weights_inverse_degree.size());

    // Each node's per-node segment is normalized independently ⇒ ends at 1.0.
    const auto idx = node_edge_index::snapshot(graph.data());
    for (size_t node = 0; node + 1 < idx.count_ts_group_per_node_outbound.size(); ++node) {
        const size_t s = idx.count_ts_group_per_node_outbound[node];
        const size_t en = idx.count_ts_group_per_node_outbound[node + 1];
        if (en > s) {
            this->verify_cumulative_segment(
                idx.outbound_forward_cumulative_weights_inverse_degree, s, en);
            this->verify_cumulative_segment(
                idx.outbound_backward_cumulative_weights_inverse_degree, s, en);
        }
    }
}

// ── 4. End-to-end sampling drives the full per-node dispatch ───────────────
// Both pickers share the recency factor, so the only difference in node 0's
// forward selection is the degree discount. The inverse-degree picker must
// select the low-degree target (node 1, ts 100) a strictly larger fraction of
// the time than ExponentialWeight does. Exercises the runtime PickerType
// dispatch + the per-node inverse-degree cumulative arrays on both backends.
TYPED_TEST(ExpWeightInverseDegreeTest, SamplingPrefersLowDegreeMoreThanExp) {
    core::Tempest graph(
        /*is_directed=*/true, TypeParam::value, -1, true, false, -1);
    test_util::add_edges(graph, low_vs_high_degree_edges());

    constexpr int kSamples = 2000;
    auto low_fraction = [&](const RandomPickerType picker) {
        int low = 0, total = 0;
        for (int s = 0; s < kSamples; ++s) {
            const Edge edge = test_util::get_node_edge_at(
                graph.data(), /*node_id=*/0, picker, /*timestamp=*/99,
                /*prev_node=*/-1, /*forward=*/true);
            if (edge.ts < 0) continue;            // no edge selected (shouldn't happen)
            ++total;
            if (edge.ts == 100) ++low;            // ts 100 == low-degree target (node 1)
        }
        EXPECT_GT(total, 0);
        return static_cast<double>(low) / static_cast<double>(total);
    };

    const double frac_exp = low_fraction(RandomPickerType::ExponentialWeight);
    const double frac_id  = low_fraction(RandomPickerType::ExponentialWeightInverseDegree);

    EXPECT_GT(frac_id, frac_exp + 0.1)
        << "inverse-degree should pick the low-degree target far more often "
        << "(exp=" << frac_exp << ", inverse_degree=" << frac_id << ")";
    EXPECT_GT(frac_id, 0.9)
        << "with (deg_high/deg_low)^2 = 16, inverse-degree should pick the "
        << "low-degree target almost always (got " << frac_id << ")";
}
