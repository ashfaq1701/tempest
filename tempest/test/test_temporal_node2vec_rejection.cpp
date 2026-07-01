// Hardening tests for the Stage-2 temporal-node2vec rejection sampler,
// covering both walk directions (Forward_In_Time and Backward_In_Time).
//
// The base test_temporal_node2vec.cpp already covers the headline
// invariants (β rules, sentinel handling, adjacency-CSR validity, and
// distribution-following under the default p, q).  These additional
// tests cover the parts of the rejection-sampling implementation that
// the base tests don't directly exercise, AND duplicate the coverage
// across forward and backward walks so a direction-specific regression
// can't slip in unnoticed:
//
//   1. Retry-loop termination under degenerate (p, q) — confirms the
//      retry cap + defensive accept-last keeps the picker returning
//      valid edges (no sentinels, no infinite loops) even when the
//      acceptance ratio is bad.  Performance-regression guard.  Runs
//      in both forward and backward modes.
//
//   2. Distribution-following across a (p, q) sweep — generalises the
//      base PQBias test to the full node2vec literature parameter
//      sweep {(1,1), (1,2), (0.5,2), (2,0.5), (0.25,4)}.  Catches β-
//      handling bugs at the parameter boundaries that the default
//      (p=2, q=0.5) cell happens to miss.  Runs in both forward and
//      backward modes.
//
// The β math is direction-agnostic: Tempest's node_adj_neighbors CSR is
// built symmetrically (every edge u→v writes both u into N(v) and v
// into N(u), edge_data.cu:327-373), so "is candidate v ∈ N(prev_node)"
// returns the same answer regardless of walk direction.  The expected
// per-edge probabilities in this file are therefore identical across
// directions; only the fixture-graph wiring (V as source vs V as
// target) and the candidate-extraction field (picked.i for forward,
// picked.u for backward + directed) differ.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../src/core/tempest.cuh"
#include "test_temporal_graph_utils.h"

namespace {

constexpr bool kIsDirected      = true;
constexpr int64_t kSentinelTs   = -1;
constexpr int  kSentinelNode    = -1;

inline bool is_sentinel(const Edge& e) {
    return e.u == kSentinelNode && e.i == kSentinelNode && e.ts == kSentinelTs;
}

// ---------------------------------------------------------------------------
// Shared n2v graph layout — three β candidates around centre vertex V.
//   • Forward layout : V has three OUTBOUND edges V→P, V→N, V→F.
//   • Backward layout: V has three INBOUND  edges P→V, N→V, F→V.
// In both layouts, an additional ADJ edge P→N puts N in N(P) via the
// symmetric node_adj_neighbors CSR (β = 1).  The directions of the V-
// adjacent edges flip with the walk direction so the candidate set at V
// is exactly {P, N, F} in either mode.
// ---------------------------------------------------------------------------
constexpr int      V            = 0;
constexpr int      P            = 1;     // "previous": β = 1/p
constexpr int      N            = 2;     // "neighbor of P": β = 1
constexpr int      F            = 3;     // "far from P":   β = 1/q
constexpr int64_t  EDGE_TS      = 100;
constexpr int64_t  ADJ_EDGE_TS  = 50;

template <bool UseGpu, bool Forward>
core::Tempest make_pq_graph(double p, double q) {
    core::Tempest g{
        /*is_directed=*/         kIsDirected,
        /*use_gpu=*/             UseGpu,
        /*max_time_capacity=*/   -1,
        /*enable_weight_computation=*/ true,
        /*enable_temporal_node2vec=*/  true,
        /*timescale_bound=*/     -1,
        /*node2vec_p=*/          p,
        /*node2vec_q=*/          q,
    };
    if constexpr (Forward) {
        test_util::add_edges(g, {
            Edge{V, P, EDGE_TS},
            Edge{V, N, EDGE_TS},
            Edge{V, F, EDGE_TS},
            Edge{P, N, ADJ_EDGE_TS},   // makes N adjacent to P
        });
    } else {
        // Mirror image: V is the target instead of the source so V's
        // INBOUND candidate set is {P, N, F}.
        test_util::add_edges(g, {
            Edge{P, V, EDGE_TS},
            Edge{N, V, EDGE_TS},
            Edge{F, V, EDGE_TS},
            Edge{P, N, ADJ_EDGE_TS},   // same ADJ edge — N still in N(P)
        });
    }
    return g;
}

// ---------------------------------------------------------------------------
// Direction-aware accessors.  For directed graphs:
//   • Forward walk : picked edge is (V → candidate); .u=V, .i=candidate.
//   • Backward walk: picked edge is (candidate → V); .u=candidate, .i=V.
// ---------------------------------------------------------------------------
inline int current_of(const Edge& e, bool forward) {
    return forward ? e.u : e.i;
}
inline int candidate_of(const Edge& e, bool forward) {
    return forward ? e.i : e.u;
}

// ---------------------------------------------------------------------------
// Analytic per-edge probability under TN2V.  The three V-adjacent edges
// all share timestamp EDGE_TS, so under the strict-paper exp bias their
// static weights are equal (= exp(0)=1 after the per-vertex t-shift);
// only the β factor matters.
//
//   target weight  ∝  β(prev=P, candidate)
//                  =  1/p  for candidate == P
//                     1    for candidate == N (in N(P) via the ADJ edge)
//                     1/q  for candidate == F
// ---------------------------------------------------------------------------
struct PQProbs { double p_dest, n_dest, f_dest; };

PQProbs analytic_pq_probs(const double p, const double q) {
    const double w_p = 1.0 / p;
    const double w_n = 1.0;
    const double w_f = 1.0 / q;
    const double tot = w_p + w_n + w_f;
    return {w_p / tot, w_n / tot, w_f / tot};
}

// ---------------------------------------------------------------------------
// Typed test fixture: cross-product of (host / device) × (forward / backward).
// ---------------------------------------------------------------------------
template <bool UseGpu_, bool Forward_>
struct Backend {
    static constexpr bool use_gpu = UseGpu_;
    static constexpr bool forward = Forward_;
};

template<typename B>
class Node2VecRejectionTest : public ::testing::Test {};

#ifdef HAS_CUDA
using Backends = ::testing::Types<
    Backend<false, true>,    // host  + forward
    Backend<true,  true>,    // device + forward
    Backend<false, false>,   // host  + backward
    Backend<true,  false>    // device + backward
>;
#else
using Backends = ::testing::Types<
    Backend<false, true>,    // host  + forward
    Backend<false, false>    // host  + backward
>;
#endif

TYPED_TEST_SUITE(Node2VecRejectionTest, Backends);

// ===========================================================================
// Test 1 — retry-loop termination under degenerate (p, q).
//
// Picks (p, q) far from 1 so β_max / E[β] is large (acceptance per-attempt
// ≤ ~0.37).  With NODE2VEC_MAX_RETRIES=64, even degenerate workloads must
// either accept on retry (most of the time) or defensively accept the last
// proposal — but never return a sentinel for a non-empty candidate set.
// Runs in both directions to confirm the retry path is wired correctly on
// both the forward (outbound) and backward (inbound, directed) code paths.
// ===========================================================================
TYPED_TEST(Node2VecRejectionTest, TerminatesWithValidEdgeUnderDegeneratePQ) {
    constexpr bool use_gpu = TypeParam::use_gpu;
    constexpr bool forward = TypeParam::forward;

    // p=0.1 → 1/p = 10 (huge return weight), q=10 → 1/q = 0.1 (tiny far
    // weight).  β_max = 10 (the return weight); E[β] ≈ (10 + 1 + 0.1) / 3
    // ≈ 3.7 → per-attempt accept ratio ≈ 0.37, expected ~3 retries.  Far
    // enough from 1 to exercise the retry path repeatedly without hitting
    // the cap.
    auto graph = make_pq_graph<use_gpu, forward>(/*p=*/0.1, /*q=*/10.0);

    constexpr int N_HOPS = 2'000;
    int sentinels = 0;
    int valid     = 0;
    for (int i = 0; i < N_HOPS; ++i) {
        const Edge picked = test_util::get_node_edge_at(
            graph.data(), V, RandomPickerType::TemporalNode2Vec,
            /*timestamp=*/-1, /*prev_node=*/P, forward);
        if (is_sentinel(picked)) {
            ++sentinels;
        } else {
            ++valid;
            ASSERT_EQ(current_of(picked, forward), V);
            const int c = candidate_of(picked, forward);
            ASSERT_TRUE(c == P || c == N || c == F)
                << "Unexpected candidate " << c;
        }
    }
    EXPECT_EQ(sentinels, 0)
        << "Rejection loop must return a valid edge for a non-empty candidate "
           "set, even under degenerate (p, q); got " << sentinels
        << " sentinels in " << N_HOPS << " hops (forward=" << forward << ")";
    EXPECT_EQ(valid, N_HOPS);
}

// ===========================================================================
// Test 2 — distribution-following across a (p, q) sweep.
//
// The base PQBias test only checks one (p, q) cell.  This sweep covers the
// full node2vec literature parameter set, catching β-handling regressions
// at parameter boundaries.  For each (p, q) we compute the analytic per-
// edge probability and check the empirical pick frequencies against it.
// Runs in both directions: the analytic probabilities are identical (β is
// symmetric on the static adjacency CSR), but the underlying weight
// buffer and candidate-vertex extraction differ between forward/backward,
// so distribution agreement in both modes proves both paths are correct.
//
// Per-cell budget: 10 000 samples × 3 buckets ≈ ~3-sigma stdev of 0.005;
// allow ±0.025 (= 5σ for the rarest bucket under the most peaked case
// p=0.25, q=4) so the test is robust.
// ===========================================================================
TYPED_TEST(Node2VecRejectionTest, DistributionMatchesAnalyticAcrossPQValues) {
    constexpr bool use_gpu = TypeParam::use_gpu;
    constexpr bool forward = TypeParam::forward;
    constexpr int  N_SAMPLES = 10'000;
    constexpr double TOLERANCE = 0.025;

    struct PQ { double p, q; };
    const std::vector<PQ> sweep = {
        {1.0,  1.0},   // uniform — every β is 1
        {1.0,  2.0},   // DeepWalk default-ish
        {0.5,  2.0},   // BFS-leaning
        {2.0,  0.5},   // DFS-leaning
        {0.25, 4.0},   // node2vec-extreme corner
    };

    for (const auto& pq : sweep) {
        SCOPED_TRACE(::testing::Message()
            << "(p=" << pq.p << ", q=" << pq.q
            << ", forward=" << forward << ")");
        auto graph = make_pq_graph<use_gpu, forward>(pq.p, pq.q);

        int c_p = 0, c_n = 0, c_f = 0;
        for (int i = 0; i < N_SAMPLES; ++i) {
            const Edge picked = test_util::get_node_edge_at(
                graph.data(), V, RandomPickerType::TemporalNode2Vec,
                /*timestamp=*/-1, /*prev_node=*/P, forward);
            ASSERT_FALSE(is_sentinel(picked));
            ASSERT_EQ(current_of(picked, forward), V);
            const int c = candidate_of(picked, forward);
            if      (c == P) ++c_p;
            else if (c == N) ++c_n;
            else if (c == F) ++c_f;
            else FAIL() << "Unexpected candidate " << c;
        }
        const double frac_p = static_cast<double>(c_p) / N_SAMPLES;
        const double frac_n = static_cast<double>(c_n) / N_SAMPLES;
        const double frac_f = static_cast<double>(c_f) / N_SAMPLES;
        const PQProbs expected = analytic_pq_probs(pq.p, pq.q);

        EXPECT_NEAR(frac_p, expected.p_dest, TOLERANCE)
            << "P fraction off at (p=" << pq.p << ", q=" << pq.q
            << ", forward=" << forward << ")";
        EXPECT_NEAR(frac_n, expected.n_dest, TOLERANCE)
            << "N fraction off at (p=" << pq.p << ", q=" << pq.q
            << ", forward=" << forward << ")";
        EXPECT_NEAR(frac_f, expected.f_dest, TOLERANCE)
            << "F fraction off at (p=" << pq.p << ", q=" << pq.q
            << ", forward=" << forward << ")";
    }
}

}  // namespace
