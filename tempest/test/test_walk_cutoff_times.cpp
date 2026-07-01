// Tests for the per-seed `cutoff_times` walk parameter.
//
// A cutoff is an EXCLUSIVE upper bound on edge time: a seed's walk may only
// traverse edges with timestamp strictly less than its cutoff ("walk this node
// as of time t"). For backward walks this only binds the start edge (later hops
// are already earlier, by monotonicity); for forward walks it caps every hop.
// NO_WALK_CUTOFF (-1) means unbounded.
//
// Three layers are exercised, all paired CPU/GPU via GPU_USAGE_TYPES (the
// CPU/GPU pairing rule in test/CPU_GPU_PAIRING.md), and across both
// directions and all three execution paths (CPU _std, FULL_WALK, NODE_GROUPED
// coop/solo):
//   1. Selector  — deterministic exact-pick checks of the cutoff filter through
//                  test_util::get_node_edge_at (TEST_FIRST / TEST_LAST pickers).
//   2. Walk      — property checks on whole walks (every real edge < cutoff,
//                  heterogeneous per-seed cutoffs, empty/unbounded edge cases).
//   3. Parity    — every kernel (NODE_GROUPED tiers + FULL_WALK) honours the
//                  cutoff identically, including on a high-degree hub seed.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "test_utils.h"
#include "test_temporal_graph_utils.h"
#include "../src/common/const.cuh"
#include "../src/data/enums.cuh"
#include "../src/proxies/Tempest.cuh"

namespace {

constexpr int      MAX_WALK_LEN       = 20;
constexpr int      NUM_WALKS_PER_NODE = 8;
constexpr uint64_t CUTOFF_SEED        = 0xC0FFEE1234ULL;

constexpr int64_t kSentinelMax = std::numeric_limits<int64_t>::max();
constexpr int64_t kSentinelMin = std::numeric_limits<int64_t>::min();

constexpr RandomPickerType kExpWeight = RandomPickerType::ExponentialWeight;

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

// The seed slot carries a direction-dependent sentinel timestamp, never a real
// edge time — skip it in cutoff assertions.
inline bool is_real_edge_ts(const int64_t ts) {
    return ts != kSentinelMax && ts != kSentinelMin && ts != EMPTY_TIMESTAMP_VALUE;
}

// Walk w (rows are seed-major; shuffle_walk_order is off) belongs to seed
// w / num_walks_per_node, hence to that seed's cutoff.
void expect_every_edge_before_cutoff(
    const WalksWithEdgeFeaturesHost& walks,
    const std::vector<int64_t>& seed_cutoffs,
    const int num_walks_per_node) {
    const auto& ws = walks.walk_set;
    const size_t* lens = ws.walk_lens_ptr();
    const int64_t* ts  = ws.timestamps_ptr();
    const size_t max_len = ws.max_len();

    for (size_t w = 0; w < ws.num_walks(); ++w) {
        const int64_t cutoff = seed_cutoffs[w / num_walks_per_node];
        if (cutoff == NO_WALK_CUTOFF) continue;
        for (size_t p = 0; p < lens[w]; ++p) {
            const int64_t t = ts[w * max_len + p];
            if (!is_real_edge_ts(t)) continue;
            EXPECT_LT(t, cutoff)
                << "walk " << w << " (seed-slot " << (w / num_walks_per_node)
                << ") hop " << p << " has edge time >= its cutoff";
        }
    }
}

size_t total_real_hops(const WalksWithEdgeFeaturesHost& walks) {
    const auto& ws = walks.walk_set;
    const size_t* lens = ws.walk_lens_ptr();
    size_t total = 0;
    for (size_t w = 0; w < ws.num_walks(); ++w) total += lens[w];
    return total;
}

void expect_walks_bit_identical(
    const WalksWithEdgeFeaturesHost& a,
    const WalksWithEdgeFeaturesHost& b) {
    const auto& wa = a.walk_set;
    const auto& wb = b.walk_set;
    ASSERT_EQ(wa.num_walks(), wb.num_walks());
    ASSERT_EQ(wa.max_len(),   wb.max_len());

    const size_t*  la = wa.walk_lens_ptr();
    const size_t*  lb = wb.walk_lens_ptr();
    const int*     na = wa.nodes_ptr();
    const int*     nb = wb.nodes_ptr();
    const int64_t* ta = wa.timestamps_ptr();
    const int64_t* tb = wb.timestamps_ptr();
    const size_t   max_len = wa.max_len();

    for (size_t w = 0; w < wa.num_walks(); ++w) {
        ASSERT_EQ(la[w], lb[w]) << "walk " << w << " length differs";
        for (size_t p = 0; p < la[w]; ++p) {
            const size_t idx = w * max_len + p;
            EXPECT_EQ(na[idx], nb[idx]) << "walk " << w << " node@" << p;
            EXPECT_EQ(ta[idx], tb[idx]) << "walk " << w << " ts@"   << p;
        }
    }
}

std::unique_ptr<Tempest> make_trw(const bool use_gpu, const bool is_directed) {
    return std::make_unique<Tempest>(
        is_directed,
        use_gpu,
        /*max_time_capacity=*/-1,
        /*enable_weight_computation=*/true,
        /*enable_temporal_node2vec=*/false,
        DEFAULT_TIMESCALE_BOUND,
        DEFAULT_NODE2VEC_P,
        DEFAULT_NODE2VEC_Q,
        EMPTY_NODE_VALUE,
        /*global_seed=*/CUTOFF_SEED,
        /*shuffle_walk_order=*/false);
}

// =====================================================================
// Layer 1 — selector level (deterministic cutoff filter)
// =====================================================================

template<typename T>
class CutoffSelectorTest : public ::testing::Test {
protected:
    std::unique_ptr<core::Tempest> graph;
    void SetUp() override {
        graph = std::make_unique<core::Tempest>(/*is_directed=*/true, T::value);
    }
    TemporalGraphData& data() { return graph->data(); }

    static void expect_edge(const Edge& e, int u, int i, int64_t ts) {
        EXPECT_EQ(e.u, u);
        EXPECT_EQ(e.i, i);
        EXPECT_EQ(e.ts, ts);
    }
    static void expect_invalid(const Edge& e) {
        EXPECT_EQ(e.u, -1);
        EXPECT_EQ(e.i, -1);
        EXPECT_EQ(e.ts, -1);
    }
};

TYPED_TEST_SUITE(CutoffSelectorTest, GPU_USAGE_TYPES);

// Forward: cutoff is an upper bound on the picked edge (ts < cutoff), on top of
// the existing lower bound from `timestamp` (ts > timestamp).
TYPED_TEST(CutoffSelectorTest, ForwardCutoffUpperBound) {
    test_util::add_edges(*this->graph, {
        Edge{10, 20, 100}, Edge{10, 30, 101}, Edge{10, 40, 102},
        Edge{10, 50, 103}, Edge{10, 60, 104},
    });
    const auto& d = this->data();
    using RP = RandomPickerType;
    constexpr bool FWD = true;

    // No cutoff: full range.
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_FIRST, -1, -1, FWD, NO_WALK_CUTOFF), 10, 20, 100);
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_LAST,  -1, -1, FWD, NO_WALK_CUTOFF), 10, 60, 104);

    // Cutoff trims the upper end (exclusive): ts < 104 ⇒ latest valid is 103.
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_LAST,  -1, -1, FWD, /*cutoff=*/104), 10, 50, 103);
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_LAST,  -1, -1, FWD, /*cutoff=*/103), 10, 40, 102);
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_FIRST, -1, -1, FWD, /*cutoff=*/101), 10, 20, 100);

    // Cutoff at/below the earliest edge excludes everything.
    this->expect_invalid(test_util::get_node_edge_at(d, 10, RP::TEST_FIRST, -1, -1, FWD, /*cutoff=*/100));
    this->expect_invalid(test_util::get_node_edge_at(d, 10, RP::TEST_FIRST, -1, -1, FWD, /*cutoff=*/50));

    // Cutoff above the latest edge is a no-op (same as unbounded).
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_LAST,  -1, -1, FWD, /*cutoff=*/1000), 10, 60, 104);
}

// Forward: cutoff (upper) composes with the monotonic lower bound `timestamp`.
TYPED_TEST(CutoffSelectorTest, ForwardCutoffWithTimestampWindow) {
    test_util::add_edges(*this->graph, {
        Edge{10, 20, 100}, Edge{10, 30, 101}, Edge{10, 40, 102},
        Edge{10, 50, 103}, Edge{10, 60, 104},
    });
    const auto& d = this->data();
    using RP = RandomPickerType;
    constexpr bool FWD = true;

    // ts > 101 AND ts < 104  ⇒ {102, 103}.
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_FIRST, 101, -1, FWD, /*cutoff=*/104), 10, 40, 102);
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_LAST,  101, -1, FWD, /*cutoff=*/104), 10, 50, 103);
    // Window collapses to a single edge.
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_FIRST, 101, -1, FWD, /*cutoff=*/103), 10, 40, 102);
    // Empty window (lower >= upper-1).
    this->expect_invalid(test_util::get_node_edge_at(d, 10, RP::TEST_FIRST, 102, -1, FWD, /*cutoff=*/103));
}

// Backward: cutoff is an upper bound (ts < cutoff) that combines with the
// existing backward upper bound from `timestamp` via min().
TYPED_TEST(CutoffSelectorTest, BackwardCutoffUpperBound) {
    test_util::add_edges(*this->graph, {
        Edge{20, 10, 100}, Edge{30, 10, 101}, Edge{40, 10, 102},
        Edge{50, 10, 103}, Edge{60, 10, 104},
    });
    const auto& d = this->data();
    using RP = RandomPickerType;
    constexpr bool BWD = false;

    // No cutoff.
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_FIRST, -1, -1, BWD, NO_WALK_CUTOFF), 20, 10, 100);
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_LAST,  -1, -1, BWD, NO_WALK_CUTOFF), 60, 10, 104);

    // Cutoff trims the latest predecessors (exclusive).
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_LAST,  -1, -1, BWD, /*cutoff=*/104), 50, 10, 103);
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_LAST,  -1, -1, BWD, /*cutoff=*/102), 30, 10, 101);
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_FIRST, -1, -1, BWD, /*cutoff=*/101), 20, 10, 100);

    // Exclude all / no-op above max.
    this->expect_invalid(test_util::get_node_edge_at(d, 10, RP::TEST_FIRST, -1, -1, BWD, /*cutoff=*/100));
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_LAST,  -1, -1, BWD, /*cutoff=*/1000), 60, 10, 104);

    // min(timestamp, cutoff): both upper bounds, the tighter wins.
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_LAST,  103, -1, BWD, /*cutoff=*/102), 30, 10, 101);
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_LAST,  102, -1, BWD, /*cutoff=*/103), 30, 10, 101);
}

// The cutoff is exclusive: an edge exactly at the cutoff is dropped.
TYPED_TEST(CutoffSelectorTest, CutoffBoundaryIsExclusive) {
    test_util::add_edges(*this->graph, {
        Edge{10, 20, 100}, Edge{10, 30, 110}, Edge{10, 40, 120},  // outbound
        Edge{50, 70, 100}, Edge{60, 70, 110}, Edge{80, 70, 120},  // inbound to 70
    });
    const auto& d = this->data();
    using RP = RandomPickerType;

    // Forward, cutoff == 110 ⇒ only ts 100 survives.
    this->expect_edge(test_util::get_node_edge_at(d, 10, RP::TEST_LAST, -1, -1, true,  /*cutoff=*/110), 10, 20, 100);
    // Backward, cutoff == 110 ⇒ only ts 100 survives.
    this->expect_edge(test_util::get_node_edge_at(d, 70, RP::TEST_LAST, -1, -1, false, /*cutoff=*/110), 50, 70, 100);
}

// =====================================================================
// Layer 2 — whole-walk properties
// =====================================================================

template<typename T>
class CutoffWalkTest : public ::testing::Test {
protected:
    void SetUp() override {
        sample_edges = read_edges_from_csv(sample_data_path());
        int64_t lo = std::numeric_limits<int64_t>::max();
        int64_t hi = std::numeric_limits<int64_t>::min();
        for (const auto& [u, v, ts] : sample_edges) {
            lo = std::min(lo, ts);
            hi = std::max(hi, ts);
        }
        ts_min = lo;
        ts_max = hi;
    }

    std::unique_ptr<Tempest> trw(const bool is_directed) const {
        auto t = make_trw(T::value, is_directed);
        t->add_multiple_edges(sample_edges);
        return t;
    }

    WalksWithEdgeFeaturesHost run(
        Tempest& t,
        const std::vector<int>& seeds,
        const std::vector<int64_t>& cutoffs,
        const WalkDirection dir,
        const RandomPickerType picker,
        const KernelLaunchType klt,
        const int64_t* cutoff_ptr_override = nullptr) const {
        const int64_t* cptr = cutoff_ptr_override
            ? cutoff_ptr_override
            : (cutoffs.empty() ? nullptr : cutoffs.data());
        return t.get_random_walks_and_times_for_nodes(
            seeds.data(), seeds.size(), cptr,
            MAX_WALK_LEN, &picker, NUM_WALKS_PER_NODE,
            /*initial_edge_bias=*/nullptr, dir, klt);
    }

    std::vector<std::tuple<int, int, int64_t>> sample_edges;
    int64_t ts_min = 0;
    int64_t ts_max = 0;
};

TYPED_TEST_SUITE(CutoffWalkTest, GPU_USAGE_TYPES);

TYPED_TEST(CutoffWalkTest, BackwardEveryEdgeBeforeCutoff) {
    auto t = this->trw(/*is_directed=*/true);
    const std::vector<int> seeds = {109, 41, 15, 7, 88};
    const int64_t mid = this->ts_min + (this->ts_max - this->ts_min) / 2;
    const std::vector<int64_t> cutoffs(seeds.size(), mid);

    for (const auto klt : {KernelLaunchType::NODE_GROUPED, KernelLaunchType::FULL_WALK}) {
        const auto walks = this->run(*t, seeds, cutoffs,
            WalkDirection::Backward_In_Time, RandomPickerType::ExponentialIndex, klt);
        expect_every_edge_before_cutoff(walks, cutoffs, NUM_WALKS_PER_NODE);
        EXPECT_GT(total_real_hops(walks), seeds.size())
            << "cutoff at the temporal midpoint should still admit many walks";
    }
}

TYPED_TEST(CutoffWalkTest, ForwardEveryEdgeBeforeCutoff) {
    auto t = this->trw(/*is_directed=*/true);
    const std::vector<int> seeds = {109, 41, 15, 7, 88};
    const int64_t mid = this->ts_min + (this->ts_max - this->ts_min) / 2;
    const std::vector<int64_t> cutoffs(seeds.size(), mid);

    for (const auto klt : {KernelLaunchType::NODE_GROUPED, KernelLaunchType::FULL_WALK}) {
        const auto walks = this->run(*t, seeds, cutoffs,
            WalkDirection::Forward_In_Time, RandomPickerType::ExponentialIndex, klt);
        // Forward caps EVERY hop below the cutoff.
        expect_every_edge_before_cutoff(walks, cutoffs, NUM_WALKS_PER_NODE);
    }
}

TYPED_TEST(CutoffWalkTest, HeterogeneousPerSeedCutoffs) {
    auto t = this->trw(/*is_directed=*/true);
    const std::vector<int> seeds = {109, 109, 109, 41, 15};
    // Each (possibly repeated) seed position gets its own cutoff, including one
    // unbounded position to confirm independence.
    const int64_t q1 = this->ts_min + (this->ts_max - this->ts_min) / 4;
    const int64_t q2 = this->ts_min + (this->ts_max - this->ts_min) / 2;
    const int64_t q3 = this->ts_min + 3 * (this->ts_max - this->ts_min) / 4;
    const std::vector<int64_t> cutoffs = {q1, q2, NO_WALK_CUTOFF, q3, q2};

    for (const auto klt : {KernelLaunchType::NODE_GROUPED, KernelLaunchType::FULL_WALK}) {
        const auto walks = this->run(*t, seeds, cutoffs,
            WalkDirection::Backward_In_Time, RandomPickerType::ExponentialWeight, klt);
        expect_every_edge_before_cutoff(walks, cutoffs, NUM_WALKS_PER_NODE);
    }
}

TYPED_TEST(CutoffWalkTest, CutoffBelowAllProducesEmptyWalks) {
    auto t = this->trw(/*is_directed=*/true);
    const std::vector<int> seeds = {109, 41, 15};
    // Exclusive bound at the global minimum ⇒ no edge qualifies for any seed.
    const std::vector<int64_t> cutoffs(seeds.size(), this->ts_min);

    for (const auto klt : {KernelLaunchType::NODE_GROUPED, KernelLaunchType::FULL_WALK}) {
        const auto walks = this->run(*t, seeds, cutoffs,
            WalkDirection::Backward_In_Time, RandomPickerType::Uniform, klt);
        EXPECT_EQ(total_real_hops(walks), 0u)
            << "no walk should produce a hop when the cutoff is at/below ts_min";
    }
}

TYPED_TEST(CutoffWalkTest, CutoffAboveAllEqualsUnbounded) {
    auto t = this->trw(/*is_directed=*/true);
    const std::vector<int> seeds = {109, 41, 15, 7};
    const std::vector<int64_t> cutoffs(seeds.size(), this->ts_max + 1);

    // A cutoff above every edge excludes nothing. The selector tests already
    // prove "cutoff above max is a no-op" deterministically on both backends;
    // here we check it end-to-end. The GPU path (Philox seeded by global_seed)
    // is reproducible across calls, so the bounded-above walks must match the
    // unbounded walks bit-for-bit. The CPU rand source is seeded per call
    // (non-deterministic), so there we only assert the run stays productive.
    for (const auto dir : {WalkDirection::Backward_In_Time, WalkDirection::Forward_In_Time}) {
        for (const auto klt : {KernelLaunchType::NODE_GROUPED, KernelLaunchType::FULL_WALK}) {
            const auto bounded = this->run(*t, seeds, cutoffs, dir, RandomPickerType::ExponentialIndex, klt);
            if constexpr (TypeParam::value) {
                const auto unbounded = this->run(*t, seeds, {}, dir, RandomPickerType::ExponentialIndex, klt);
                expect_walks_bit_identical(bounded, unbounded);
            } else {
                EXPECT_GT(total_real_hops(bounded), seeds.size());
            }
        }
    }
}

TYPED_TEST(CutoffWalkTest, UndirectedBackwardRespectsCutoff) {
    auto t = this->trw(/*is_directed=*/false);
    const std::vector<int> seeds = {109, 41, 15};
    const int64_t mid = this->ts_min + (this->ts_max - this->ts_min) / 2;
    const std::vector<int64_t> cutoffs(seeds.size(), mid);

    for (const auto klt : {KernelLaunchType::NODE_GROUPED, KernelLaunchType::FULL_WALK}) {
        const auto walks = this->run(*t, seeds, cutoffs,
            WalkDirection::Backward_In_Time, RandomPickerType::ExponentialWeight, klt);
        expect_every_edge_before_cutoff(walks, cutoffs, NUM_WALKS_PER_NODE);
    }
}

// =====================================================================
// Layer 3 — cross-kernel parity (all kernels honour the cutoff)
// =====================================================================

template<typename T>
class CutoffParityTest : public ::testing::Test {
protected:
    void SetUp() override {
        sample_edges = read_edges_from_csv(sample_data_path());
        for (const auto& [u, v, ts] : sample_edges) ts_max = std::max(ts_max, ts);
    }
    std::unique_ptr<Tempest> trw() const {
        auto t = make_trw(T::value, /*is_directed=*/true);
        t->add_multiple_edges(sample_edges);
        return t;
    }
    std::vector<std::tuple<int, int, int64_t>> sample_edges;
    int64_t ts_max = 0;
};

TYPED_TEST_SUITE(CutoffParityTest, GPU_USAGE_TYPES);

// The hub seed (highest degree) forces the NODE_GROUPED coop block/warp tiers;
// both they and FULL_WALK must honour the cutoff at step 0 and beyond.
TYPED_TEST(CutoffParityTest, AllKernelsHonourCutoffOnHubSeed) {
    auto t = this->trw();
    const std::vector<int> seeds = {109};  // 1346 edges → coop tiers
    const int64_t cutoff = this->ts_max / 2;
    const std::vector<int64_t> cutoffs(seeds.size(), cutoff);

    for (const auto dir : {WalkDirection::Backward_In_Time, WalkDirection::Forward_In_Time}) {
        for (const auto klt : {KernelLaunchType::NODE_GROUPED,
                               KernelLaunchType::NODE_GROUPED_GLOBAL_ONLY,
                               KernelLaunchType::FULL_WALK}) {
            const auto walks = t->get_random_walks_and_times_for_nodes(
                seeds.data(), seeds.size(), cutoffs.data(),
                MAX_WALK_LEN, /*walk_bias=*/&kExpWeight, NUM_WALKS_PER_NODE,
                /*initial_edge_bias=*/nullptr, dir, klt);
            expect_every_edge_before_cutoff(walks, cutoffs, NUM_WALKS_PER_NODE);
        }
    }
}

// With a cutoff applied, NODE_GROUPED and FULL_WALK must still produce
// structurally comparable walk populations (same seed, both cutoff-bounded) —
// not a jammed/empty population on one path only.
TYPED_TEST(CutoffParityTest, NodeGroupedAndFullWalkBothProductiveUnderCutoff) {
    auto t = this->trw();
    const std::vector<int> seeds = {109, 41, 15};
    const int64_t cutoff = (3 * this->ts_max) / 4;
    const std::vector<int64_t> cutoffs(seeds.size(), cutoff);

    const auto grouped = t->get_random_walks_and_times_for_nodes(
        seeds.data(), seeds.size(), cutoffs.data(), MAX_WALK_LEN,
        &kExpWeight, NUM_WALKS_PER_NODE, nullptr,
        WalkDirection::Backward_In_Time, KernelLaunchType::NODE_GROUPED);
    const auto full = t->get_random_walks_and_times_for_nodes(
        seeds.data(), seeds.size(), cutoffs.data(), MAX_WALK_LEN,
        &kExpWeight, NUM_WALKS_PER_NODE, nullptr,
        WalkDirection::Backward_In_Time, KernelLaunchType::FULL_WALK);

    expect_every_edge_before_cutoff(grouped, cutoffs, NUM_WALKS_PER_NODE);
    expect_every_edge_before_cutoff(full,    cutoffs, NUM_WALKS_PER_NODE);

    const size_t hops_grouped = total_real_hops(grouped);
    const size_t hops_full    = total_real_hops(full);
    EXPECT_GT(hops_grouped, seeds.size());
    EXPECT_GT(hops_full,    seeds.size());

    // Mean walk length within a wide band (Philox streams differ across paths).
    const double mean_grouped = static_cast<double>(hops_grouped) / grouped.walk_set.num_walks();
    const double mean_full    = static_cast<double>(hops_full)    / full.walk_set.num_walks();
    EXPECT_GT(mean_grouped, 0.5 * mean_full);
    EXPECT_LT(mean_grouped, 2.0 * mean_full);
}

} // namespace
