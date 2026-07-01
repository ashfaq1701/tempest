#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "test_utils.h"
#include "../src/common/const.cuh"
#include "../src/data/enums.cuh"
#include "../src/proxies/Tempest.cuh"

// Structural parity between FULL_WALK and NODE_GROUPED — not bit-exact
// (different Philox stepping). Asserts walk count, slot-0 mapping,
// per-hop edge validity, and length-distribution proximity.

namespace {

constexpr int      MAX_WALK_LEN       = 20;
constexpr int      NUM_WALKS_PER_NODE = 5;
constexpr uint64_t PARITY_SEED        = 0x5EED1234ULL;

// pickers that route through the cooperative pipeline (Node2Vec excluded).
inline const std::initializer_list<RandomPickerType> kParityPickers = {
    RandomPickerType::Linear,
    RandomPickerType::Uniform,
    RandomPickerType::ExponentialIndex,
    RandomPickerType::ExponentialWeight,
};

inline const char* picker_name(const RandomPickerType p) {
    switch (p) {
        case RandomPickerType::Linear:            return "Linear";
        case RandomPickerType::Uniform:           return "Uniform";
        case RandomPickerType::ExponentialIndex:  return "ExponentialIndex";
        case RandomPickerType::ExponentialWeight: return "ExponentialWeight";
        case RandomPickerType::TemporalNode2Vec:  return "TemporalNode2Vec";
        default:                                  return "?";
    }
}

inline bool picker_needs_weights(const RandomPickerType p) {
    return p == RandomPickerType::ExponentialWeight ||
           p == RandomPickerType::TemporalNode2Vec;
}

// wide band because Philox streams diverge; tight enough to catch jammed walks.
constexpr double MEAN_LEN_RATIO_LOW  = 0.5;
constexpr double MEAN_LEN_RATIO_HIGH = 2.0;

#ifdef HAS_CUDA
using PARITY_BACKENDS = ::testing::Types<
    std::integral_constant<bool, false>,
    std::integral_constant<bool, true>
>;
#else
using PARITY_BACKENDS = ::testing::Types<
    std::integral_constant<bool, false>
>;
#endif

template <typename T>
class NodeGroupedParityTest : public ::testing::Test {
protected:
    void SetUp() override {
        sample_edges = read_edges_from_csv(sample_data_path());
        for (const auto& [u, v, ts] : sample_edges) {
            edge_set.insert(pack_edge(u, v, ts));
        }
    }

    // shuffle_walk_order=false so slot 0 maps the same across both paths.
    std::unique_ptr<Tempest> make_trw(const bool enable_weights) const {
        auto trw = std::make_unique<Tempest>(
            /*is_directed=*/true,
            /*use_gpu=*/T::value,
            /*max_time_capacity=*/-1,
            /*enable_weight_computation=*/enable_weights,
            /*enable_temporal_node2vec=*/false,
            DEFAULT_TIMESCALE_BOUND,
            DEFAULT_NODE2VEC_P,
            DEFAULT_NODE2VEC_Q,
            EMPTY_NODE_VALUE,
            /*global_seed=*/PARITY_SEED,
            /*shuffle_walk_order=*/false);
        trw->add_multiple_edges(sample_edges);
        return trw;
    }

    static uint64_t pack_edge(int u, int v, int64_t ts) {
        uint64_t k = static_cast<uint64_t>(static_cast<uint32_t>(u));
        k = (k * 1099511628211ULL) ^ static_cast<uint32_t>(v);
        k = (k * 1099511628211ULL) ^ static_cast<uint64_t>(ts);
        return k;
    }

    std::vector<std::tuple<int, int, int64_t>> sample_edges;
    std::unordered_set<uint64_t> edge_set;
};

TYPED_TEST_SUITE(NodeGroupedParityTest, PARITY_BACKENDS);

struct WalkDigest {
    size_t num_walks       = 0;
    size_t produced_walks  = 0;
    size_t total_hops      = 0;
    size_t walks_ge_3      = 0;
    double mean_len_all    = 0.0;
};

WalkDigest digest_walks(const WalksWithEdgeFeaturesHost& walks) {
    WalkDigest d{};
    const size_t* lens = walks.walk_set.walk_lens_ptr();
    d.num_walks = walks.walk_set.num_walks();
    for (size_t w = 0; w < d.num_walks; ++w) {
        if (lens[w] == 0) continue;
        ++d.produced_walks;
        d.total_hops += lens[w];
        if (lens[w] >= 3) ++d.walks_ge_3;
    }
    if (d.produced_walks > 0) {
        d.mean_len_all =
            static_cast<double>(d.total_hops) /
            static_cast<double>(d.produced_walks);
    }
    return d;
}

// skip walks where only one side dead-ended (asymmetric padding).
void assert_slot0_agreement(
    const WalksWithEdgeFeaturesHost& a,
    const WalksWithEdgeFeaturesHost& b,
    const size_t max_walk_len,
    size_t* compared_out) {
    ASSERT_EQ(a.walk_set.num_walks(), b.walk_set.num_walks());
    const size_t num_walks = a.walk_set.num_walks();
    const int* an = a.walk_set.nodes_ptr();
    const int* bn = b.walk_set.nodes_ptr();
    const size_t* al = a.walk_set.walk_lens_ptr();
    const size_t* bl = b.walk_set.walk_lens_ptr();

    size_t compared = 0;
    for (size_t w = 0; w < num_walks; ++w) {
        if (al[w] == 0 || bl[w] == 0) continue;
        ASSERT_EQ(an[w * max_walk_len], bn[w * max_walk_len])
            << "Walk " << w << " slot 0 differs across paths.";
        ++compared;
    }
    *compared_out = compared;
}

// caller-visible walks are always chronologically ordered post-reverse,
// so cur >= prev holds for both forward and backward.
template <bool Forward>
void assert_walks_are_valid(
    const WalksWithEdgeFeaturesHost& walks,
    const size_t max_walk_len) {
    const int*     nodes = walks.walk_set.nodes_ptr();
    const int64_t* ts    = walks.walk_set.timestamps_ptr();
    const size_t*  lens  = walks.walk_set.walk_lens_ptr();
    const size_t num_walks = walks.walk_set.num_walks();

    for (size_t w = 0; w < num_walks; ++w) {
        const size_t wl = lens[w];
        if (wl == 0) continue;

        const size_t base = w * max_walk_len;
        for (size_t i = 0; i < wl; ++i) {
            ASSERT_NE(nodes[base + i], EMPTY_NODE_VALUE)
                << "Walk " << w << " hop " << i << " is -1 sentinel";
        }
        for (size_t i = 1; i < wl; ++i) {
            const int64_t prev = ts[base + i - 1];
            const int64_t cur  = ts[base + i];
            ASSERT_GE(cur, prev)
                << "Walk " << w
                << (Forward ? " forward" : " backward")
                << " ts non-monotone between hop " << (i - 1)
                << " (ts=" << prev << ") and hop " << i
                << " (ts=" << cur << ")";
        }
    }
}

}  // namespace

TYPED_TEST(NodeGroupedParityTest, AllNodesConstrained_Forward_StructuralParity) {
    for (const auto picker : kParityPickers) {
        SCOPED_TRACE(testing::Message() << "picker=" << picker_name(picker));
        const bool enable_weights = picker_needs_weights(picker);
        auto trw_full = this->make_trw(enable_weights);
        auto trw_grp  = this->make_trw(enable_weights);

        const auto walks_full = trw_full->get_random_walks_and_times_for_all_nodes(
            MAX_WALK_LEN, &picker, NUM_WALKS_PER_NODE,
            /*initial_edge_bias=*/nullptr,
            WalkDirection::Forward_In_Time,
            KernelLaunchType::FULL_WALK);
        const auto walks_grp = trw_grp->get_random_walks_and_times_for_all_nodes(
            MAX_WALK_LEN, &picker, NUM_WALKS_PER_NODE,
            /*initial_edge_bias=*/nullptr,
            WalkDirection::Forward_In_Time,
            KernelLaunchType::NODE_GROUPED);

        ASSERT_EQ(walks_full.walk_set.num_walks(), walks_grp.walk_set.num_walks());

        size_t compared = 0;
        ASSERT_NO_FATAL_FAILURE(
            assert_slot0_agreement(walks_full, walks_grp,
                                   static_cast<size_t>(MAX_WALK_LEN),
                                   &compared));
        ASSERT_GT(compared, 0u)
            << "No walks were compared — both paths dead-ended on every seed.";

        ASSERT_NO_FATAL_FAILURE(
            assert_walks_are_valid<true>(walks_full,
                                         static_cast<size_t>(MAX_WALK_LEN)));
        ASSERT_NO_FATAL_FAILURE(
            assert_walks_are_valid<true>(walks_grp,
                                         static_cast<size_t>(MAX_WALK_LEN)));

        const auto d_full = digest_walks(walks_full);
        const auto d_grp  = digest_walks(walks_grp);
        ASSERT_GT(d_full.produced_walks, 0u);
        ASSERT_GT(d_grp.produced_walks,  0u);
        ASSERT_GT(d_full.mean_len_all, 1.0);
        ASSERT_GT(d_grp.mean_len_all, 1.0);

        const double ratio = d_grp.mean_len_all / d_full.mean_len_all;
        EXPECT_GT(ratio, MEAN_LEN_RATIO_LOW);
        EXPECT_LT(ratio, MEAN_LEN_RATIO_HIGH);
    }
}

// unconstrained start (start_node_id == -1); slot-0 may legitimately differ.
TYPED_TEST(NodeGroupedParityTest, Unconstrained_Forward_StructuralParity) {
    for (const auto picker : kParityPickers) {
        SCOPED_TRACE(testing::Message() << "picker=" << picker_name(picker));
        const bool enable_weights = picker_needs_weights(picker);
        auto trw_full = this->make_trw(enable_weights);
        auto trw_grp  = this->make_trw(enable_weights);

        constexpr int NUM_WALKS_TOTAL = 400;
        const auto walks_full = trw_full->get_random_walks_and_times(
            MAX_WALK_LEN, &picker, NUM_WALKS_TOTAL,
            /*initial_edge_bias=*/nullptr,
            WalkDirection::Forward_In_Time,
            KernelLaunchType::FULL_WALK);
        const auto walks_grp = trw_grp->get_random_walks_and_times(
            MAX_WALK_LEN, &picker, NUM_WALKS_TOTAL,
            /*initial_edge_bias=*/nullptr,
            WalkDirection::Forward_In_Time,
            KernelLaunchType::NODE_GROUPED);

        ASSERT_EQ(walks_full.walk_set.num_walks(), walks_grp.walk_set.num_walks());

        ASSERT_NO_FATAL_FAILURE(
            assert_walks_are_valid<true>(walks_full,
                                         static_cast<size_t>(MAX_WALK_LEN)));
        ASSERT_NO_FATAL_FAILURE(
            assert_walks_are_valid<true>(walks_grp,
                                         static_cast<size_t>(MAX_WALK_LEN)));

        const auto d_full = digest_walks(walks_full);
        const auto d_grp  = digest_walks(walks_grp);
        ASSERT_GT(d_full.produced_walks, 0u);
        ASSERT_GT(d_grp.produced_walks,  0u);

        const double ratio = d_grp.mean_len_all / d_full.mean_len_all;
        EXPECT_GT(ratio, MEAN_LEN_RATIO_LOW);
        EXPECT_LT(ratio, MEAN_LEN_RATIO_HIGH);
    }
}

TYPED_TEST(NodeGroupedParityTest, AllNodesConstrained_Backward_StructuralParity) {
    for (const auto picker : kParityPickers) {
        SCOPED_TRACE(testing::Message() << "picker=" << picker_name(picker));
        const bool enable_weights = picker_needs_weights(picker);
        auto trw_full = this->make_trw(enable_weights);
        auto trw_grp  = this->make_trw(enable_weights);

        const auto walks_full = trw_full->get_random_walks_and_times_for_all_nodes(
            MAX_WALK_LEN, &picker, NUM_WALKS_PER_NODE,
            /*initial_edge_bias=*/nullptr,
            WalkDirection::Backward_In_Time,
            KernelLaunchType::FULL_WALK);
        const auto walks_grp = trw_grp->get_random_walks_and_times_for_all_nodes(
            MAX_WALK_LEN, &picker, NUM_WALKS_PER_NODE,
            /*initial_edge_bias=*/nullptr,
            WalkDirection::Backward_In_Time,
            KernelLaunchType::NODE_GROUPED);

        ASSERT_EQ(walks_full.walk_set.num_walks(), walks_grp.walk_set.num_walks());

        ASSERT_NO_FATAL_FAILURE(
            assert_walks_are_valid<false>(walks_full,
                                          static_cast<size_t>(MAX_WALK_LEN)));
        ASSERT_NO_FATAL_FAILURE(
            assert_walks_are_valid<false>(walks_grp,
                                          static_cast<size_t>(MAX_WALK_LEN)));

        const auto d_full = digest_walks(walks_full);
        const auto d_grp  = digest_walks(walks_grp);
        ASSERT_GT(d_full.produced_walks, 0u);
        ASSERT_GT(d_grp.produced_walks,  0u);
        const double ratio = d_grp.mean_len_all / d_full.mean_len_all;
        EXPECT_GT(ratio, MEAN_LEN_RATIO_LOW);
        EXPECT_LT(ratio, MEAN_LEN_RATIO_HIGH);
    }
}
