#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "../src/common/const.cuh"
#include "../src/data/enums.cuh"
#include "../src/proxies/Tempest.cuh"

namespace {

constexpr int      MAX_WALK_LEN      = 10;
constexpr uint64_t TIER_SEED         = 0xC0FFEEULL;
constexpr int      HUB_NODE          = 100000;
constexpr int      DEST_NODE_BASE    = 200000;

constexpr RandomPickerType LINEAR_PICKER    = RandomPickerType::Linear;
constexpr RandomPickerType NODE2VEC_PICKER  = RandomPickerType::TemporalNode2Vec;

#ifdef HAS_CUDA
using TIER_BACKENDS = ::testing::Types<
    std::integral_constant<bool, false>,
    std::integral_constant<bool, true>
>;
#else
using TIER_BACKENDS = ::testing::Types<
    std::integral_constant<bool, false>
>;
#endif

template <typename T>
class NodeGroupedTierRoutingTest : public ::testing::Test {
protected:
    // num_sources sources -> HUB at ts=0; HUB -> hub_groups dests at ts=1..G.
    static std::vector<std::tuple<int, int, int64_t>>
    convergence_graph(const int num_sources, const int hub_groups) {
        std::vector<std::tuple<int, int, int64_t>> edges;
        edges.reserve(static_cast<size_t>(num_sources + hub_groups));
        for (int s = 0; s < num_sources; ++s) {
            edges.emplace_back(s, HUB_NODE, int64_t{0});
        }
        for (int g = 0; g < hub_groups; ++g) {
            edges.emplace_back(HUB_NODE, DEST_NODE_BASE + g,
                               static_cast<int64_t>(g + 1));
        }
        return edges;
    }

    std::unique_ptr<Tempest> make_trw(
        const bool enable_weights,
        const bool enable_node2vec = false) const {
        return std::make_unique<Tempest>(
            /*is_directed=*/true,
            /*use_gpu=*/T::value,
            /*max_time_capacity=*/-1,
            /*enable_weight_computation=*/enable_weights,
            /*enable_temporal_node2vec=*/enable_node2vec,
            DEFAULT_TIMESCALE_BOUND,
            DEFAULT_NODE2VEC_P,
            DEFAULT_NODE2VEC_Q,
            EMPTY_NODE_VALUE,
            /*global_seed=*/TIER_SEED,
            /*shuffle_walk_order=*/false);
    }
};

TYPED_TEST_SUITE(NodeGroupedTierRoutingTest, TIER_BACKENDS);

void assert_all_walks_valid(const WalksWithEdgeFeaturesHost& walks,
                            const size_t max_walk_len) {
    const int*     nodes = walks.walk_set.nodes_ptr();
    const int64_t* ts    = walks.walk_set.timestamps_ptr();
    const size_t*  lens  = walks.walk_set.walk_lens_ptr();
    const size_t num_walks = walks.walk_set.num_walks();

    size_t produced = 0;
    for (size_t w = 0; w < num_walks; ++w) {
        const size_t wl = lens[w];
        if (wl == 0) continue;
        ++produced;
        const size_t base = w * max_walk_len;
        for (size_t i = 0; i < wl; ++i) {
            ASSERT_NE(nodes[base + i], EMPTY_NODE_VALUE)
                << "walk " << w << " hop " << i << " is the -1 sentinel";
        }
        for (size_t i = 1; i < wl; ++i) {
            ASSERT_GE(ts[base + i], ts[base + i - 1])
                << "walk " << w << " ts non-monotone between hop "
                << (i - 1) << " (ts=" << ts[base + i - 1]
                << ") and hop " << i << " (ts=" << ts[base + i] << ")";
        }
    }
    ASSERT_GT(produced, 0u) << "no walks produced — kernel almost certainly broke.";
}

}  // namespace

// W=10, G=20 -> warp_smem.
TYPED_TEST(NodeGroupedTierRoutingTest, WarpSmem_WalksAreValid) {
    auto trw = this->make_trw(/*enable_weights=*/false);
    trw->add_multiple_edges(this->convergence_graph(
        /*num_sources=*/10, /*hub_groups=*/20));

    const auto walks = trw->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &LINEAR_PICKER, /*num_walks_per_node=*/1,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::NODE_GROUPED);

    ASSERT_NO_FATAL_FAILURE(
        assert_all_walks_valid(walks, static_cast<size_t>(MAX_WALK_LEN)));
}

// W=10, G=400 -> warp_global.
TYPED_TEST(NodeGroupedTierRoutingTest, WarpGlobal_WalksAreValid) {
    auto trw = this->make_trw(/*enable_weights=*/false);
    trw->add_multiple_edges(this->convergence_graph(
        /*num_sources=*/10, /*hub_groups=*/400));

    const auto walks = trw->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &LINEAR_PICKER, /*num_walks_per_node=*/1,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::NODE_GROUPED);

    ASSERT_NO_FATAL_FAILURE(
        assert_all_walks_valid(walks, static_cast<size_t>(MAX_WALK_LEN)));
}

// W=300, G=100 -> block_smem.
TYPED_TEST(NodeGroupedTierRoutingTest, BlockSmem_WalksAreValid) {
    auto trw = this->make_trw(/*enable_weights=*/false);
    trw->add_multiple_edges(this->convergence_graph(
        /*num_sources=*/300, /*hub_groups=*/100));

    const auto walks = trw->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &LINEAR_PICKER, /*num_walks_per_node=*/1,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::NODE_GROUPED);

    ASSERT_NO_FATAL_FAILURE(
        assert_all_walks_valid(walks, static_cast<size_t>(MAX_WALK_LEN)));
}

// W=300, G=3000 -> block_global.
TYPED_TEST(NodeGroupedTierRoutingTest, BlockGlobal_WalksAreValid) {
    auto trw = this->make_trw(/*enable_weights=*/false);
    trw->add_multiple_edges(this->convergence_graph(
        /*num_sources=*/300, /*hub_groups=*/3000));

    const auto walks = trw->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &LINEAR_PICKER, /*num_walks_per_node=*/1,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::NODE_GROUPED);

    ASSERT_NO_FATAL_FAILURE(
        assert_all_walks_valid(walks, static_cast<size_t>(MAX_WALK_LEN)));
}

// Node2Vec now flows through the cooperative pipeline like every other
// picker; smoke check that the path is wired correctly end-to-end.
TYPED_TEST(NodeGroupedTierRoutingTest, Node2Vec_NodeGroupedSmokeTest) {
    auto trw = this->make_trw(/*enable_weights=*/true,
                              /*enable_node2vec=*/true);
    trw->add_multiple_edges(this->convergence_graph(
        /*num_sources=*/10, /*hub_groups=*/20));

    const auto walks = trw->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &NODE2VEC_PICKER, /*num_walks_per_node=*/1,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::NODE_GROUPED);

    ASSERT_NO_FATAL_FAILURE(
        assert_all_walks_valid(walks, static_cast<size_t>(MAX_WALK_LEN)));
}
