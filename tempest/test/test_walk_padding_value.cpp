#include <gtest/gtest.h>
#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>

#include "test_utils.h"
#include "../src/common/const.cuh"
#include "../src/data/enums.cuh"
#include "../src/proxies/Tempest.cuh"

namespace {

// well outside any real node id in sample_data.csv (max id 111).
constexpr int CUSTOM_WALK_PADDING_VALUE = 1000000;
constexpr int MAX_WALK_LEN = 20;
constexpr int NUM_WALKS_PER_NODE = 10;
constexpr RandomPickerType LINEAR_PICKER = RandomPickerType::Linear;

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

template <typename T>
class WalkPaddingValueTest : public ::testing::Test {
protected:
    void SetUp() override {
        sample_edges = read_edges_from_csv(sample_data_path());
        trw = std::make_unique<Tempest>(
            /*is_directed=*/true,
            /*use_gpu=*/T::value,
            /*max_time_capacity=*/-1,
            /*enable_weight_computation=*/false,
            /*enable_temporal_node2vec=*/false,
            /*timescale_bound=*/DEFAULT_TIMESCALE_BOUND,
            /*node2vec_p=*/DEFAULT_NODE2VEC_P,
            /*node2vec_q=*/DEFAULT_NODE2VEC_Q,
            /*walk_padding_value=*/CUSTOM_WALK_PADDING_VALUE);
        trw->add_multiple_edges(sample_edges);
    }

    std::vector<std::tuple<int, int, int64_t>> sample_edges;
    std::unique_ptr<Tempest> trw;
};

TYPED_TEST_SUITE(WalkPaddingValueTest, GPU_USAGE_TYPES);

struct WalkStats {
    size_t num_walks = 0;
    size_t num_walks_longer_than_3 = 0;
};

// (a) prefix has no sentinel/padding leaks; (b) tail is all padding;
// (c) caller gates on enough walks > length 3.
void assert_walks_respect_padding(
    const WalksWithEdgeFeaturesHost& walks,
    const int expected_padding_value,
    const size_t max_walk_len,
    WalkStats* out_stats) {
    *out_stats = WalkStats{};

    const int* nodes_ptr = walks.walk_set.nodes_ptr();
    const size_t* walk_lens_ptr = walks.walk_set.walk_lens_ptr();
    const size_t num_walks_total = walks.walk_set.num_walks();

    for (size_t w = 0; w < num_walks_total; ++w) {
        const size_t walk_len = walk_lens_ptr[w];
        if (walk_len == 0) continue;

        ++out_stats->num_walks;
        if (walk_len > 3) {
            ++out_stats->num_walks_longer_than_3;
        }

        const size_t row_base = w * max_walk_len;

        for (size_t i = 0; i < walk_len; ++i) {
            const int node = nodes_ptr[row_base + i];
            ASSERT_NE(node, -1)
                << "Walk " << w << " hop " << i << " leaked -1 sentinel";
            ASSERT_NE(node, expected_padding_value)
                << "Walk " << w << " hop " << i << " equals padding value";
        }

        for (size_t i = walk_len; i < max_walk_len; ++i) {
            const int node = nodes_ptr[row_base + i];
            ASSERT_EQ(node, expected_padding_value)
                << "Walk " << w << " tail slot " << i << " is " << node;
        }
    }
}

void expect_enough_long_walks(const WalkStats& stats) {
    ASSERT_GT(stats.num_walks, 0u);
    const double frac_long =
        static_cast<double>(stats.num_walks_longer_than_3)
        / static_cast<double>(stats.num_walks);
    EXPECT_GE(frac_long, 0.05)
        << "Only " << stats.num_walks_longer_than_3 << " / "
        << stats.num_walks << " walks reached length > 3.";
}

}  // namespace

TYPED_TEST(WalkPaddingValueTest, FullWalkRespectsConfiguredPaddingValue) {
    const auto walks = this->trw->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &LINEAR_PICKER, NUM_WALKS_PER_NODE,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::FULL_WALK);

    WalkStats stats{};
    ASSERT_NO_FATAL_FAILURE(assert_walks_respect_padding(
        walks, CUSTOM_WALK_PADDING_VALUE,
        static_cast<size_t>(MAX_WALK_LEN), &stats));
    expect_enough_long_walks(stats);
}

TYPED_TEST(WalkPaddingValueTest, NodeGroupedRespectsConfiguredPaddingValue) {
    const auto walks = this->trw->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &LINEAR_PICKER, NUM_WALKS_PER_NODE,
        /*initial_edge_bias=*/nullptr,
        WalkDirection::Forward_In_Time,
        KernelLaunchType::NODE_GROUPED);

    WalkStats stats{};
    ASSERT_NO_FATAL_FAILURE(assert_walks_respect_padding(
        walks, CUSTOM_WALK_PADDING_VALUE,
        static_cast<size_t>(MAX_WALK_LEN), &stats));
    expect_enough_long_walks(stats);
}
