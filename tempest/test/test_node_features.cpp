#include <gtest/gtest.h>

#include <vector>

#include "../src/core/tempest.cuh"
#include "../src/graph/node_features.cuh"

template<typename T>
class NodeFeaturesTest : public ::testing::Test {
protected:
    core::Tempest trw;

    NodeFeaturesTest() : trw(/*is_directed=*/true, /*use_gpu=*/T::value) {}

    TemporalGraphData&       data()       { return trw.data(); }
    const TemporalGraphData& data() const { return trw.data(); }
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

TYPED_TEST_SUITE(NodeFeaturesTest, GPU_USAGE_TYPES);

TYPED_TEST(NodeFeaturesTest, SetNodeFeaturesPreservesOldDataOnGrow) {
    const std::vector<int>   first_ids{1, 3};
    const std::vector<float> first_features{1.0f, 2.0f, 3.0f, 4.0f};
    node_features::set_node_features(
        this->data(), /*max_node_id=*/3,
        first_ids.data(), first_ids.size(),
        first_features.data(), /*feature_dim=*/2);

    {
        const auto snap = node_features::snapshot(this->data());
        ASSERT_GE(snap.node_features.size(), 4u * 2u);
        EXPECT_FLOAT_EQ(snap.node_features[1 * 2],     1.0f);
        EXPECT_FLOAT_EQ(snap.node_features[1 * 2 + 1], 2.0f);
    }

    const std::vector<int>   second_ids{5};
    const std::vector<float> second_features{9.0f, 10.0f};
    node_features::set_node_features(
        this->data(), /*max_node_id=*/5,
        second_ids.data(), second_ids.size(),
        second_features.data(), 2);

    const auto snap = node_features::snapshot(this->data());
    EXPECT_FLOAT_EQ(snap.node_features[1 * 2],     1.0f);
    EXPECT_FLOAT_EQ(snap.node_features[1 * 2 + 1], 2.0f);
    EXPECT_FLOAT_EQ(snap.node_features[5 * 2],     9.0f);
    EXPECT_FLOAT_EQ(snap.node_features[5 * 2 + 1], 10.0f);
    EXPECT_EQ(snap.feature_dim, 2u);
}

TYPED_TEST(NodeFeaturesTest, SetNodeFeaturesWithPointers) {
    const std::vector<int>   node_ids{0, 2};
    const std::vector<float> node_features{7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f};

    node_features::set_node_features(
        this->data(), /*max_node_id=*/2,
        node_ids.data(), node_ids.size(),
        node_features.data(), /*feature_dim=*/3);

    const auto snap = node_features::snapshot(this->data());
    EXPECT_FLOAT_EQ(snap.node_features[2],     9.0f);
    EXPECT_FLOAT_EQ(snap.node_features[2 * 3], 1.0f);
    EXPECT_EQ(snap.feature_dim, 3u);
}
