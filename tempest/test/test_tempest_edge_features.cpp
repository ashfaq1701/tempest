#include <gtest/gtest.h>

#include <array>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "test_utils.h"
#include "../src/proxies/Tempest.cuh"

constexpr int MAX_WALK_LEN = 10;
constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;

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

template<typename T>
class TemporalWalkTestWithEdgeFeatures : public ::testing::Test {
protected:
    void SetUp() override {
        tempest = std::make_unique<Tempest>(true, T::value, -1, true, false, 10.0);

        edges = {
            {1, 2, 100},
            {2, 3, 101},
            {3, 5, 101},
            {1, 5, 110},
            {2, 5, 120},
            {2, 4, 110},
            {3, 4, 130},
            {4, 5, 130},
            {4, 1, 140},
            {5, 1, 140},
            {3, 1, 150},
            {2, 1, 150},
            {5, 3, 160},
            {3, 2, 160},
            {4, 2, 170}
        };

        edge_features = {
            0.1f, 0.2f, 0.3f,
            0.4f, 0.5f, 0.6f,
            0.7f, 0.8f, 0.9f,
            1.0f, 1.1f, 1.2f,
            1.3f, 1.4f, 1.5f,
            1.6f, 1.7f, 1.8f,
            1.9f, 2.0f, 2.1f,
            2.2f, 2.3f, 2.4f,
            2.5f, 2.6f, 2.7f,
            2.8f, 2.9f, 3.0f,
            3.1f, 3.2f, 3.3f,
            3.4f, 3.5f, 3.6f,
            3.7f, 3.8f, 3.9f,
            4.0f, 4.1f, 4.2f,
            4.3f, 4.4f, 4.5f
        };

        tempest->add_multiple_edges(edges, edge_features.data(), feature_dim);

        for (size_t i = 0; i < edges.size(); ++i) {
            const auto edge = edges[i];
            expected_feature_by_edge[edge] = {
                edge_features[i * feature_dim],
                edge_features[i * feature_dim + 1],
                edge_features[i * feature_dim + 2]
            };
        }
    }

    [[nodiscard]] static size_t edge_slot_index(const WalksWithEdgeFeaturesHost& walks, size_t walk_idx, size_t step_idx) {
        return walk_idx * (walks.walk_set.max_len() - 1) + step_idx;
    }

    static constexpr size_t feature_dim = 3;

    std::vector<std::tuple<int, int, int64_t>> edges;
    std::vector<float> edge_features;
    std::map<std::tuple<int, int, int64_t>, std::array<float, feature_dim>> expected_feature_by_edge;
    std::unique_ptr<Tempest> tempest;
};

TYPED_TEST_SUITE(TemporalWalkTestWithEdgeFeatures, GPU_USAGE_TYPES);

TYPED_TEST(TemporalWalkTestWithEdgeFeatures, ReturnsFeatureMetadata) {
    const auto walks = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, 20);

    EXPECT_EQ(walks.feature_dim, static_cast<int>(this->feature_dim));
    EXPECT_NE(walks.walk_edge_features.data(), nullptr);
}

TYPED_TEST(TemporalWalkTestWithEdgeFeatures, PopulatesWalkEdgeFeaturesForTraversedEdges) {
    const auto walks = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, 40, nullptr, WalkDirection::Forward_In_Time);

    const auto& walk_set = walks.walk_set;
    ASSERT_GT(walk_set.num_walks(), 0);

    const int*     nodes      = walk_set.nodes_ptr();
    const int64_t* timestamps = walk_set.timestamps_ptr();
    const size_t*  walk_lens  = walk_set.walk_lens_ptr();
    const int64_t* edge_ids   = walk_set.edge_ids_ptr();
    const float*   edge_feats = walks.walk_edge_features.data();

    for (size_t walk_idx = 0; walk_idx < walk_set.num_walks(); ++walk_idx) {
        const size_t walk_len = walk_lens[walk_idx];
        if (walk_len < 2) {
            continue;
        }

        const size_t node_offset = walk_idx * walk_set.max_len();

        for (size_t step_idx = 0; step_idx + 1 < walk_len; ++step_idx) {
            const auto edge_slot = this->edge_slot_index(walks, walk_idx, step_idx);
            const int64_t edge_id = edge_ids[edge_slot];
            ASSERT_NE(edge_id, EMPTY_EDGE_ID);

            const auto edge = std::make_tuple(
                nodes[node_offset + step_idx],
                nodes[node_offset + step_idx + 1],
                timestamps[node_offset + step_idx + 1]);

            const auto it = this->expected_feature_by_edge.find(edge);
            ASSERT_NE(it, this->expected_feature_by_edge.end())
                << "Traversed edge missing in expected-feature map";

            const float* sampled = edge_feats + (edge_slot * this->feature_dim);
            EXPECT_FLOAT_EQ(sampled[0], it->second[0]);
            EXPECT_FLOAT_EQ(sampled[1], it->second[1]);
            EXPECT_FLOAT_EQ(sampled[2], it->second[2]);
        }
    }
}

TYPED_TEST(TemporalWalkTestWithEdgeFeatures, KeepsUnusedEdgeSlotsAsEmptyAndZeroedFeatures) {
    const auto walks = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, 20);

    const auto& walk_set = walks.walk_set;
    const size_t*  walk_lens  = walk_set.walk_lens_ptr();
    const int64_t* edge_ids   = walk_set.edge_ids_ptr();
    const float*   edge_feats = walks.walk_edge_features.data();

    for (size_t walk_idx = 0; walk_idx < walk_set.num_walks(); ++walk_idx) {
        const size_t walk_len = walk_lens[walk_idx];

        for (size_t step_idx = walk_len > 0 ? walk_len - 1 : 0; step_idx < walk_set.max_len() - 1; ++step_idx) {
            const size_t edge_slot = this->edge_slot_index(walks, walk_idx, step_idx);
            EXPECT_EQ(edge_ids[edge_slot], EMPTY_EDGE_ID);

            const float* sampled = edge_feats + (edge_slot * this->feature_dim);
            EXPECT_FLOAT_EQ(sampled[0], 0.0f);
            EXPECT_FLOAT_EQ(sampled[1], 0.0f);
            EXPECT_FLOAT_EQ(sampled[2], 0.0f);
        }
    }
}

TYPED_TEST(TemporalWalkTestWithEdgeFeatures, RejectsFeatureDimMismatchOnSubsequentIngestion) {
    const std::vector<std::tuple<int, int, int64_t>> new_edges = {
        {7, 8, 200},
        {8, 9, 201}
    };

    std::vector<float> mismatched_features = {
        0.1f, 0.2f,
        0.3f, 0.4f
    };

    EXPECT_THROW(
        this->tempest->add_multiple_edges(new_edges, mismatched_features.data(), 2),
        std::runtime_error);
}

TYPED_TEST(TemporalWalkTestWithEdgeFeatures, RejectsMissingFeaturesWhenFeatureModeIsEnabled) {
    const std::vector<std::tuple<int, int, int64_t>> new_edges = {
        {7, 8, 200}
    };

    EXPECT_THROW(
        this->tempest->add_multiple_edges(new_edges),
        std::runtime_error);
}
