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
class TemporalGraphWeightTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_edges = {
            Edge{1, 2, 10},
            Edge{1, 3, 10},
            Edge{2, 3, 20},
            Edge{2, 4, 20},
            Edge{3, 4, 30},
            Edge{4, 1, 40}
        };
    }

    static void verify_cumulative_weights(const std::vector<double>& weights) {
        ASSERT_FALSE(weights.empty());
        for (size_t i = 0; i < weights.size(); i++) {
            EXPECT_GE(weights[i], 0.0);
            if (i > 0) {
                EXPECT_GE(weights[i], weights[i-1]);
            }
        }
        EXPECT_NEAR(weights.back(), 1.0, 1e-6);
    }

    static std::vector<double> get_individual_weights(const std::vector<double>& cumulative) {
        std::vector<double> weights;
        weights.push_back(cumulative[0]);
        for (size_t i = 1; i < cumulative.size(); i++) {
            weights.push_back(cumulative[i] - cumulative[i-1]);
        }
        return weights;
    }

    std::vector<Edge> test_edges;
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

TYPED_TEST_SUITE(TemporalGraphWeightTest, GPU_USAGE_TYPES);

TYPED_TEST(TemporalGraphWeightTest, EdgeWeightComputation) {
    core::Tempest graph(
        /*is_directed=*/false, TypeParam::value,
        /*max_time_capacity=*/-1, /*enable_weight_computation=*/true,
        /*enable_temporal_node2vec=*/false, -1);
    test_util::add_edges(graph, this->test_edges);

    const auto snap = edge_data::snapshot(graph.data());
    ASSERT_EQ(snap.forward_cumulative_weights_exponential.size(), 4u);
    ASSERT_EQ(snap.backward_cumulative_weights_exponential.size(), 4u);

    this->verify_cumulative_weights(snap.forward_cumulative_weights_exponential);
    this->verify_cumulative_weights(snap.backward_cumulative_weights_exponential);

    const auto forward_weights  = this->get_individual_weights(snap.forward_cumulative_weights_exponential);
    const auto backward_weights = this->get_individual_weights(snap.backward_cumulative_weights_exponential);

    for (size_t i = 0; i < forward_weights.size() - 1; i++) {
        EXPECT_GE(forward_weights[i], forward_weights[i+1]);
    }

    for (size_t i = 0; i < backward_weights.size() - 1; i++) {
        EXPECT_LE(backward_weights[i], backward_weights[i+1]);
    }
}

TYPED_TEST(TemporalGraphWeightTest, NodeWeightComputation) {
    core::Tempest graph(
        /*is_directed=*/true, TypeParam::value, -1, /*enable_weight_computation=*/true);
    test_util::add_edges(graph, this->test_edges);

    const auto idx = node_edge_index::snapshot(graph.data());
    constexpr int node_id = 2;

    const size_t start_pos = idx.count_ts_group_per_node_outbound[node_id];
    const size_t end_pos   = idx.count_ts_group_per_node_outbound[node_id + 1];
    ASSERT_GT(end_pos, start_pos);

    const std::vector<double> node_out_weights(
        idx.outbound_forward_cumulative_weights_exponential.begin() + static_cast<int>(start_pos),
        idx.outbound_forward_cumulative_weights_exponential.begin() + static_cast<int>(end_pos));

    ASSERT_FALSE(node_out_weights.empty());
    for (size_t i = 0; i < node_out_weights.size(); i++) {
        EXPECT_GE(node_out_weights[i], 0.0);
        if (i > 0) {
            EXPECT_GE(node_out_weights[i], node_out_weights[i-1]);
        }
    }
    EXPECT_NEAR(node_out_weights.back(), 1.0, 1e-6);
}

TYPED_TEST(TemporalGraphWeightTest, WeightBasedSampling) {
    core::Tempest graph(
        /*is_directed=*/true, TypeParam::value, -1, true, false, -1);
    test_util::add_edges(graph, this->test_edges);

    std::map<int64_t, int> forward_samples;
    for (int i = 0; i < 100; i++) {
        auto edge = test_util::get_edge_at(graph.data(), RandomPickerType::ExponentialWeight, 20, true);
        EXPECT_GT(edge.ts, 20);
        forward_samples[edge.ts]++;
    }
    EXPECT_GT(forward_samples[30], forward_samples[40]);

    std::map<int64_t, int> backward_samples;
    for (int i = 0; i < 100; i++) {
        auto edge = test_util::get_edge_at(graph.data(), RandomPickerType::ExponentialWeight, 30, false);
        EXPECT_LT(edge.ts, 30);
        backward_samples[edge.ts]++;
    }
    EXPECT_GT(backward_samples[20], backward_samples[10]);

    backward_samples.clear();
    for (int i = 0; i < 100; i++) {
        auto edge = test_util::get_edge_at(graph.data(), RandomPickerType::ExponentialWeight, 50, false);
        backward_samples[edge.ts]++;
    }
    EXPECT_GT(backward_samples[40], backward_samples[30]);
}

TYPED_TEST(TemporalGraphWeightTest, EdgeCases) {
    {
        const core::Tempest empty_graph(false, TypeParam::value, -1, true);
        EXPECT_EQ(empty_graph.data().forward_cumulative_weights_exponential.size(),  0u);
        EXPECT_EQ(empty_graph.data().backward_cumulative_weights_exponential.size(), 0u);
    }

    {
        core::Tempest single_edge_graph(false, TypeParam::value, -1, true);
        test_util::add_edges(single_edge_graph, {Edge{1, 2, 10}});

        const auto snap = edge_data::snapshot(single_edge_graph.data());
        ASSERT_EQ(snap.forward_cumulative_weights_exponential.size(),  1u);
        ASSERT_EQ(snap.backward_cumulative_weights_exponential.size(), 1u);
        EXPECT_NEAR(snap.forward_cumulative_weights_exponential[0],  1.0, 1e-6);
        EXPECT_NEAR(snap.backward_cumulative_weights_exponential[0], 1.0, 1e-6);
    }
}

TYPED_TEST(TemporalGraphWeightTest, TimescaleBoundZero) {
    core::Tempest graph(
        /*is_directed=*/true, TypeParam::value, -1, true, false, 0);
    test_util::add_edges(graph, this->test_edges);

    const auto snap = edge_data::snapshot(graph.data());
    this->verify_cumulative_weights(snap.forward_cumulative_weights_exponential);
    this->verify_cumulative_weights(snap.backward_cumulative_weights_exponential);
}

TYPED_TEST(TemporalGraphWeightTest, TimescaleBoundSampling) {
    core::Tempest scaled_graph(
        /*is_directed=*/true, TypeParam::value, -1, true, false, 10.0);
    core::Tempest unscaled_graph(
        /*is_directed=*/true, TypeParam::value, -1, true, false, -1);

    test_util::add_edges(scaled_graph,   this->test_edges);
    test_util::add_edges(unscaled_graph, this->test_edges);

    std::map<int64_t, int> scaled_samples, unscaled_samples;
    constexpr int num_samples = 1000;

    Buffer<double> random_nums_buf = generate_n_random_numbers(num_samples * 4, TypeParam::value);
    const double* random_nums = random_nums_buf.data();

    for (int i = 0; i < num_samples; i++) {
        auto edge1 = test_util::get_edge_at_with_provided_nums(
            scaled_graph.data(), RandomPickerType::ExponentialWeight, random_nums + i * 4, 20, true);
        auto edge2 = test_util::get_edge_at_with_provided_nums(
            unscaled_graph.data(), RandomPickerType::ExponentialWeight, random_nums + i * 4 + 2, 20, true);
        ++scaled_samples[edge1.ts];
        ++unscaled_samples[edge2.ts];
    }

    EXPECT_GT(scaled_samples[30],   scaled_samples[40]);
    EXPECT_GT(unscaled_samples[30], unscaled_samples[40]);
}

TYPED_TEST(TemporalGraphWeightTest, WeightScalingPrecision) {
    core::Tempest graph(
        /*is_directed=*/true, TypeParam::value, -1, true, false, 2.0);
    test_util::add_edges(graph, {
        Edge{1, 2, 100},
        Edge{1, 3, 200},
        Edge{1, 4, 300}
    });

    const auto snap = edge_data::snapshot(graph.data());
    const auto weights = this->get_individual_weights(snap.forward_cumulative_weights_exponential);
    ASSERT_EQ(weights.size(), 3u);

    constexpr double time_scale = 2.0 / 200.0;

    for (size_t i = 0; i < weights.size() - 1; i++) {
        constexpr auto time_diff = 100.0;
        const double expected_ratio = exp(-time_diff * time_scale);
        const double actual_ratio = weights[i+1] / weights[i];
        EXPECT_NEAR(actual_ratio, expected_ratio, 1e-6);
    }

    constexpr int node_id = 1;
    const auto idx = node_edge_index::snapshot(graph.data());
    const size_t start = idx.count_ts_group_per_node_outbound[node_id];

    const auto node_individual_weights =
        this->get_individual_weights(idx.outbound_forward_cumulative_weights_exponential);

    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_NEAR(node_individual_weights[start + i], weights[i], 1e-6);
    }
}

TYPED_TEST(TemporalGraphWeightTest, DifferentTimescaleBounds) {
    const std::vector<double> bounds = {2.0, 5.0, 10.};

    for (double bound : bounds) {
        constexpr int num_samples = 10000;

        core::Tempest graph(
            /*is_directed=*/true, TypeParam::value, -1, true, false, bound);
        test_util::add_edges(graph, this->test_edges);

        Buffer<double> random_nums_buf = generate_n_random_numbers(num_samples * 2, TypeParam::value);
        const double* random_nums = random_nums_buf.data();

        std::map<int64_t, int> samples;
        for (int i = 0; i < num_samples; i++) {
            auto edge = test_util::get_edge_at_with_provided_nums(
                graph.data(), RandomPickerType::ExponentialWeight, random_nums + i * 2, -1, true);
            ++samples[edge.ts];
        }

        const std::vector<int64_t> timestamps = {10, 20, 30, 40};
        for (size_t i = 0; i < timestamps.size() - 1; i++) {
            EXPECT_GT(samples[timestamps[i]], samples[timestamps[i + 1]])
                << "At bound " << bound;
        }
    }
}

TYPED_TEST(TemporalGraphWeightTest, SingleTimestampWithBounds) {
    const std::vector<Edge> single_ts_edges = {
        Edge {1, 2, 100},
        Edge {2, 3, 100},
        Edge {3, 4, 100}
    };

    for (double bound : {-1.0, 0.0, 10.0, 50.0}) {
        core::Tempest graph(
            /*is_directed=*/true, TypeParam::value, -1, true, false, bound);
        test_util::add_edges(graph, single_ts_edges);

        const auto snap = edge_data::snapshot(graph.data());
        ASSERT_EQ(snap.forward_cumulative_weights_exponential.size(),  1u);
        ASSERT_EQ(snap.backward_cumulative_weights_exponential.size(), 1u);
        EXPECT_NEAR(snap.forward_cumulative_weights_exponential[0],  1.0, 1e-6);
        EXPECT_NEAR(snap.backward_cumulative_weights_exponential[0], 1.0, 1e-6);
    }
}
