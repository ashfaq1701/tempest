#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "../src/common/const.cuh"
#include "../src/core/tempest.cuh"
#include "../src/graph/edge_data.cuh"
#include "../src/graph/node_edge_index.cuh"

namespace {

static void do_update_timestamp_groups(TemporalGraphData& data) {
#ifdef HAS_CUDA
    if (data.use_gpu) {
        edge_data::update_timestamp_groups_cuda(data);
        return;
    }
#endif
    edge_data::update_timestamp_groups_std(data);
}

static void do_update_temporal_weights(TemporalGraphData& data, double timescale_bound) {
#ifdef HAS_CUDA
    if (data.use_gpu) {
        node_edge_index::update_temporal_weights_cuda(data, timescale_bound);
        return;
    }
#endif
    node_edge_index::update_temporal_weights_std(data, timescale_bound);
}

} // namespace

template<typename T>
class NodeEdgeIndexWeightTest : public ::testing::Test {
protected:
    static void verify_node_weights(const std::vector<size_t>& group_offsets,
                                    const std::vector<double>& weights) {
        for (size_t node = 0; node + 1 < group_offsets.size(); node++) {
            const size_t start = group_offsets[node];
            const size_t end = group_offsets[node + 1];

            if (start < end) {
                for (size_t i = start; i < end; i++) {
                    EXPECT_GE(weights[i], 0.0);
                    if (i > start) {
                        EXPECT_GE(weights[i], weights[i-1]);
                    }
                }
                EXPECT_NEAR(weights[end-1], 1.0, 1e-6);
            }
        }
    }

    static std::vector<double> get_individual_weights(
        const std::vector<double>& cumulative,
        const std::vector<size_t>& offsets,
        const size_t node) {
        const size_t start = offsets[node];
        const size_t end = offsets[node + 1];

        std::vector<double> weights;
        weights.push_back(cumulative[start]);
        for (size_t i = start + 1; i < end; i++) {
            weights.push_back(cumulative[i] - cumulative[i - 1]);
        }
        return weights;
    }

    core::Tempest make_test_graph(const bool directed) {
        core::Tempest local(directed, T::value, -1, /*enable_weight_computation=*/true);
        auto& d = local.data();
        edge_data::push_back(d, 1, 2, 10);
        edge_data::push_back(d, 1, 3, 10);
        edge_data::push_back(d, 1, 4, 20);
        edge_data::push_back(d, 2, 3, 20);
        edge_data::push_back(d, 2, 4, 30);
        edge_data::push_back(d, 3, 4, 40);
        do_update_timestamp_groups(d);
        node_edge_index::rebuild(d);
        do_update_temporal_weights(d, -1);
        return local;
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

TYPED_TEST_SUITE(NodeEdgeIndexWeightTest, GPU_USAGE_TYPES);

TYPED_TEST(NodeEdgeIndexWeightTest, EmptyGraph) {
    core::Tempest trw(/*is_directed=*/true, TypeParam::value, -1, true);
    auto& data = trw.data();
    node_edge_index::rebuild(data);
    do_update_temporal_weights(data, -1);

    const auto idx = node_edge_index::snapshot(data);
    EXPECT_TRUE(idx.outbound_forward_cumulative_weights_exponential.empty());
    EXPECT_TRUE(idx.outbound_backward_cumulative_weights_exponential.empty());
    EXPECT_TRUE(idx.inbound_backward_cumulative_weights_exponential.empty());
}

TYPED_TEST(NodeEdgeIndexWeightTest, DirectedWeightNormalization) {
    auto local = this->make_test_graph(true);
    const auto idx = node_edge_index::snapshot(local.data());

    this->verify_node_weights(idx.count_ts_group_per_node_outbound,
                              idx.outbound_forward_cumulative_weights_exponential);
    this->verify_node_weights(idx.count_ts_group_per_node_outbound,
                              idx.outbound_backward_cumulative_weights_exponential);
    this->verify_node_weights(idx.count_ts_group_per_node_inbound,
                              idx.inbound_backward_cumulative_weights_exponential);
}

TYPED_TEST(NodeEdgeIndexWeightTest, WeightBiasPerNode) {
    core::Tempest trw(/*is_directed=*/true, TypeParam::value, -1, true);
    auto& data = trw.data();
    edge_data::push_back(data, 1, 2, 10);
    edge_data::push_back(data, 1, 3, 20);
    edge_data::push_back(data, 1, 4, 30);
    do_update_timestamp_groups(data);
    node_edge_index::rebuild(data);
    do_update_temporal_weights(data, -1);

    const auto idx = node_edge_index::snapshot(data);

    const auto forward = this->get_individual_weights(
        idx.outbound_forward_cumulative_weights_exponential,
        idx.count_ts_group_per_node_outbound, 1);
    for (size_t i = 0; i < forward.size() - 1; i++) {
        const double expected_ratio = exp(-10);
        EXPECT_NEAR(forward[i+1]/forward[i], expected_ratio, 1e-6);
    }

    const auto backward = this->get_individual_weights(
        idx.outbound_backward_cumulative_weights_exponential,
        idx.count_ts_group_per_node_outbound, 1);
    for (size_t i = 0; i < backward.size() - 1; i++) {
        const double expected_ratio = exp(10);
        EXPECT_NEAR(backward[i + 1]/backward[i], expected_ratio, 1e-6);
    }
}

TYPED_TEST(NodeEdgeIndexWeightTest, ScaledWeightRatios) {
    core::Tempest trw(/*is_directed=*/true, TypeParam::value, -1, true);
    auto& data = trw.data();
    edge_data::push_back(data, 1, 2, 100);
    edge_data::push_back(data, 1, 3, 300);
    edge_data::push_back(data, 1, 4, 500);
    do_update_timestamp_groups(data);
    node_edge_index::rebuild(data);

    constexpr double timescale_bound = 2.0;
    do_update_temporal_weights(data, timescale_bound);

    const auto idx = node_edge_index::snapshot(data);

    const auto forward = this->get_individual_weights(
        idx.outbound_forward_cumulative_weights_exponential,
        idx.count_ts_group_per_node_outbound, 1);

    constexpr double time_scale = timescale_bound / 400.0;

    for (size_t i = 0; i < forward.size() - 1; i++) {
        const double scaled_diff = 200 * time_scale;
        const double expected_ratio = exp(-scaled_diff);
        EXPECT_NEAR(forward[i+1]/forward[i], expected_ratio, 1e-6);
    }

    const auto backward = this->get_individual_weights(
        idx.outbound_backward_cumulative_weights_exponential,
        idx.count_ts_group_per_node_outbound, 1);
    for (size_t i = 0; i < backward.size() - 1; i++) {
        constexpr double scaled_diff = 200 * time_scale;
        const double expected_ratio = exp(scaled_diff);
        EXPECT_NEAR(backward[i+1]/backward[i], expected_ratio, 1e-6);
    }
}

TYPED_TEST(NodeEdgeIndexWeightTest, UndirectedWeightNormalization) {
    auto local = this->make_test_graph(false);
    const auto idx = node_edge_index::snapshot(local.data());

    this->verify_node_weights(idx.count_ts_group_per_node_outbound,
                              idx.outbound_forward_cumulative_weights_exponential);
    this->verify_node_weights(idx.count_ts_group_per_node_outbound,
                              idx.outbound_backward_cumulative_weights_exponential);
    EXPECT_TRUE(idx.inbound_backward_cumulative_weights_exponential.empty());
}

TYPED_TEST(NodeEdgeIndexWeightTest, WeightConsistencyAcrossUpdates) {
    auto local = this->make_test_graph(true);
    auto& data = local.data();

    const auto before = node_edge_index::snapshot(data);

    edge_data::set_size(data, 0);
    do_update_timestamp_groups(data);
    edge_data::push_back(data, 1, 2, 10);
    edge_data::push_back(data, 1, 3, 10);
    do_update_timestamp_groups(data);
    node_edge_index::rebuild(data);
    do_update_temporal_weights(data, -1);

    const auto after = node_edge_index::snapshot(data);

    EXPECT_NE(before.outbound_forward_cumulative_weights_exponential.size(),
              after.outbound_forward_cumulative_weights_exponential.size());
    this->verify_node_weights(after.count_ts_group_per_node_outbound,
                              after.outbound_forward_cumulative_weights_exponential);
}

TYPED_TEST(NodeEdgeIndexWeightTest, SingleTimestampGroupPerNode) {
    core::Tempest trw(/*is_directed=*/true, TypeParam::value, -1, true);
    auto& data = trw.data();
    edge_data::push_back(data, 1, 2, 10);
    edge_data::push_back(data, 1, 3, 10);
    edge_data::push_back(data, 2, 3, 10);
    do_update_timestamp_groups(data);
    node_edge_index::rebuild(data);
    do_update_temporal_weights(data, -1);

    const auto idx = node_edge_index::snapshot(data);
    for (size_t node = 0; node + 1 < idx.count_ts_group_per_node_outbound.size(); node++) {
        const size_t start = idx.count_ts_group_per_node_outbound[node];
        const size_t end   = idx.count_ts_group_per_node_outbound[node + 1];
        if (start < end) {
            EXPECT_EQ(end - start, 1u);
            EXPECT_NEAR(idx.outbound_forward_cumulative_weights_exponential[start],  1.0, 1e-6);
            EXPECT_NEAR(idx.outbound_backward_cumulative_weights_exponential[start], 1.0, 1e-6);
        }
    }
}

TYPED_TEST(NodeEdgeIndexWeightTest, TimescaleBoundZero) {
    core::Tempest trw(/*is_directed=*/true, TypeParam::value, -1, true);
    auto& data = trw.data();
    edge_data::push_back(data, 1, 2, 10);
    edge_data::push_back(data, 1, 3, 20);
    edge_data::push_back(data, 1, 4, 30);
    do_update_timestamp_groups(data);
    node_edge_index::rebuild(data);
    do_update_temporal_weights(data, 0);

    const auto idx = node_edge_index::snapshot(data);
    this->verify_node_weights(idx.count_ts_group_per_node_outbound,
                              idx.outbound_forward_cumulative_weights_exponential);
    this->verify_node_weights(idx.count_ts_group_per_node_outbound,
                              idx.outbound_backward_cumulative_weights_exponential);
}

TYPED_TEST(NodeEdgeIndexWeightTest, TimescaleBoundWithSingleTimestamp) {
    core::Tempest trw(/*is_directed=*/true, TypeParam::value, -1, true);
    auto& data = trw.data();
    constexpr int node_id = 1;
    edge_data::push_back(data, node_id, 2, 10);
    edge_data::push_back(data, node_id, 3, 10);
    edge_data::push_back(data, node_id, 4, 10);
    do_update_timestamp_groups(data);
    node_edge_index::rebuild(data);

    for (const double bound : {-1.0, 0.0, 10.0, 50.0}) {
        do_update_temporal_weights(data, bound);

        const auto idx = node_edge_index::snapshot(data);
        const size_t start = idx.count_ts_group_per_node_outbound[node_id];
        const size_t end   = idx.count_ts_group_per_node_outbound[node_id + 1];
        ASSERT_EQ(end - start, 1u);
        EXPECT_NEAR(idx.outbound_forward_cumulative_weights_exponential[start], 1.0, 1e-6);
        EXPECT_NEAR(idx.outbound_backward_cumulative_weights_exponential[start], 1.0, 1e-6);
    }
}

TYPED_TEST(NodeEdgeIndexWeightTest, WeightOrderPreservation) {
    core::Tempest trw(/*is_directed=*/true, TypeParam::value, -1, true);
    auto& data = trw.data();
    edge_data::push_back(data, 1, 2, 10);
    edge_data::push_back(data, 1, 3, 20);
    edge_data::push_back(data, 1, 4, 30);
    do_update_timestamp_groups(data);
    node_edge_index::rebuild(data);

    do_update_temporal_weights(data, -1);
    const auto unscaled = node_edge_index::snapshot(data);
    const auto unscaled_forward  = unscaled.outbound_forward_cumulative_weights_exponential;
    const auto unscaled_backward = unscaled.outbound_backward_cumulative_weights_exponential;

    do_update_temporal_weights(data, 10.0);
    const auto scaled = node_edge_index::snapshot(data);

    const size_t start = scaled.count_ts_group_per_node_outbound[1];
    const size_t end   = scaled.count_ts_group_per_node_outbound[2];
    for (size_t i = start + 1; i < end; i++) {
        EXPECT_EQ(unscaled_forward[i]  > unscaled_forward[i-1],
                  scaled.outbound_forward_cumulative_weights_exponential[i]  >
                  scaled.outbound_forward_cumulative_weights_exponential[i-1]);
        EXPECT_EQ(unscaled_backward[i] > unscaled_backward[i-1],
                  scaled.outbound_backward_cumulative_weights_exponential[i] >
                  scaled.outbound_backward_cumulative_weights_exponential[i-1]);
    }
}

TYPED_TEST(NodeEdgeIndexWeightTest, TimescaleNormalizationTest) {
    core::Tempest trw(/*is_directed=*/true, TypeParam::value, -1, true);
    auto& data = trw.data();
    edge_data::push_back(data, 1, 2, 100);
    edge_data::push_back(data, 1, 3, 200);
    edge_data::push_back(data, 1, 4, 1000);
    edge_data::push_back(data, 1, 5, 100000);
    do_update_timestamp_groups(data);
    node_edge_index::rebuild(data);

    constexpr double timescale_bound = 5.0;
    do_update_temporal_weights(data, timescale_bound);

    const auto idx = node_edge_index::snapshot(data);
    const auto edges = edge_data::snapshot(data);

    const auto weights = this->get_individual_weights(
        idx.outbound_forward_cumulative_weights_exponential,
        idx.count_ts_group_per_node_outbound, 1);

    double max_weight_ratio = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < weights.size(); i++) {
        for (size_t j = 0; j < weights.size(); j++) {
            if (weights[j] > 0) {
                max_weight_ratio = std::max(max_weight_ratio, log(weights[i] / weights[j]));
            }
        }
    }
    EXPECT_NEAR(max_weight_ratio, timescale_bound, 1e-6);

    const auto& ts = edges.unique_timestamps;
    for (size_t i = 0; i < weights.size() - 1; i++) {
        const double time_ratio = static_cast<double>(ts[i+1] - ts[i]) /
                                   static_cast<double>(ts[weights.size()-1] - ts[0]);
        const double weight_ratio = weights[i] / weights[i+1];
        EXPECT_NEAR(log(weight_ratio), timescale_bound * time_ratio, 1e-6);
    }
}
