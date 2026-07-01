#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../src/core/tempest.cuh"
#include "../src/graph/edge_data.cuh"

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
        edge_data::update_temporal_weights_cuda(data, timescale_bound);
        return;
    }
#endif
    edge_data::update_temporal_weights_std(data, timescale_bound);
}

static void add_test_edges(TemporalGraphData& data) {
    edge_data::push_back(data, 1, 2, 10);
    edge_data::push_back(data, 1, 3, 10);
    edge_data::push_back(data, 2, 3, 20);
    edge_data::push_back(data, 2, 4, 20);
    edge_data::push_back(data, 3, 4, 30);
    edge_data::push_back(data, 4, 1, 40);
    do_update_timestamp_groups(data);
}

} // namespace

template<typename T>
class EdgeDataWeightTest : public ::testing::Test {
protected:
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
        weights.reserve(cumulative.size());
        weights.push_back(cumulative[0]);
        for (size_t i = 1; i < cumulative.size(); i++) {
            weights.push_back(cumulative[i] - cumulative[i-1]);
        }
        return weights;
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

TYPED_TEST_SUITE(EdgeDataWeightTest, GPU_USAGE_TYPES);

TYPED_TEST(EdgeDataWeightTest, SingleTimestampGroup) {
    core::Tempest trw(/*is_directed=*/true, TypeParam::value,
        /*max_time_capacity=*/-1, /*enable_weight_computation=*/true);
    auto& data = trw.data();
    edge_data::push_back(data, 1, 2, 10);
    edge_data::push_back(data, 2, 3, 10);
    do_update_timestamp_groups(data);
    do_update_temporal_weights(data, -1);

    const auto snap = edge_data::snapshot(data);
    ASSERT_EQ(snap.forward_cumulative_weights_exponential.size(), 1u);
    ASSERT_EQ(snap.backward_cumulative_weights_exponential.size(), 1u);

    EXPECT_NEAR(snap.forward_cumulative_weights_exponential[0], 1.0, 1e-6);
    EXPECT_NEAR(snap.backward_cumulative_weights_exponential[0], 1.0, 1e-6);
}

TYPED_TEST(EdgeDataWeightTest, WeightNormalization) {
    core::Tempest trw(true, TypeParam::value, -1, true);
    auto& data = trw.data();
    add_test_edges(data);
    do_update_temporal_weights(data, -1);

    const auto snap = edge_data::snapshot(data);
    ASSERT_EQ(snap.forward_cumulative_weights_exponential.size(), 4u);
    ASSERT_EQ(snap.backward_cumulative_weights_exponential.size(), 4u);

    this->verify_cumulative_weights(snap.forward_cumulative_weights_exponential);
    this->verify_cumulative_weights(snap.backward_cumulative_weights_exponential);
}

TYPED_TEST(EdgeDataWeightTest, ForwardWeightBias) {
    core::Tempest trw(true, TypeParam::value, -1, true);
    auto& data = trw.data();
    add_test_edges(data);
    do_update_temporal_weights(data, -1);

    const auto snap = edge_data::snapshot(data);
    const auto forward_weights =
        this->get_individual_weights(snap.forward_cumulative_weights_exponential);

    for (size_t i = 0; i < forward_weights.size() - 1; i++) {
        EXPECT_GT(forward_weights[i], forward_weights[i+1])
            << "Forward weight at index " << i << " should be greater than weight at " << i+1;
    }
}

TYPED_TEST(EdgeDataWeightTest, BackwardWeightBias) {
    core::Tempest trw(true, TypeParam::value, -1, true);
    auto& data = trw.data();
    add_test_edges(data);
    do_update_temporal_weights(data, -1);

    const auto snap = edge_data::snapshot(data);
    const auto backward_weights =
        this->get_individual_weights(snap.backward_cumulative_weights_exponential);

    for (size_t i = 0; i < backward_weights.size() - 1; i++) {
        EXPECT_LT(backward_weights[i], backward_weights[i+1])
            << "Backward weight at index " << i << " should be less than weight at " << i+1;
    }
}

TYPED_TEST(EdgeDataWeightTest, WeightExponentialDecay) {
    core::Tempest trw(true, TypeParam::value, -1, true);
    auto& data = trw.data();

    edge_data::push_back(data, 1, 2, 10);
    edge_data::push_back(data, 2, 3, 20);
    edge_data::push_back(data, 3, 4, 30);
    do_update_timestamp_groups(data);
    do_update_temporal_weights(data, -1);

    const auto snap = edge_data::snapshot(data);
    const auto forward_weights = this->get_individual_weights(snap.forward_cumulative_weights_exponential);
    const auto backward_weights = this->get_individual_weights(snap.backward_cumulative_weights_exponential);

    for (size_t i = 0; i < forward_weights.size() - 1; i++) {
        const auto time_diff = snap.unique_timestamps[i+1] - snap.unique_timestamps[i];
        if (forward_weights[i+1] > 0 && forward_weights[i] > 0) {
            const double log_ratio = log(forward_weights[i+1]/forward_weights[i]);
            EXPECT_NEAR(log_ratio, -time_diff, 1e-6);
        }
    }

    for (size_t i = 0; i < backward_weights.size() - 1; i++) {
        const auto time_diff = snap.unique_timestamps[i+1] - snap.unique_timestamps[i];
        if (backward_weights[i+1] > 0 && backward_weights[i] > 0) {
            const double log_ratio = log(backward_weights[i+1]/backward_weights[i]);
            EXPECT_NEAR(log_ratio, time_diff, 1e-6);
        }
    }
}

TYPED_TEST(EdgeDataWeightTest, UpdateWeights) {
    core::Tempest trw(true, TypeParam::value, -1, true);
    auto& data = trw.data();
    add_test_edges(data);
    do_update_temporal_weights(data, -1);

    const auto before = edge_data::snapshot(data);

    edge_data::push_back(data, 1, 4, 50);
    do_update_timestamp_groups(data);
    do_update_temporal_weights(data, -1);

    const auto after = edge_data::snapshot(data);

    EXPECT_NE(before.forward_cumulative_weights_exponential.size(),
              after.forward_cumulative_weights_exponential.size());
    EXPECT_NE(before.backward_cumulative_weights_exponential.size(),
              after.backward_cumulative_weights_exponential.size());

    this->verify_cumulative_weights(after.forward_cumulative_weights_exponential);
    this->verify_cumulative_weights(after.backward_cumulative_weights_exponential);
}

TYPED_TEST(EdgeDataWeightTest, TimescaleBoundZero) {
    core::Tempest trw(true, TypeParam::value, -1, true);
    auto& data = trw.data();
    add_test_edges(data);
    do_update_temporal_weights(data, 0);

    const auto snap = edge_data::snapshot(data);
    this->verify_cumulative_weights(snap.forward_cumulative_weights_exponential);
    this->verify_cumulative_weights(snap.backward_cumulative_weights_exponential);
}

TYPED_TEST(EdgeDataWeightTest, TimescaleBoundPositive) {
    core::Tempest trw(true, TypeParam::value, -1, true);
    auto& data = trw.data();
    add_test_edges(data);
    constexpr double timescale_bound = 30.0;
    do_update_temporal_weights(data, timescale_bound);

    const auto snap = edge_data::snapshot(data);
    const auto& f_cum = snap.forward_cumulative_weights_exponential;
    const auto& b_cum = snap.backward_cumulative_weights_exponential;

    std::vector<double> forward_weights;
    forward_weights.push_back(f_cum[0]);
    for (size_t i = 1; i < f_cum.size(); i++) {
        forward_weights.push_back(f_cum[i] - f_cum[i-1]);
    }
    for (size_t i = 0; i < forward_weights.size() - 1; i++) {
        EXPECT_GT(forward_weights[i], forward_weights[i+1]);
    }

    std::vector<double> backward_weights;
    backward_weights.push_back(b_cum[0]);
    for (size_t i = 1; i < b_cum.size(); i++) {
        backward_weights.push_back(b_cum[i] - b_cum[i-1]);
    }
    for (size_t i = 0; i < backward_weights.size() - 1; i++) {
        EXPECT_LT(backward_weights[i], backward_weights[i+1]);
    }
}

TYPED_TEST(EdgeDataWeightTest, ScalingComparison) {
    core::Tempest trw(true, TypeParam::value, -1, true);
    auto& data = trw.data();
    add_test_edges(data);

    std::vector<double> weights_unscaled, weights_scaled;

    do_update_temporal_weights(data, -1);
    {
        const auto snap = edge_data::snapshot(data);
        for (size_t i = 1; i < snap.forward_cumulative_weights_exponential.size(); i++) {
            weights_unscaled.push_back(
                snap.forward_cumulative_weights_exponential[i] /
                snap.forward_cumulative_weights_exponential[i-1]);
        }
    }

    do_update_temporal_weights(data, 50.0);
    {
        const auto snap = edge_data::snapshot(data);
        for (size_t i = 1; i < snap.forward_cumulative_weights_exponential.size(); i++) {
            weights_scaled.push_back(
                snap.forward_cumulative_weights_exponential[i] /
                snap.forward_cumulative_weights_exponential[i-1]);
        }
    }

    for (size_t i = 0; i < weights_unscaled.size(); i++) {
        EXPECT_NEAR(weights_scaled[i], weights_unscaled[i], 1e-2);
    }
}

TYPED_TEST(EdgeDataWeightTest, ScaledWeightBounds) {
    core::Tempest trw(true, TypeParam::value, -1, true);
    auto& data = trw.data();
    edge_data::push_back(data, 1, 2, 100);
    edge_data::push_back(data, 2, 3, 300);
    edge_data::push_back(data, 3, 4, 700);
    do_update_timestamp_groups(data);

    constexpr double timescale_bound = 2.0;
    do_update_temporal_weights(data, timescale_bound);

    const auto snap = edge_data::snapshot(data);
    const auto forward_weights = this->get_individual_weights(snap.forward_cumulative_weights_exponential);
    const auto backward_weights = this->get_individual_weights(snap.backward_cumulative_weights_exponential);

    for (size_t i = 0; i < forward_weights.size(); i++) {
        for (size_t j = 0; j < forward_weights.size(); j++) {
            if (forward_weights[j] > 0 && forward_weights[i] > 0) {
                const double log_ratio = log(forward_weights[i] / forward_weights[j]);
                EXPECT_LE(abs(log_ratio), timescale_bound + 1e-6);
            }
        }
    }

    for (size_t i = 0; i < backward_weights.size(); i++) {
        for (size_t j = 0; j < backward_weights.size(); j++) {
            if (backward_weights[j] > 0 && backward_weights[i] > 0) {
                const double log_ratio = log(backward_weights[i] / backward_weights[j]);
                EXPECT_LE(abs(log_ratio), timescale_bound + 1e-6);
            }
        }
    }
}

TYPED_TEST(EdgeDataWeightTest, DifferentTimescaleBounds) {
    core::Tempest trw(true, TypeParam::value, -1, true);
    auto& data = trw.data();
    add_test_edges(data);

    const std::vector<double> bounds = {5.0, 10.0, 20.0};
    std::vector<std::vector<double>> scaled_ratios;

    for (const double bound : bounds) {
        do_update_temporal_weights(data, bound);
        const auto snap = edge_data::snapshot(data);
        std::vector<double> ratios;
        for (size_t i = 1; i < snap.forward_cumulative_weights_exponential.size(); i++) {
            ratios.push_back(snap.forward_cumulative_weights_exponential[i] /
                             snap.forward_cumulative_weights_exponential[i-1]);
        }
        scaled_ratios.push_back(ratios);
    }

    for (size_t i = 0; i < scaled_ratios[0].size(); i++) {
        for (size_t j = 1; j < scaled_ratios.size(); j++) {
            EXPECT_EQ(scaled_ratios[0][i] > 1.0, scaled_ratios[j][i] > 1.0);
        }
    }
}

TYPED_TEST(EdgeDataWeightTest, SingleTimestampWithBounds) {
    core::Tempest trw(true, TypeParam::value, -1, true);
    auto& data = trw.data();
    edge_data::push_back(data, 1, 2, 100);
    edge_data::push_back(data, 2, 3, 100);
    edge_data::push_back(data, 3, 4, 100);
    do_update_timestamp_groups(data);

    for (double bound : {-1.0, 0.0, 10.0, 50.0}) {
        do_update_temporal_weights(data, bound);
        const auto snap = edge_data::snapshot(data);
        ASSERT_EQ(snap.forward_cumulative_weights_exponential.size(), 1u);
        ASSERT_EQ(snap.backward_cumulative_weights_exponential.size(), 1u);
        EXPECT_NEAR(snap.forward_cumulative_weights_exponential[0], 1.0, 1e-6);
        EXPECT_NEAR(snap.backward_cumulative_weights_exponential[0], 1.0, 1e-6);
    }
}

TYPED_TEST(EdgeDataWeightTest, WeightMonotonicity) {
    core::Tempest trw(true, TypeParam::value, -1, true);
    auto& data = trw.data();
    add_test_edges(data);

    const double timescale_bound = 20.0;
    do_update_temporal_weights(data, timescale_bound);

    const auto snap = edge_data::snapshot(data);
    const auto& f_cum = snap.forward_cumulative_weights_exponential;
    const auto& b_cum = snap.backward_cumulative_weights_exponential;

    for (size_t i = 2; i < f_cum.size(); i++) {
        const double prev_diff = f_cum[i-1] - f_cum[i-2];
        const double curr_diff = f_cum[i]   - f_cum[i-1];
        EXPECT_GE(prev_diff, curr_diff);
    }

    for (size_t i = 2; i < b_cum.size(); i++) {
        const double prev_diff = b_cum[i-1] - b_cum[i-2];
        const double curr_diff = b_cum[i]   - b_cum[i-1];
        EXPECT_LE(prev_diff, curr_diff);
    }
}

TYPED_TEST(EdgeDataWeightTest, TimescaleScalingPrecision) {
    core::Tempest trw(true, TypeParam::value, -1, true);
    auto& data = trw.data();
    edge_data::push_back(data, 1, 2, 100);
    edge_data::push_back(data, 2, 3, 300);
    edge_data::push_back(data, 3, 4, 700);
    do_update_timestamp_groups(data);

    constexpr double timescale_bound = 2.0;
    do_update_temporal_weights(data, timescale_bound);

    const auto snap = edge_data::snapshot(data);
    const auto forward_weights = this->get_individual_weights(snap.forward_cumulative_weights_exponential);
    const auto backward_weights = this->get_individual_weights(snap.backward_cumulative_weights_exponential);

    constexpr double time_scale = timescale_bound / 600.0;

    for (size_t i = 0; i < forward_weights.size() - 1; i++) {
        const auto time_diff = static_cast<double>(
            snap.unique_timestamps[i+1] - snap.unique_timestamps[i]);
        const double expected_ratio = exp(-time_diff * time_scale);
        const double actual_ratio = forward_weights[i+1] / forward_weights[i];
        EXPECT_NEAR(actual_ratio, expected_ratio, 1e-6);
    }

    for (size_t i = 0; i < backward_weights.size() - 1; i++) {
        const auto time_diff = static_cast<double>(
            snap.unique_timestamps[i+1] - snap.unique_timestamps[i]);
        const double expected_ratio = exp(time_diff * time_scale);
        const double actual_ratio = backward_weights[i+1] / backward_weights[i];
        EXPECT_NEAR(actual_ratio, expected_ratio, 1e-6);
    }
}
