#include <gtest/gtest.h>
#include <random>

#include "../src/proxies/RandomPicker.cuh"
#include "../src/random/pickers.cuh"
#include "../src/utils/utils.cuh"
#include "../src/common/random_gen.cuh"

template<typename T>
class WeightBasedRandomPickerTest : public ::testing::Test
{
protected:

    WeightBasedRandomPicker picker;

    WeightBasedRandomPickerTest(): picker(T::value) {}

    void verify_sampling_range(const std::vector<double>& weights,
                               const int start,
                               const int end,
                               const int num_samples = 1000)
    {
        std::map<int, int> sample_counts;

        Buffer<double> random_nums_buf = generate_n_random_numbers(num_samples, T::value);
        double* random_nums = random_nums_buf.data();

        for (int i = 0; i < num_samples; i++)
        {
            int picked = picker.pick_random_with_provided_number(weights, start, end, random_nums + i);
            EXPECT_GE(picked, start) << "Sampled index below start";
            EXPECT_LT(picked, end) << "Sampled index at or above end";
            ++sample_counts[picked];
        }

        for (int i = start; i < end; i++)
        {
            EXPECT_GT(sample_counts[i], 0)
                << "Index " << i << " was never sampled";
        }
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

TYPED_TEST_SUITE(WeightBasedRandomPickerTest, GPU_USAGE_TYPES);

TYPED_TEST(WeightBasedRandomPickerTest, ValidationChecks)
{
    std::vector<double> weights = {0.2, 0.5, 0.7, 1.0};

    EXPECT_EQ(this->picker.pick_random(weights, -1, 2), -1);

    EXPECT_EQ(this->picker.pick_random(weights, 2, 2), -1);
    EXPECT_EQ(this->picker.pick_random(weights, 2, 1), -1);

    EXPECT_EQ(this->picker.pick_random(weights, 0, 5), -1);
}

TYPED_TEST(WeightBasedRandomPickerTest, FullRangeSampling)
{
    std::vector<double> weights = {0.2, 0.5, 0.7, 1.0};
    this->verify_sampling_range(weights, 0, 4);
}

TYPED_TEST(WeightBasedRandomPickerTest, SubrangeSampling)
{
    std::vector<double> weights = {0.2, 0.5, 0.7, 1.0};
    this->verify_sampling_range(weights, 1, 3);
    this->verify_sampling_range(weights, 0, 2);
    this->verify_sampling_range(weights, 2, 4);
}

TYPED_TEST(WeightBasedRandomPickerTest, SingleElementRange)
{
    std::vector<double> weights = {0.2, 0.5, 0.7, 1.0};
    constexpr int num_samples = 100;

    Buffer<double> random_nums_buf = generate_n_random_numbers(num_samples, TypeParam::value);
    double* random_nums = random_nums_buf.data();

    for (int i = 0; i < num_samples; i++)
    {
        EXPECT_EQ(this->picker.pick_random_with_provided_number(weights, 1, 2, random_nums + i), 1);
    }
}

TYPED_TEST(WeightBasedRandomPickerTest, WeightDistributionTest)
{
    std::vector<double> weights = {0.25, 0.5, 0.75, 1.0};

    std::map<int, int> sample_counts;
    constexpr int num_samples = 100000;

    Buffer<double> random_nums_buf = generate_n_random_numbers(num_samples, TypeParam::value);
    double* random_nums = random_nums_buf.data();

    for (int i = 0; i < num_samples; i++)
    {
        int picked = this->picker.pick_random_with_provided_number(weights, 0, 4, random_nums + i);
        ++sample_counts[picked];
    }

    for (int i = 0; i < 4; i++)
    {
        const double proportion = static_cast<double>(sample_counts[i]) / num_samples;
        EXPECT_NEAR(proportion, 0.25, 0.01)
            << "Proportion for index " << i << " was " << proportion;
    }
}

TYPED_TEST(WeightBasedRandomPickerTest, EdgeCaseWeights)
{
    std::vector<double> small_diffs = {0.1, 0.100001, 0.100002, 0.100003};
    EXPECT_NE(this->picker.pick_random(small_diffs, 0, 4), -1);

    std::vector<double> large_diffs = {0.1, 0.5, 0.9, 1000.0};
    EXPECT_NE(this->picker.pick_random(large_diffs, 0, 4), -1);
}

TEST(WeightPickerPiecewiseCDFTest, SliceStartFixAtNodeBoundary)
{
    // back-to-back piecewise per-node CDFs, each ending at 1.0
    const std::vector<double> weights = {0.3, 1.0, 0.25, 1.0};
    constexpr size_t node_b_begin = 2;
    constexpr size_t node_b_end   = 4;

    constexpr int num_samples = 100000;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::map<int, int> sample_counts;
    for (int i = 0; i < num_samples; i++)
    {
        const double r = dist(rng);
        const int picked = random_pickers::pick_random_exponential_weights(
            weights.data(), weights.size(),
            node_b_begin, node_b_end, r,
            /*slice_start=*/node_b_begin);
        ASSERT_GE(picked, static_cast<int>(node_b_begin));
        ASSERT_LT(picked, static_cast<int>(node_b_end));
        ++sample_counts[picked];
    }

    const double p_group_2 = static_cast<double>(sample_counts[2]) / num_samples;
    const double p_group_3 = static_cast<double>(sample_counts[3]) / num_samples;
    EXPECT_NEAR(p_group_2, 0.25, 0.01)
        << "Node B group 0 should be sampled ~25% of the time; got "
        << p_group_2;
    EXPECT_NEAR(p_group_3, 0.75, 0.01)
        << "Node B group 1 should be sampled ~75% of the time; got "
        << p_group_3;

    // default slice_start=0 on piecewise CDF degenerates to always last group
    std::mt19937 rng_buggy(42);
    int degenerate_hits = 0;
    for (int i = 0; i < num_samples; i++)
    {
        const double r = dist(rng_buggy);
        const int picked = random_pickers::pick_random_exponential_weights(
            weights.data(), weights.size(),
            node_b_begin, node_b_end, r);
        if (picked == static_cast<int>(node_b_end - 1)) ++degenerate_hits;
    }
    EXPECT_EQ(degenerate_hits, num_samples);
}