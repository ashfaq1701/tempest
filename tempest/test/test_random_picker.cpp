#include <cmath>

#include <gtest/gtest.h>
#include "../src/proxies/RandomPicker.cuh"
#include "../src/common/random_gen.cuh"

constexpr int RANDOM_START = 0;
constexpr int RANDOM_END = 10000;
constexpr int RANDOM_NUM_SAMPLES = 100000;

template<typename T>
class RandomPickerTest : public ::testing::Test {
protected:

    LinearRandomPicker linear_picker;
    ExponentialIndexRandomPicker exp_picker;

    RandomPickerTest(): linear_picker(T::value), exp_picker(T::value) {}

    double compute_average_picks(const bool use_exponential, const bool prioritize_end) {
        double sum = 0;
        Buffer<double> random_nums_buf = generate_n_random_numbers(RANDOM_NUM_SAMPLES, T::value);
        double* random_nums = random_nums_buf.data();

        for (int i = 0; i < RANDOM_NUM_SAMPLES; i++) {
            const int pick = use_exponential ?
                                 exp_picker.pick_random_with_provided_number(RANDOM_START, RANDOM_END, prioritize_end, random_nums + i) :
                                 linear_picker.pick_random_with_provided_number(RANDOM_START, RANDOM_END, prioritize_end, random_nums + i);
            sum += pick;
        }
        return sum / RANDOM_NUM_SAMPLES;
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

TYPED_TEST_SUITE(RandomPickerTest, GPU_USAGE_TYPES);

TYPED_TEST(RandomPickerTest, PrioritizeEndGivesHigherAverage) {
    const double linear_end_prioritized = this->compute_average_picks(false, true);
    const double linear_start_prioritized = this->compute_average_picks(false, false);
    EXPECT_GT(linear_end_prioritized, linear_start_prioritized)
        << "Linear picker with prioritize_end=true should give higher average ("
        << linear_end_prioritized << ") than prioritize_end=false ("
        << linear_start_prioritized << ")";

    const double exp_end_prioritized = this->compute_average_picks(true, true);
    const double exp_start_prioritized = this->compute_average_picks(true, false);
    EXPECT_GT(exp_end_prioritized, exp_start_prioritized)
        << "Exponential picker with prioritize_end=true should give higher average ("
        << exp_end_prioritized << ") than prioritize_end=false ("
        << exp_start_prioritized << ")";
}

TYPED_TEST(RandomPickerTest, ExponentialMoreExtremeForEnd) {
    const double linear_end_prioritized = this->compute_average_picks(false, true);
    const double exp_end_prioritized = this->compute_average_picks(true, true);

    EXPECT_GT(exp_end_prioritized, linear_end_prioritized)
        << "Exponential picker with prioritize_end=true should give higher average ("
        << exp_end_prioritized << ") than Linear picker ("
        << linear_end_prioritized << ")";
}

TYPED_TEST(RandomPickerTest, ExponentialMoreExtremeForStart) {
    const double linear_start_prioritized = this->compute_average_picks(false, false);
    const double exp_start_prioritized = this->compute_average_picks(true, false);

    EXPECT_LT(exp_start_prioritized, linear_start_prioritized)
        << "Exponential picker with prioritize_end=false should give lower average ("
        << exp_start_prioritized << ") than Linear picker ("
        << linear_start_prioritized << ")";
}

TYPED_TEST(RandomPickerTest, BoundsTest) {
    const int start = 5;
    const int end = 10;
    const int num_tests = 1000;

    Buffer<double> random_nums_buf = generate_n_random_numbers(num_tests * 4, TypeParam::value);
    double* random_nums = random_nums_buf.data();

    for (int i = 0; i < num_tests; i++) {
        int linear_result = this->linear_picker.pick_random_with_provided_number(start, end, true, random_nums + i * 4);
        EXPECT_GE(linear_result, start);
        EXPECT_LT(linear_result, end);

        linear_result = this->linear_picker.pick_random_with_provided_number(start, end, false, random_nums + i * 4 + 1);
        EXPECT_GE(linear_result, start);
        EXPECT_LT(linear_result, end);

        int exp_result = this->exp_picker.pick_random_with_provided_number(start, end, true, random_nums + i * 4 + 2);
        EXPECT_GE(exp_result, start);
        EXPECT_LT(exp_result, end);

        exp_result = this->exp_picker.pick_random_with_provided_number(start, end, false, random_nums + i * 4 + 3);
        EXPECT_GE(exp_result, start);
        EXPECT_LT(exp_result, end);
    }
}

TYPED_TEST(RandomPickerTest, SingleElementRangeTest) {
    constexpr int start = 5;
    constexpr int end = 6;

    EXPECT_EQ(this->linear_picker.pick_random(start, end, true), start);
    EXPECT_EQ(this->linear_picker.pick_random(start, end, false), start);
    EXPECT_EQ(this->exp_picker.pick_random(start, end, true), start);
    EXPECT_EQ(this->exp_picker.pick_random(start, end, false), start);
}

TYPED_TEST(RandomPickerTest, TwoElementRangeDistributionTestForLinearRandomPicker) {
    const int start = 0;
    const int end = 2;
    int count_ones_end_prioritized = 0;
    int count_ones_start_prioritized = 0;
    const int num_trials = RANDOM_NUM_SAMPLES;

    Buffer<double> rand_nums_buf = generate_n_random_numbers(num_trials * 2, TypeParam::value);
    double* rand_nums = rand_nums_buf.data();

    for (int i = 0; i < num_trials; i++) {
        int result_end = this->linear_picker.pick_random_with_provided_number(start, end, true, rand_nums + i * 2);
        if (result_end == 1) {
            count_ones_end_prioritized++;
        }

        int result_start = this->linear_picker.pick_random_with_provided_number(start, end, false, rand_nums + i * 2 + 1);
        if (result_start == 1) {
            count_ones_start_prioritized++;
        }
    }

    constexpr double expected_prob_end = 2.0 / 3.0;
    constexpr double expected_prob_start = 1.0 / 3.0;

    const double actual_prob_end = static_cast<double>(count_ones_end_prioritized) / num_trials;
    const double actual_prob_start = static_cast<double>(count_ones_start_prioritized) / num_trials;

    constexpr double tolerance = 0.02;

    EXPECT_NEAR(actual_prob_end, expected_prob_end, tolerance)
        << "When prioritizing end, probability of picking 1 should be approximately "
        << expected_prob_end << " but got " << actual_prob_end;

    EXPECT_NEAR(actual_prob_start, expected_prob_start, tolerance)
        << "When prioritizing start, probability of picking 1 should be approximately "
        << expected_prob_start << " but got " << actual_prob_start;
}

TYPED_TEST(RandomPickerTest, TwoElementRangeDistributionTestForExponentialRandomPicker) {
    const int start = 0;
    const int end = 2;
    int count_ones_end_prioritized = 0;
    int count_ones_start_prioritized = 0;
    constexpr int num_trials = RANDOM_NUM_SAMPLES;

    Buffer<double> rand_nums_buf = generate_n_random_numbers(num_trials * 2, TypeParam::value);
    double* rand_nums = rand_nums_buf.data();

    for (int i = 0; i < num_trials; i++) {
        int result_end = this->exp_picker.pick_random_with_provided_number(start, end, true, rand_nums + i * 2);
        if (result_end == 1) {
            count_ones_end_prioritized++;
        }

        int result_start = this->exp_picker.pick_random_with_provided_number(start, end, false, rand_nums + i * 2 + 1);
        if (result_start == 1) {
            count_ones_start_prioritized++;
        }
    }

    const double e = std::exp(1.0);
    const double e_squared = e * e;
    const double expected_prob_end = (e - 1.0) * e / (e_squared - 1.0);
    const double expected_prob_start = (e - 1.0) / (e_squared - 1.0);

    const double actual_prob_end = static_cast<double>(count_ones_end_prioritized) / num_trials;
    const double actual_prob_start = static_cast<double>(count_ones_start_prioritized) / num_trials;

    constexpr double tolerance = 0.02;

    EXPECT_NEAR(actual_prob_end, expected_prob_end, tolerance)
        << "When prioritizing end, probability of picking 1 should be approximately "
        << expected_prob_end << " but got " << actual_prob_end;

    EXPECT_NEAR(actual_prob_start, expected_prob_start, tolerance)
        << "When prioritizing start, probability of picking 1 should be approximately "
        << expected_prob_start << " but got " << actual_prob_start;
}
