#ifndef PICKERS_H
#define PICKERS_H

#include <cstddef>
#include <cmath>
#ifdef HAS_CUDA
#include <curand_kernel.h>
#endif

#include "../common/macros.cuh"
#include "../data/enums.cuh"

namespace random_pickers {

    HOST DEVICE inline int pick_random_linear(
        const int start,
        const int end,
        const bool prioritize_end,
        const double rand_number) {
        if (start >= end) {
            return -1;
        }

        const int len_seq = end - start;

        // arithmetic-weight CDF inverted via quadratic i^2 + i - 2r = 0.
        const double total_weight = static_cast<double>(len_seq) *
                                       (static_cast<double>(len_seq) + 1.0) / 2.0;
        const double scaled_random_value = total_weight * rand_number;

        const double discriminant = 1.0 + 8.0 * scaled_random_value;
        const double root = (-1.0 + std::sqrt(discriminant)) / 2.0;
        const int index = static_cast<int>(std::floor(root));

        if (prioritize_end) {
            return start + std::min(index, len_seq - 1);
        } else {
            const int revered_index = len_seq - 1 - index;
            return start + std::max(0, revered_index);
        }
    }

    HOST DEVICE inline int pick_random_exponential_index(
        const int start,
        const int end,
        const bool prioritize_end,
        const double rand_number) {
        if (start >= end) {
            return -1;
        }

        const int len_seq = end - start;

        double k;
        if (len_seq < 710) {
            k = log1p(rand_number * expm1(len_seq)) - 1;
        } else {
            // expm1 overflows; fall back to large-x approximation.
            k = len_seq + std::log(rand_number) - 1;
        }

        // +1 corrects the inverse-CDF rounding bias before clamping.
        const int rounded_index = std::max(0, std::min(static_cast<int>(k + 1), len_seq - 1));

        if (prioritize_end) {
            return start + rounded_index;
        } else {
            return start + (len_seq - 1 - rounded_index);
        }
    }

    HOST DEVICE inline int pick_random_uniform(const int start, const int end, const double rand_number) {
        if (start >= end) {
            return -1;
        }

        return start + static_cast<int>(rand_number * (end - start));
    }

    // slice_start: start of the piecewise-CDF segment containing [group_start, group_end).
    // per-node callers pass node_group_begin so the prefix at a node boundary is 0
    // rather than the previous segment's total. smem_weights, when set, redirects
    // weights[i] reads to smem_weights[i - slice_start].
    HOST DEVICE inline int pick_random_exponential_weights(
        const double* weights,
        const size_t weights_size,
        const size_t group_start,
        const size_t group_end,
        const double random_number,
        const size_t slice_start = 0,
        const double* smem_weights = nullptr) {
        if (group_start >= group_end || group_end > weights_size) {
            return -1;
        }

        // branch on a kernel-uniform pointer; compiler hoists it out of the loop.
        auto w_at = [&](const size_t pos) -> double {
            return smem_weights != nullptr
                ? smem_weights[pos - slice_start]
                : weights[pos];
        };

        double start_sum = 0.0;
        if (group_start > slice_start) {
            start_sum = w_at(group_start - 1);
        }
        const double end_sum = w_at(group_end - 1);

        if (end_sum < start_sum) {
            return -1;
        }

        const double random_val = start_sum + random_number * (end_sum - start_sum);

        size_t lo = group_start;
        size_t hi = group_end;
        while (lo < hi) {
            const size_t mid = lo + ((hi - lo) >> 1);
            if (w_at(mid) < random_val) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return static_cast<int>(lo);
    }

    HOST DEVICE constexpr inline bool is_index_based_picker(const RandomPickerType picker_type) {
        return picker_type == RandomPickerType::Linear
            || picker_type == RandomPickerType::Uniform
            || picker_type == RandomPickerType::ExponentialIndex
            || picker_type == RandomPickerType::TEST_FIRST
            || picker_type == RandomPickerType::TEST_LAST;
    }

    HOST DEVICE inline int pick_using_index_based_picker(
        const RandomPickerType random_picker,
        const int start,
        const int end,
        const bool prioritize_end,
        const double random_number) {
        switch (random_picker) {
            case RandomPickerType::Linear:
                return pick_random_linear(start, end, prioritize_end, random_number);
            case RandomPickerType::ExponentialIndex:
                return pick_random_exponential_index(start, end, prioritize_end, random_number);
            case RandomPickerType::Uniform:
                return pick_random_uniform(start, end, random_number);
            // ONLY FOR TEST
            case RandomPickerType::TEST_FIRST:
                return start;
            case RandomPickerType::TEST_LAST:
                return end - 1;
            default:
                return -1;
        }
    }

    HOST DEVICE inline int pick_using_weight_based_picker(
        const RandomPickerType random_picker,
        const double* weights,
        const size_t weights_size,
        const size_t group_start,
        const size_t group_end,
        const double random_number,
        const size_t slice_start = 0,
        const double* smem_weights = nullptr) {
        if (random_picker != RandomPickerType::ExponentialWeight &&
            random_picker != RandomPickerType::TemporalNode2Vec) {
            return -1;
        }
        return pick_random_exponential_weights(
            weights, weights_size, group_start, group_end, random_number,
            slice_start, smem_weights);
    }
}

#endif // PICKERS_H
