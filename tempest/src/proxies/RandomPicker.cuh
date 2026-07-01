#ifndef RANDOM_PICKER_H
#define RANDOM_PICKER_H

#include <vector>
#include "../src/common/setup.cuh"

#ifdef HAS_CUDA

__global__ void pick_exponential_random_number_cuda_kernel(
    int* result,
    int start,
    int end,
    bool prioritize_end,
    const double* rand_nums);

#endif

class ExponentialIndexRandomPicker {
    bool use_gpu;

public:

    explicit ExponentialIndexRandomPicker(bool use_gpu);

    int pick_random(int start, int end, bool prioritize_end) const;

    int pick_random_with_provided_number(int start, int end, bool prioritize_end, const double* rand_nums) const;
};

#ifdef HAS_CUDA

__global__ void pick_linear_random_number_cuda_kernel(
    int* result,
    int start,
    int end,
    bool prioritize_end,
    const double* rand_nums);

#endif

class LinearRandomPicker {

    bool use_gpu;

public:

    explicit LinearRandomPicker(bool use_gpu);

    int pick_random(int start, int end, bool prioritize_end) const;

    int pick_random_with_provided_number(int start, int end, bool prioritize_end, const double* rand_nums) const;
};

#ifdef HAS_CUDA

__global__ void pick_uniform_random_number_cuda_kernel(
    int* result,
    int start,
    int end,
    const double* rand_nums);

#endif

class UniformRandomPicker {
    bool use_gpu;

public:
    explicit UniformRandomPicker(bool use_gpu);

    int pick_random(int start, int end, bool /* prioritize_end */) const;

    int pick_random_with_provided_number(int start, int end, bool /* prioritize_end */, const double* rand_nums) const;
};

#ifdef HAS_CUDA

__global__ void pick_weighted_random_number_cuda_kernel(
    int* result,
    double* weights,
    size_t weights_size,
    size_t group_start,
    size_t group_end,
    const double* rand_nums);

#endif

class WeightBasedRandomPicker {
    bool use_gpu;

public:
    explicit WeightBasedRandomPicker(bool use_gpu);

    int pick_random(const double* weights, size_t weights_size, size_t group_start, size_t group_end) const;

    int pick_random_with_provided_number(const double* weights, size_t weights_size, size_t group_start, size_t group_end, const double* rand_nums) const;

    int pick_random(const std::vector<double>& cumulative_weights, int group_start, int group_end) const;

    int pick_random_with_provided_number(const std::vector<double>& cumulative_weights, int group_start, int group_end, const double* rand_nums) const;
};

#endif // RANDOM_PICKER_H
