#include "RandomPicker.cuh"

#include <common/random_gen.cuh>

#include "../common/error_handlers.cuh"
#include "../src/random/pickers.cuh"
#include "../src/common/setup.cuh"

#ifdef HAS_CUDA

__global__ void pick_exponential_random_number_cuda_kernel(
    int* result,
    const int start,
    const int end,
    const bool prioritize_end,
    const double* rand_nums) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        *result = random_pickers::pick_random_exponential_index(start, end, prioritize_end, rand_nums[idx]);
    }
}

#endif

ExponentialIndexRandomPicker::ExponentialIndexRandomPicker(const bool use_gpu) : use_gpu(use_gpu) {}

int ExponentialIndexRandomPicker::pick_random(const int start, const int end, const bool prioritize_end) const {
    Buffer<double> rand_nums = generate_n_random_numbers(1, use_gpu);
    int result;

    #ifdef HAS_CUDA
    if (use_gpu) {
        int* d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(int)));

        pick_exponential_random_number_cuda_kernel<<<1, 1>>>(d_result, start, end, prioritize_end, rand_nums.data());
        CUDA_KERNEL_CHECK("After pick_exponential_random_number_cuda_kernel execution");

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
    }
    else
    #endif
    {
        result = random_pickers::pick_random_exponential_index(start, end, prioritize_end, rand_nums.data()[0]);
    }

    return result;
}

int ExponentialIndexRandomPicker::pick_random_with_provided_number(const int start, const int end, const bool prioritize_end, const double* rand_nums) const {
    int result;

    #ifdef HAS_CUDA
    if (use_gpu) {
        int* d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(int)));

        pick_exponential_random_number_cuda_kernel<<<1, 1>>>(d_result, start, end, prioritize_end, rand_nums);
        CUDA_KERNEL_CHECK("After pick_exponential_random_number_cuda_kernel execution");

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
    }
    else
    #endif
    {
        result = random_pickers::pick_random_exponential_index(start, end, prioritize_end, rand_nums[0]);
    }

    return result;
}

#ifdef HAS_CUDA

__global__ void pick_linear_random_number_cuda_kernel(
    int* result,
    const int start,
    const int end,
    const bool prioritize_end,
    const double* rand_nums) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        *result = random_pickers::pick_random_linear(start, end, prioritize_end, rand_nums[idx]);
    }
}

#endif

LinearRandomPicker::LinearRandomPicker(const bool use_gpu) : use_gpu(use_gpu) {}

int LinearRandomPicker::pick_random(const int start, const int end, const bool prioritize_end) const {
    Buffer<double> rand_nums = generate_n_random_numbers(1, use_gpu);
    int result;

    #ifdef HAS_CUDA
    if (use_gpu) {
        int* d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(int)));

        pick_linear_random_number_cuda_kernel<<<1, 1>>>(d_result, start, end, prioritize_end, rand_nums.data());
        CUDA_KERNEL_CHECK("After pick_linear_random_number_cuda_kernel execution");

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
    }
    else
    #endif
    {
        result = random_pickers::pick_random_linear(start, end, prioritize_end, rand_nums.data()[0]);
    }

    return result;
}

int LinearRandomPicker::pick_random_with_provided_number(const int start, const int end, const bool prioritize_end, const double* rand_nums) const {
    int result;

    #ifdef HAS_CUDA
    if (use_gpu) {
        int* d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(int)));

        pick_linear_random_number_cuda_kernel<<<1, 1>>>(d_result, start, end, prioritize_end, rand_nums);
        CUDA_KERNEL_CHECK("After pick_linear_random_number_cuda_kernel execution");

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
    }
    else
    #endif
    {
        result = random_pickers::pick_random_linear(start, end, prioritize_end, rand_nums[0]);
    }

    return result;
}

#ifdef HAS_CUDA
__global__ void pick_uniform_random_number_cuda_kernel(
    int* result,
    const int start,
    const int end,
    const double* rand_nums) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        *result = random_pickers::pick_random_uniform(start, end, rand_nums[idx]);
    }
}
#endif

UniformRandomPicker::UniformRandomPicker(const bool use_gpu) : use_gpu(use_gpu) {}

int UniformRandomPicker::pick_random(const int start, const int end, const bool /* prioritize_end */) const {
    Buffer<double> rand_nums = generate_n_random_numbers(1, use_gpu);
    int result;

    #ifdef HAS_CUDA
    if (use_gpu) {
        int* d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(int)));

        pick_uniform_random_number_cuda_kernel<<<1, 1>>>(d_result, start, end, rand_nums.data());
        CUDA_KERNEL_CHECK("After pick_uniform_random_number_cuda_kernel execution");

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
    }
    else
    #endif
    {
        result = random_pickers::pick_random_uniform(start, end, rand_nums.data()[0]);
    }

    return result;
}

int UniformRandomPicker::pick_random_with_provided_number(const int start, const int end, const bool /* prioritize_end */, const double* rand_nums) const {
    int result;

    #ifdef HAS_CUDA
    if (use_gpu) {
        int* d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(int)));

        pick_uniform_random_number_cuda_kernel<<<1, 1>>>(d_result, start, end, rand_nums);
        CUDA_KERNEL_CHECK("After pick_uniform_random_number_cuda_kernel execution");

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
    }
    else
    #endif
    {
        result = random_pickers::pick_random_uniform(start, end, rand_nums[0]);
    }

    return result;
}


#ifdef HAS_CUDA
__global__ void pick_weighted_random_number_cuda_kernel(
    int* result,
    double* weights,
    const size_t weights_size,
    const size_t group_start,
    const size_t group_end,
    const double* rand_nums) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        *result = random_pickers::pick_random_exponential_weights(
            weights,
            weights_size,
            group_start,
            group_end,
            rand_nums[idx]);
    }
}
#endif

WeightBasedRandomPicker::WeightBasedRandomPicker(const bool use_gpu) : use_gpu(use_gpu) {}

int WeightBasedRandomPicker::pick_random(const double* weights, const size_t weights_size, const size_t group_start, const size_t group_end) const {
    Buffer<double> rand_nums = generate_n_random_numbers(1, use_gpu);
    int result;

    #ifdef HAS_CUDA
    if (use_gpu) {
        double* d_weights;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_weights, weights_size * sizeof(double)));

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_weights, weights, weights_size * sizeof(double), cudaMemcpyHostToDevice));

        int* d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(int)));

        pick_weighted_random_number_cuda_kernel<<<1, 1>>>(d_result, d_weights, weights_size, group_start, group_end, rand_nums.data());
        CUDA_KERNEL_CHECK("After pick_weighted_random_number_cuda_kernel execution");

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_weights));
    }
    else
    #endif
    {
        result = random_pickers::pick_random_exponential_weights(
            const_cast<double*>(weights),
            weights_size,
            group_start,
            group_end,
            rand_nums.data()[0]);
    }

    return result;
}

int WeightBasedRandomPicker::pick_random_with_provided_number(const double* weights, const size_t weights_size, const size_t group_start, const size_t group_end, const double* rand_nums) const {
    int result;

    #ifdef HAS_CUDA
    if (use_gpu) {
        double* d_weights;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_weights, weights_size * sizeof(double)));

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_weights, weights, weights_size * sizeof(double), cudaMemcpyHostToDevice));

        int* d_result;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(int)));

        pick_weighted_random_number_cuda_kernel<<<1, 1>>>(d_result, d_weights, weights_size, group_start, group_end, rand_nums);
        CUDA_KERNEL_CHECK("After pick_weighted_random_number_cuda_kernel execution");

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_weights));
    }
    else
    #endif
    {
        result = random_pickers::pick_random_exponential_weights(
            const_cast<double*>(weights),
            weights_size,
            group_start,
            group_end,
            rand_nums[0]);
    }

    return result;
}

int WeightBasedRandomPicker::pick_random(const std::vector<double>& cumulative_weights, const int group_start, const int group_end) const {
    return pick_random(cumulative_weights.data(), cumulative_weights.size(),
                     static_cast<size_t>(group_start), static_cast<size_t>(group_end));
}

int WeightBasedRandomPicker::pick_random_with_provided_number(const std::vector<double>& cumulative_weights, const int group_start, const int group_end, const double* rand_nums) const {
    return pick_random_with_provided_number(cumulative_weights.data(), cumulative_weights.size(),
                     static_cast<size_t>(group_start), static_cast<size_t>(group_end), rand_nums);
}
