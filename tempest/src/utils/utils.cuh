#ifndef UTILS_H
#define UTILS_H

#ifdef HAS_CUDA
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#endif

#include <cstddef>

#include "../data/structs.cuh"
#include "../data/buffer.cuh"
#include "../common/macros.cuh"
#include "../common/cuda_config.cuh"
#include "../common/error_handlers.cuh"

HOST inline DataBlock<int> repeat_elements(const DataBlock<int>& arr, int times, const bool use_gpu) {
    const size_t input_size = arr.size;
    const size_t output_size = input_size * times;

    DataBlock<int> repeated_items(output_size, use_gpu);
    if (input_size == 0 || times <= 0) {
        return repeated_items;
    }

    #ifdef HAS_CUDA
    if (use_gpu) {
        const int* arr_ptr = arr.data;

        thrust::transform(
            DEVICE_EXECUTION_POLICY,
            thrust::counting_iterator(0),
            thrust::counting_iterator<int>(output_size),
            thrust::device_pointer_cast(repeated_items.data),
            [arr_ptr, times] DEVICE (const int idx) {
                const int original_idx = idx / times;
                return arr_ptr[original_idx];
            }
        );

        CUDA_KERNEL_CHECK("After thrust transform in repeat_elements");
    }
    else
    #endif
    {
        for (size_t i = 0; i < input_size; ++i) {
            for (int j = 0; j < times; ++j) {
                repeated_items.data[i * times + j] = arr.data[i];
            }
        }
    }

    return repeated_items;
}

// Element-type-generic: each of `input_size` elements is repeated `times`
// contiguously (seed-major). Used for per-walk node ids (int) and per-walk
// start-time cutoffs (int64_t), which must fan out with identical layout so a
// shared-seed co-shuffle keeps them aligned.
template <typename T>
HOST inline DataBlock<T> repeat_elements(const T* arr, const size_t input_size, int times, const bool use_gpu) {
    const size_t output_size = input_size * times;

    DataBlock<T> repeated_items(output_size, use_gpu);
    if (input_size == 0 || times <= 0) {
        return repeated_items;
    }

    #ifdef HAS_CUDA
    if (use_gpu) {
        Buffer<T> d_arr_buf(input_size, true);
        T* d_arr = d_arr_buf.data();
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(d_arr, arr, input_size * sizeof(T), cudaMemcpyHostToDevice));

        thrust::transform(
            DEVICE_EXECUTION_POLICY,
            thrust::counting_iterator(0),
            thrust::counting_iterator<int>(output_size),
            thrust::device_pointer_cast(repeated_items.data),
            [d_arr, times] DEVICE (const int idx) {
                const int original_idx = idx / times;
                return d_arr[original_idx];
            }
        );

        CUDA_KERNEL_CHECK("After thrust transform in repeat_elements (vector)");
    }
    else
    #endif
    {
        for (size_t i = 0; i < input_size; ++i) {
            for (int j = 0; j < times; ++j) {
                repeated_items.data[i * times + j] = arr[i];
            }
        }
    }

    return repeated_items;
}

HOST DEVICE inline int pick_other_number(const int first, const int second, const int picked_number) {
    return (picked_number == first) ? second : first;
}

HOST DEVICE inline size_t next_power_of_two(const size_t n) {
    if ((n & (n - 1)) == 0) {
        return n;
    }

    size_t power = 1;
    while (power < n) {
        power <<= 1;
    }

    return power;
}

#endif // UTILS_H
