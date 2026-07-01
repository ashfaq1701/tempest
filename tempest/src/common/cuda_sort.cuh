#ifndef CUDA_SORT_CUH
#define CUDA_SORT_CUH

#ifdef HAS_CUDA

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include "error_handlers.cuh"
#include "../data/buffer.cuh"

template <typename KeyType, typename ValueType>
inline void cub_sort_pairs(
    const KeyType* d_keys_in,
    KeyType* d_keys_out,
    const ValueType* d_values_in,
    ValueType* d_values_out,
    const size_t num_items,
    const cudaStream_t stream = 0) {

    if (num_items == 0) return;

    size_t temp_bytes = 0;
    CUB_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        static_cast<int>(num_items),
        /*begin_bit=*/0, /*end_bit=*/static_cast<int>(sizeof(KeyType) * 8),
        stream));

    Buffer<uint8_t> temp(/*use_gpu=*/true);
    temp.resize(temp_bytes);

    CUB_CHECK(cub::DeviceRadixSort::SortPairs(
        temp.data(), temp_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        static_cast<int>(num_items),
        /*begin_bit=*/0, /*end_bit=*/static_cast<int>(sizeof(KeyType) * 8),
        stream));
}

#endif  // HAS_CUDA

#endif  // CUDA_SORT_CUH
