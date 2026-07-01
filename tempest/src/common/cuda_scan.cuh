#ifndef CUDA_SCAN_CUH
#define CUDA_SCAN_CUH

#ifdef HAS_CUDA

#include <cub/device/device_scan.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_partition.cuh>
#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include "error_handlers.cuh"
#include "../data/buffer.cuh"

template <typename InputIteratorT, typename OutputIteratorT>
inline void cub_inclusive_sum(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    const size_t num_items,
    const cudaStream_t stream = 0) {

    if (num_items == 0) return;

    size_t temp_bytes = 0;
    CUB_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, temp_bytes, d_in, d_out, num_items, stream));

    Buffer<uint8_t> temp(/*use_gpu=*/true);
    temp.resize(temp_bytes);

    CUB_CHECK(cub::DeviceScan::InclusiveSum(
        temp.data(), temp_bytes, d_in, d_out, num_items, stream));
}

template <typename InputIteratorT, typename OutputIteratorT>
inline void cub_exclusive_sum(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    const size_t num_items,
    const cudaStream_t stream = 0) {

    if (num_items == 0) return;

    size_t temp_bytes = 0;
    CUB_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_bytes, d_in, d_out, num_items, stream));

    Buffer<uint8_t> temp(/*use_gpu=*/true);
    temp.resize(temp_bytes);

    CUB_CHECK(cub::DeviceScan::ExclusiveSum(
        temp.data(), temp_bytes, d_in, d_out, num_items, stream));
}

template <typename KeyType, typename LengthType, typename NumRunsType>
inline void cub_run_length_encode(
    const KeyType* d_keys_in,
    KeyType* d_unique_out,
    LengthType* d_counts_out,
    NumRunsType* d_num_runs_out,
    const size_t num_items,
    const cudaStream_t stream = 0) {

    if (num_items == 0) return;

    size_t temp_bytes = 0;
    CUB_CHECK(cub::DeviceRunLengthEncode::Encode(
        nullptr, temp_bytes,
        d_keys_in, d_unique_out, d_counts_out, d_num_runs_out,
        num_items, stream));

    Buffer<uint8_t> temp(/*use_gpu=*/true);
    temp.resize(temp_bytes);

    CUB_CHECK(cub::DeviceRunLengthEncode::Encode(
        temp.data(), temp_bytes,
        d_keys_in, d_unique_out, d_counts_out, d_num_runs_out,
        num_items, stream));
}

template <typename InputT, typename FlagT, typename NumSelectedT>
inline void cub_partition_flagged(
    const InputT* d_in,
    const FlagT* d_flags,
    InputT* d_out,
    NumSelectedT* d_num_selected_out,
    const size_t num_items,
    const cudaStream_t stream = 0) {

    if (num_items == 0) return;

    size_t temp_bytes = 0;
    CUB_CHECK(cub::DevicePartition::Flagged(
        nullptr, temp_bytes,
        d_in, d_flags, d_out, d_num_selected_out,
        num_items, stream));

    Buffer<uint8_t> temp(/*use_gpu=*/true);
    temp.resize(temp_bytes);

    CUB_CHECK(cub::DevicePartition::Flagged(
        temp.data(), temp_bytes,
        d_in, d_flags, d_out, d_num_selected_out,
        num_items, stream));
}

// avoids cub::DeviceHistogram::HistogramEven's int32 overflow when num_levels^2 > 2^31.
template <typename SampleT>
inline void compute_csr_offsets_from_samples(
    const SampleT* d_samples,
    const size_t num_samples,
    size_t* d_offsets,
    const size_t num_offsets,
    const cudaStream_t stream = 0) {

    if (num_samples == 0 || num_offsets == 0) return;

    Buffer<SampleT> sorted(/*use_gpu=*/true);
    sorted.resize(num_samples);
    CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
        sorted.data(), d_samples, num_samples * sizeof(SampleT),
        cudaMemcpyDeviceToDevice, stream));

    thrust::sort(thrust::cuda::par.on(stream),
                 sorted.data(), sorted.data() + num_samples);

    thrust::lower_bound(
        thrust::cuda::par.on(stream),
        sorted.data(), sorted.data() + num_samples,
        thrust::make_counting_iterator<SampleT>(0),
        thrust::make_counting_iterator<SampleT>(static_cast<SampleT>(num_offsets)),
        d_offsets);
}

#endif  // HAS_CUDA

#endif  // CUDA_SCAN_CUH
