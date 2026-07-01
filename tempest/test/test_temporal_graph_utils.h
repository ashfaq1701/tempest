#ifndef TEST_TEMPORAL_GRAPH_UTILS_H
#define TEST_TEMPORAL_GRAPH_UTILS_H

#include <stdexcept>
#include <vector>

#include "../src/core/tempest.cuh"
#include "../src/data/structs.cuh"
#include "../src/data/enums.cuh"
#include "../src/data/temporal_graph_data.cuh"
#include "../src/data/temporal_graph_view.cuh"
#include "../src/graph/edge_selectors.cuh"
#include "../src/graph/temporal_graph.cuh"
#include "../src/graph/temporal_node2vec_helpers.cuh"
#include "../src/common/error_handlers.cuh"
#include "../src/common/random_gen.cuh"

#ifdef HAS_CUDA
#include <cuda_runtime.h>

namespace test_util_detail {

template <bool Forward, RandomPickerType PickerType>
__global__ void get_edge_at_kernel(
    Edge* result, TemporalGraphView view,
    const int64_t timestamp, const double* rand_nums) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = temporal_graph::get_edge_at_device<Forward, PickerType>(
            view, timestamp, rand_nums[0], rand_nums[1]);
    }
}

template <bool IsDirected, bool Forward, RandomPickerType PickerType>
__global__ void get_node_edge_at_kernel(
    Edge* result, TemporalGraphView view, const int node_id,
    const int64_t timestamp, const int64_t cutoff, const int prev_node,
    const double* rand_nums) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = temporal_graph::get_node_edge_at_device<Forward, PickerType, IsDirected>(
            view, node_id, timestamp, cutoff, prev_node, rand_nums[0], rand_nums[1]);
    }
}

__global__ inline void compute_node2vec_beta_kernel(
    double* result, TemporalGraphView view, const int prev_node, const int w) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = temporal_graph::compute_node2vec_beta_device(view, prev_node, w);
    }
}

} // namespace test_util_detail
#endif

namespace test_util {

inline Edge get_edge_at_with_provided_nums(
    const TemporalGraphData& data,
    const RandomPickerType picker_type,
    const double* rand_nums,
    const int64_t timestamp = -1,
    const bool forward = true) {
    const TemporalGraphView view = make_temporal_graph_view(data);
    Edge result{};

    #define DISPATCH_HOST(FWD, PICKER) \
        result = temporal_graph::get_edge_at_host<FWD, PICKER>( \
            view, timestamp, rand_nums[0], rand_nums[1]); break;

#ifdef HAS_CUDA
    #define DISPATCH_DEVICE(FWD, PICKER) \
        test_util_detail::get_edge_at_kernel<FWD, PICKER><<<1, 1>>>( \
            d_result, view, timestamp, rand_nums); \
        CUDA_KERNEL_CHECK("get_edge_at_kernel"); break;
#endif

    #define HANDLE_PICKER_HOST(FWD) \
        switch (picker_type) { \
            case RandomPickerType::Uniform:           DISPATCH_HOST(FWD, RandomPickerType::Uniform) \
            case RandomPickerType::Linear:            DISPATCH_HOST(FWD, RandomPickerType::Linear) \
            case RandomPickerType::ExponentialIndex:  DISPATCH_HOST(FWD, RandomPickerType::ExponentialIndex) \
            case RandomPickerType::ExponentialWeight: DISPATCH_HOST(FWD, RandomPickerType::ExponentialWeight) \
            case RandomPickerType::TemporalNode2Vec:  DISPATCH_HOST(FWD, RandomPickerType::TemporalNode2Vec) \
            case RandomPickerType::TEST_FIRST:        DISPATCH_HOST(FWD, RandomPickerType::TEST_FIRST) \
            case RandomPickerType::TEST_LAST:         DISPATCH_HOST(FWD, RandomPickerType::TEST_LAST) \
            default: break; \
        }

#ifdef HAS_CUDA
    #define HANDLE_PICKER_DEVICE(FWD) \
        switch (picker_type) { \
            case RandomPickerType::Uniform:           DISPATCH_DEVICE(FWD, RandomPickerType::Uniform) \
            case RandomPickerType::Linear:            DISPATCH_DEVICE(FWD, RandomPickerType::Linear) \
            case RandomPickerType::ExponentialIndex:  DISPATCH_DEVICE(FWD, RandomPickerType::ExponentialIndex) \
            case RandomPickerType::ExponentialWeight: DISPATCH_DEVICE(FWD, RandomPickerType::ExponentialWeight) \
            case RandomPickerType::TemporalNode2Vec:  DISPATCH_DEVICE(FWD, RandomPickerType::TemporalNode2Vec) \
            case RandomPickerType::TEST_FIRST:        DISPATCH_DEVICE(FWD, RandomPickerType::TEST_FIRST) \
            case RandomPickerType::TEST_LAST:         DISPATCH_DEVICE(FWD, RandomPickerType::TEST_LAST) \
            default: break; \
        }

    if (data.use_gpu) {
        Edge* d_result = nullptr;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(Edge)));
        if (forward) { HANDLE_PICKER_DEVICE(true) } else { HANDLE_PICKER_DEVICE(false) }
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(Edge), cudaMemcpyDeviceToHost));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
    } else
#endif
    {
        if (forward) { HANDLE_PICKER_HOST(true) } else { HANDLE_PICKER_HOST(false) }
    }

    #undef DISPATCH_HOST
    #undef HANDLE_PICKER_HOST
#ifdef HAS_CUDA
    #undef DISPATCH_DEVICE
    #undef HANDLE_PICKER_DEVICE
#endif

    return result;
}

inline Edge get_edge_at(
    const TemporalGraphData& data,
    const RandomPickerType picker_type,
    const int64_t timestamp = -1,
    const bool forward = true) {
    Buffer<double> rand_nums_buf = generate_n_random_numbers(2, data.use_gpu);
    const Edge result = get_edge_at_with_provided_nums(
        data, picker_type, rand_nums_buf.data(), timestamp, forward);
    return result;
}

inline Edge get_node_edge_at(
    const TemporalGraphData& data,
    const int node_id,
    const RandomPickerType picker_type,
    const int64_t timestamp,
    const int prev_node,
    const bool forward = true,
    const int64_t cutoff = NO_WALK_CUTOFF) {
    const TemporalGraphView view = make_temporal_graph_view(data);
    Edge result{};
    const bool is_directed = data.is_directed;

    Buffer<double> rand_nums_buf = generate_n_random_numbers(2, data.use_gpu);
    double* rand_nums = rand_nums_buf.data();

    #define DISPATCH_HOST(FWD, PICKER, DIR) \
        result = temporal_graph::get_node_edge_at_host<FWD, PICKER, DIR>( \
            view, node_id, timestamp, cutoff, prev_node, rand_nums[0], rand_nums[1]); break;

#ifdef HAS_CUDA
    #define DISPATCH_DEVICE(FWD, PICKER, DIR) \
        test_util_detail::get_node_edge_at_kernel<DIR, FWD, PICKER><<<1, 1>>>( \
            d_result, view, node_id, timestamp, cutoff, prev_node, rand_nums); \
        CUDA_KERNEL_CHECK("get_node_edge_at_kernel"); break;
#endif

    #define HANDLE_PICKER_HOST(FWD, DIR) \
        switch (picker_type) { \
            case RandomPickerType::Uniform:           DISPATCH_HOST(FWD, RandomPickerType::Uniform,           DIR) \
            case RandomPickerType::Linear:            DISPATCH_HOST(FWD, RandomPickerType::Linear,            DIR) \
            case RandomPickerType::ExponentialIndex:  DISPATCH_HOST(FWD, RandomPickerType::ExponentialIndex,  DIR) \
            case RandomPickerType::ExponentialWeight: DISPATCH_HOST(FWD, RandomPickerType::ExponentialWeight, DIR) \
            case RandomPickerType::TemporalNode2Vec:  DISPATCH_HOST(FWD, RandomPickerType::TemporalNode2Vec,  DIR) \
            case RandomPickerType::TEST_FIRST:        DISPATCH_HOST(FWD, RandomPickerType::TEST_FIRST,        DIR) \
            case RandomPickerType::TEST_LAST:         DISPATCH_HOST(FWD, RandomPickerType::TEST_LAST,         DIR) \
            default: break; \
        }

#ifdef HAS_CUDA
    #define HANDLE_PICKER_DEVICE(FWD, DIR) \
        switch (picker_type) { \
            case RandomPickerType::Uniform:           DISPATCH_DEVICE(FWD, RandomPickerType::Uniform,           DIR) \
            case RandomPickerType::Linear:            DISPATCH_DEVICE(FWD, RandomPickerType::Linear,            DIR) \
            case RandomPickerType::ExponentialIndex:  DISPATCH_DEVICE(FWD, RandomPickerType::ExponentialIndex,  DIR) \
            case RandomPickerType::ExponentialWeight: DISPATCH_DEVICE(FWD, RandomPickerType::ExponentialWeight, DIR) \
            case RandomPickerType::TemporalNode2Vec:  DISPATCH_DEVICE(FWD, RandomPickerType::TemporalNode2Vec,  DIR) \
            case RandomPickerType::TEST_FIRST:        DISPATCH_DEVICE(FWD, RandomPickerType::TEST_FIRST,        DIR) \
            case RandomPickerType::TEST_LAST:         DISPATCH_DEVICE(FWD, RandomPickerType::TEST_LAST,         DIR) \
            default: break; \
        }

    if (data.use_gpu) {
        Edge* d_result = nullptr;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(Edge)));
        if (is_directed) {
            if (forward) { HANDLE_PICKER_DEVICE(true,  true)  } else { HANDLE_PICKER_DEVICE(false, true)  }
        } else {
            if (forward) { HANDLE_PICKER_DEVICE(true,  false) } else { HANDLE_PICKER_DEVICE(false, false) }
        }
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(Edge), cudaMemcpyDeviceToHost));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
    } else
#endif
    {
        if (is_directed) {
            if (forward) { HANDLE_PICKER_HOST(true,  true)  } else { HANDLE_PICKER_HOST(false, true)  }
        } else {
            if (forward) { HANDLE_PICKER_HOST(true,  false) } else { HANDLE_PICKER_HOST(false, false) }
        }
    }

    #undef DISPATCH_HOST
    #undef HANDLE_PICKER_HOST
#ifdef HAS_CUDA
    #undef DISPATCH_DEVICE
    #undef HANDLE_PICKER_DEVICE
#endif

    return result;
}

inline double compute_node2vec_beta(
    const TemporalGraphData& data, const int prev_node, const int w) {
    const TemporalGraphView view = make_temporal_graph_view(data);
#ifdef HAS_CUDA
    if (data.use_gpu) {
        double* d_result = nullptr;
        CUDA_CHECK_AND_CLEAR(cudaMalloc(&d_result, sizeof(double)));
        test_util_detail::compute_node2vec_beta_kernel<<<1, 1>>>(
            d_result, view, prev_node, w);
        CUDA_KERNEL_CHECK("compute_node2vec_beta_kernel");
        double result;
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK_AND_CLEAR(cudaFree(d_result));
        return result;
    }
#endif
    return temporal_graph::compute_node2vec_beta_host(view, prev_node, w);
}

inline size_t count_timestamps_less_than(
    const TemporalGraphData& data, const int64_t timestamp) {
#ifdef HAS_CUDA
    if (data.use_gpu) return temporal_graph::count_timestamps_less_than_cuda(data, timestamp);
#endif
    return temporal_graph::count_timestamps_less_than_std(data, timestamp);
}

inline size_t count_timestamps_greater_than(
    const TemporalGraphData& data, const int64_t timestamp) {
#ifdef HAS_CUDA
    if (data.use_gpu) return temporal_graph::count_timestamps_greater_than_cuda(data, timestamp);
#endif
    return temporal_graph::count_timestamps_greater_than_std(data, timestamp);
}

inline size_t count_node_timestamps_less_than(
    const TemporalGraphData& data, const int node_id, const int64_t timestamp) {
#ifdef HAS_CUDA
    if (data.use_gpu) return temporal_graph::count_node_timestamps_less_than_cuda(data, node_id, timestamp);
#endif
    return temporal_graph::count_node_timestamps_less_than_std(data, node_id, timestamp);
}

inline size_t count_node_timestamps_greater_than(
    const TemporalGraphData& data, const int node_id, const int64_t timestamp) {
#ifdef HAS_CUDA
    if (data.use_gpu) return temporal_graph::count_node_timestamps_greater_than_cuda(data, node_id, timestamp);
#endif
    return temporal_graph::count_node_timestamps_greater_than_std(data, node_id, timestamp);
}

inline void add_edges(
    core::Tempest& trw,
    const std::vector<Edge>& edges) {
    std::vector<int> srcs; srcs.reserve(edges.size());
    std::vector<int> tgts; tgts.reserve(edges.size());
    std::vector<int64_t> ts; ts.reserve(edges.size());
    for (const auto& e : edges) {
        srcs.push_back(e.u);
        tgts.push_back(e.i);
        ts.push_back(e.ts);
    }
    trw.add_multiple_edges(srcs.data(), tgts.data(), ts.data(), srcs.size());
}

} // namespace test_util

#endif // TEST_TEMPORAL_GRAPH_UTILS_H
