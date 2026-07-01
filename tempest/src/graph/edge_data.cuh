#ifndef EDGE_DATA_CUH
#define EDGE_DATA_CUH

#include <cstddef>
#include <vector>
#include <algorithm>

#ifdef HAS_CUDA
#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/upper_bound.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#endif

#include "../common/macros.cuh"
#include "../common/error_handlers.cuh"
#include "../common/cuda_config.cuh"
#include "../data/structs.cuh"
#include "../data/temporal_graph_data.cuh"
#include "../data/temporal_graph_view.cuh"
#include "../data/buffer.cuh"

namespace edge_data {

    HOST DEVICE size_t size(const TemporalGraphData& data);
    HOST void set_size(TemporalGraphData& data, size_t size);
    HOST bool empty(const TemporalGraphData& data);

    HOST void add_edges(
        TemporalGraphData& data,
        const int* sources,
        const int* targets,
        const int64_t* timestamps,
        size_t num_new_edges);

    HOST void add_edges(
        TemporalGraphData& data,
        const int* sources,
        const int* targets,
        const int64_t* timestamps,
        size_t num_new_edges,
        const float* edge_features,
        size_t feature_dim);

    HOST inline void push_back(TemporalGraphData& data,
                               const int src, const int tgt, const int64_t ts) {
        const int srcs[1]     = {src};
        const int tgts[1]     = {tgt};
        const int64_t times[1] = {ts};
        add_edges(data, srcs, tgts, times, 1);
    }

    HOST std::vector<Edge> get_edges(const TemporalGraphData& data);

    HOST std::vector<int> get_active_node_ids(const TemporalGraphData& data);

    HOST size_t active_node_count(const TemporalGraphData& data);

    HOST DEVICE inline int get_max_node_id(const TemporalGraphData& data) {
        return data.max_node_id;
    }

    // host-safe: dispatches on data.use_gpu, copies scalars D->H as needed
    HOST inline SizeRange get_timestamp_group_range(
        const TemporalGraphData& data, const size_t group_idx) {
        if (group_idx >= data.unique_timestamps.size()) {
            return SizeRange{0, 0};
        }
        const size_t* offsets = data.timestamp_group_offsets.data();
        return SizeRange{
            read_one_host_safe(offsets + group_idx,     data.use_gpu),
            read_one_host_safe(offsets + group_idx + 1, data.use_gpu),
        };
    }

    HOST inline size_t get_timestamp_group_count(const TemporalGraphData& data) {
        return data.unique_timestamps.size();
    }

    HOST inline size_t find_group_after_timestamp(
        const TemporalGraphData& data, const int64_t timestamp) {
        const size_t n = data.unique_timestamps.size();
        if (n == 0) return 0;

        const int64_t* begin = data.unique_timestamps.data();
#ifdef HAS_CUDA
        if (data.use_gpu) {
            const auto it = thrust::upper_bound(
                DEVICE_EXECUTION_POLICY,
                thrust::device_pointer_cast(begin),
                thrust::device_pointer_cast(begin + n),
                timestamp);
            return it - thrust::device_pointer_cast(begin);
        }
#endif
        const int64_t* end = begin + n;
        const auto it = std::upper_bound(begin, end, timestamp);
        return it - begin;
    }

    HOST inline size_t find_group_before_timestamp(
        const TemporalGraphData& data, const int64_t timestamp) {
        const size_t n = data.unique_timestamps.size();
        if (n == 0) return 0;

        const int64_t* begin = data.unique_timestamps.data();
#ifdef HAS_CUDA
        if (data.use_gpu) {
            const auto it = thrust::lower_bound(
                DEVICE_EXECUTION_POLICY,
                thrust::device_pointer_cast(begin),
                thrust::device_pointer_cast(begin + n),
                timestamp);
            return (it - thrust::device_pointer_cast(begin)) - 1;
        }
#endif
        const int64_t* end = begin + n;
        const auto it = std::lower_bound(begin, end, timestamp);
        return (it - begin) - 1;
    }

    HOST bool is_node_active(const TemporalGraphData& data, int node_id);

    HOST void populate_active_nodes_std(TemporalGraphData& data);

    HOST void build_node_adjacency_csr_std(TemporalGraphData& data);

    HOST void update_timestamp_groups_std(TemporalGraphData& data);

    HOST void update_temporal_weights_std(
        TemporalGraphData& data, double timescale_bound);

#ifdef HAS_CUDA
    HOST void populate_active_nodes_cuda(TemporalGraphData& data);

    HOST void build_node_adjacency_csr_cuda(TemporalGraphData& data);

    HOST void update_timestamp_groups_cuda(TemporalGraphData& data);

    HOST void update_temporal_weights_cuda(
        TemporalGraphData& data, double timescale_bound);

    DEVICE inline size_t find_group_after_timestamp_device(
        const TemporalGraphView& view, const int64_t timestamp) {
        if (view.num_groups == 0) return 0;

        const int64_t* begin = view.unique_timestamps;
        const int64_t* end = begin + view.num_groups;

        const auto it = cuda::std::upper_bound(begin, end, timestamp);
        return it - begin;
    }

    DEVICE inline size_t find_group_before_timestamp_device(
        const TemporalGraphView& view, const int64_t timestamp) {
        if (view.num_groups == 0) return 0;

        const int64_t* begin = view.unique_timestamps;
        const int64_t* end = begin + view.num_groups;

        const auto it = cuda::std::lower_bound(begin, end, timestamp);
        return (it - begin) - 1;
    }

    DEVICE inline bool is_node_active_device(
        const TemporalGraphView& view, const int node_id) {
        if (node_id < 0 || node_id >= view.active_node_ids_size) {
            return false;
        }
        return view.active_node_ids[node_id] == 1;
    }
#endif

    HOST size_t get_memory_used(const TemporalGraphData& data);

    // test/debug helper; batches D->H copies, prefer one per assertion block
    struct EdgeDataSnapshot {
        std::vector<int>     sources;
        std::vector<int>     targets;
        std::vector<int64_t> timestamps;
        std::vector<size_t>  timestamp_group_offsets;
        std::vector<int64_t> unique_timestamps;
        std::vector<double>  forward_cumulative_weights_exponential;
        std::vector<double>  backward_cumulative_weights_exponential;
        std::vector<int>     active_node_ids;
        std::vector<size_t>  node_adj_offsets;
        std::vector<int>     node_adj_neighbors;
    };

    HOST inline EdgeDataSnapshot snapshot(
        const TemporalGraphData& data
#ifdef HAS_CUDA
        , cudaStream_t stream = 0
#endif
        ) {
        EdgeDataSnapshot snap;

        // pre-size so .data() pointers stay stable through the async copies
        snap.sources.resize(data.sources.size());
        snap.targets.resize(data.targets.size());
        snap.timestamps.resize(data.timestamps.size());
        snap.timestamp_group_offsets.resize(data.timestamp_group_offsets.size());
        snap.unique_timestamps.resize(data.unique_timestamps.size());
        snap.forward_cumulative_weights_exponential.resize(
            data.forward_cumulative_weights_exponential.size());
        snap.backward_cumulative_weights_exponential.resize(
            data.backward_cumulative_weights_exponential.size());
        snap.active_node_ids.resize(data.active_node_ids.size());
        snap.node_adj_offsets.resize(data.node_adj_offsets.size());
        snap.node_adj_neighbors.resize(data.node_adj_neighbors.size());

#ifdef HAS_CUDA
        data.sources.copy_to_host_async(
            snap.sources.data(), snap.sources.size(), stream);
        data.targets.copy_to_host_async(
            snap.targets.data(), snap.targets.size(), stream);
        data.timestamps.copy_to_host_async(
            snap.timestamps.data(), snap.timestamps.size(), stream);
        data.timestamp_group_offsets.copy_to_host_async(
            snap.timestamp_group_offsets.data(),
            snap.timestamp_group_offsets.size(), stream);
        data.unique_timestamps.copy_to_host_async(
            snap.unique_timestamps.data(),
            snap.unique_timestamps.size(), stream);
        data.forward_cumulative_weights_exponential.copy_to_host_async(
            snap.forward_cumulative_weights_exponential.data(),
            snap.forward_cumulative_weights_exponential.size(), stream);
        data.backward_cumulative_weights_exponential.copy_to_host_async(
            snap.backward_cumulative_weights_exponential.data(),
            snap.backward_cumulative_weights_exponential.size(), stream);
        data.active_node_ids.copy_to_host_async(
            snap.active_node_ids.data(),
            snap.active_node_ids.size(), stream);
        data.node_adj_offsets.copy_to_host_async(
            snap.node_adj_offsets.data(),
            snap.node_adj_offsets.size(), stream);
        data.node_adj_neighbors.copy_to_host_async(
            snap.node_adj_neighbors.data(),
            snap.node_adj_neighbors.size(), stream);

        if (data.use_gpu) {
            CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(stream));
        }
#else
        data.sources.copy_to_host_async(snap.sources.data(), snap.sources.size());
        data.targets.copy_to_host_async(snap.targets.data(), snap.targets.size());
        data.timestamps.copy_to_host_async(snap.timestamps.data(), snap.timestamps.size());
        data.timestamp_group_offsets.copy_to_host_async(
            snap.timestamp_group_offsets.data(), snap.timestamp_group_offsets.size());
        data.unique_timestamps.copy_to_host_async(
            snap.unique_timestamps.data(), snap.unique_timestamps.size());
        data.forward_cumulative_weights_exponential.copy_to_host_async(
            snap.forward_cumulative_weights_exponential.data(),
            snap.forward_cumulative_weights_exponential.size());
        data.backward_cumulative_weights_exponential.copy_to_host_async(
            snap.backward_cumulative_weights_exponential.data(),
            snap.backward_cumulative_weights_exponential.size());
        data.active_node_ids.copy_to_host_async(
            snap.active_node_ids.data(), snap.active_node_ids.size());
        data.node_adj_offsets.copy_to_host_async(
            snap.node_adj_offsets.data(), snap.node_adj_offsets.size());
        data.node_adj_neighbors.copy_to_host_async(
            snap.node_adj_neighbors.data(), snap.node_adj_neighbors.size());
#endif

        return snap;
    }

}

#endif // EDGE_DATA_CUH
