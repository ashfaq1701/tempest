#ifndef NODE_EDGE_INDEX_CUH
#define NODE_EDGE_INDEX_CUH

#include <cstddef>
#include <vector>

#include "../common/macros.cuh"
#include "../common/error_handlers.cuh"
#include "../data/structs.cuh"
#include "../data/temporal_graph_data.cuh"
#include "../data/buffer.cuh"

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace node_edge_index {

    HOST void clear(TemporalGraphData& data);

    // host-safe queries: dispatch on data.use_gpu, copy scalars D->H as needed
    HOST SizeRange get_edge_range(
        const TemporalGraphData& data,
        int dense_node_id,
        bool forward);

    // Batched O(1)-per-node degree lookup. For each queried node id (the array
    // may contain duplicates and out-of-range / inactive ids, which yield 0),
    // reads degree = offsets[id+1] - offsets[id] from the CSR node-edge index.
    // forward=true  -> outbound degree (directed) / total degree (undirected),
    // forward=false -> inbound degree  (directed) / total degree (undirected).
    // Mirrored CPU (OpenMP) / GPU (thrust) implementation, dispatched on
    // data.use_gpu. Returns a host vector of length n.
    HOST std::vector<int64_t> get_node_degrees(
        const TemporalGraphData& data,
        const int* nodes,
        size_t n,
        bool forward);

    HOST SizeRange get_timestamp_group_range(
        const TemporalGraphData& data,
        int dense_node_id,
        size_t group_idx,
        bool forward);

    HOST const Buffer<size_t>& get_timestamp_offset_vector(
        const TemporalGraphData& data,
        bool forward);

    HOST size_t get_timestamp_group_count(
        const TemporalGraphData& data,
        int dense_node_id,
        bool forward);

    HOST void allocate_node_group_offsets(
        TemporalGraphData& data,
        size_t node_index_capacity);

    HOST void allocate_node_ts_sorted_indices(TemporalGraphData& data);

    HOST void compute_node_group_offsets_std(TemporalGraphData& data);

    HOST void compute_node_ts_sorted_indices_std(
        TemporalGraphData& data,
        size_t outbound_buffer_size,
        int* outbound_node_ids,
        int* inbound_node_ids);

    HOST void allocate_and_compute_node_ts_group_counts_and_offsets_std(
        TemporalGraphData& data,
        size_t node_count,
        const int* outbound_node_ids,
        const int* inbound_node_ids);

    HOST void update_temporal_weights_std(
        TemporalGraphData& data,
        double timescale_bound);

    #ifdef HAS_CUDA

    HOST void compute_node_group_offsets_cuda(TemporalGraphData& data);

    HOST void compute_node_ts_sorted_indices_cuda(
        TemporalGraphData& data,
        size_t outbound_buffer_size,
        int* outbound_node_ids,
        int* inbound_node_ids);

    HOST void allocate_and_compute_node_ts_group_counts_and_offsets_cuda(
        TemporalGraphData& data,
        size_t node_count,
        const int* outbound_node_ids,
        const int* inbound_node_ids);

    HOST void update_temporal_weights_cuda(
        TemporalGraphData& data,
        double timescale_bound);

    #endif

    HOST void rebuild(TemporalGraphData& data);

    HOST size_t get_memory_used(const TemporalGraphData& data);

    // test/debug helper; see edge_data::snapshot
    struct NodeEdgeIndexSnapshot {
        std::vector<size_t> node_group_outbound_offsets;
        std::vector<size_t> node_group_inbound_offsets;
        std::vector<size_t> node_ts_sorted_outbound_indices;
        std::vector<size_t> node_ts_sorted_inbound_indices;
        std::vector<size_t> count_ts_group_per_node_outbound;
        std::vector<size_t> count_ts_group_per_node_inbound;
        std::vector<size_t> node_ts_group_outbound_offsets;
        std::vector<size_t> node_ts_group_inbound_offsets;
        std::vector<double> outbound_forward_cumulative_weights_exponential;
        std::vector<double> outbound_backward_cumulative_weights_exponential;
        std::vector<double> inbound_backward_cumulative_weights_exponential;
    };

    HOST inline NodeEdgeIndexSnapshot snapshot(
        const TemporalGraphData& data
#ifdef HAS_CUDA
        , cudaStream_t stream = 0
#endif
        ) {
        NodeEdgeIndexSnapshot snap;

        snap.node_group_outbound_offsets.resize(data.node_group_outbound_offsets.size());
        snap.node_group_inbound_offsets.resize(data.node_group_inbound_offsets.size());
        snap.node_ts_sorted_outbound_indices.resize(data.node_ts_sorted_outbound_indices.size());
        snap.node_ts_sorted_inbound_indices.resize(data.node_ts_sorted_inbound_indices.size());
        snap.count_ts_group_per_node_outbound.resize(data.count_ts_group_per_node_outbound.size());
        snap.count_ts_group_per_node_inbound.resize(data.count_ts_group_per_node_inbound.size());
        snap.node_ts_group_outbound_offsets.resize(data.node_ts_group_outbound_offsets.size());
        snap.node_ts_group_inbound_offsets.resize(data.node_ts_group_inbound_offsets.size());
        snap.outbound_forward_cumulative_weights_exponential.resize(
            data.outbound_forward_cumulative_weights_exponential.size());
        snap.outbound_backward_cumulative_weights_exponential.resize(
            data.outbound_backward_cumulative_weights_exponential.size());
        snap.inbound_backward_cumulative_weights_exponential.resize(
            data.inbound_backward_cumulative_weights_exponential.size());

#ifdef HAS_CUDA
        data.node_group_outbound_offsets.copy_to_host_async(
            snap.node_group_outbound_offsets.data(),
            snap.node_group_outbound_offsets.size(), stream);
        data.node_group_inbound_offsets.copy_to_host_async(
            snap.node_group_inbound_offsets.data(),
            snap.node_group_inbound_offsets.size(), stream);
        data.node_ts_sorted_outbound_indices.copy_to_host_async(
            snap.node_ts_sorted_outbound_indices.data(),
            snap.node_ts_sorted_outbound_indices.size(), stream);
        data.node_ts_sorted_inbound_indices.copy_to_host_async(
            snap.node_ts_sorted_inbound_indices.data(),
            snap.node_ts_sorted_inbound_indices.size(), stream);
        data.count_ts_group_per_node_outbound.copy_to_host_async(
            snap.count_ts_group_per_node_outbound.data(),
            snap.count_ts_group_per_node_outbound.size(), stream);
        data.count_ts_group_per_node_inbound.copy_to_host_async(
            snap.count_ts_group_per_node_inbound.data(),
            snap.count_ts_group_per_node_inbound.size(), stream);
        data.node_ts_group_outbound_offsets.copy_to_host_async(
            snap.node_ts_group_outbound_offsets.data(),
            snap.node_ts_group_outbound_offsets.size(), stream);
        data.node_ts_group_inbound_offsets.copy_to_host_async(
            snap.node_ts_group_inbound_offsets.data(),
            snap.node_ts_group_inbound_offsets.size(), stream);
        data.outbound_forward_cumulative_weights_exponential.copy_to_host_async(
            snap.outbound_forward_cumulative_weights_exponential.data(),
            snap.outbound_forward_cumulative_weights_exponential.size(), stream);
        data.outbound_backward_cumulative_weights_exponential.copy_to_host_async(
            snap.outbound_backward_cumulative_weights_exponential.data(),
            snap.outbound_backward_cumulative_weights_exponential.size(), stream);
        data.inbound_backward_cumulative_weights_exponential.copy_to_host_async(
            snap.inbound_backward_cumulative_weights_exponential.data(),
            snap.inbound_backward_cumulative_weights_exponential.size(), stream);

        if (data.use_gpu) {
            CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(stream));
        }
#else
        data.node_group_outbound_offsets.copy_to_host_async(
            snap.node_group_outbound_offsets.data(), snap.node_group_outbound_offsets.size());
        data.node_group_inbound_offsets.copy_to_host_async(
            snap.node_group_inbound_offsets.data(), snap.node_group_inbound_offsets.size());
        data.node_ts_sorted_outbound_indices.copy_to_host_async(
            snap.node_ts_sorted_outbound_indices.data(), snap.node_ts_sorted_outbound_indices.size());
        data.node_ts_sorted_inbound_indices.copy_to_host_async(
            snap.node_ts_sorted_inbound_indices.data(), snap.node_ts_sorted_inbound_indices.size());
        data.count_ts_group_per_node_outbound.copy_to_host_async(
            snap.count_ts_group_per_node_outbound.data(), snap.count_ts_group_per_node_outbound.size());
        data.count_ts_group_per_node_inbound.copy_to_host_async(
            snap.count_ts_group_per_node_inbound.data(), snap.count_ts_group_per_node_inbound.size());
        data.node_ts_group_outbound_offsets.copy_to_host_async(
            snap.node_ts_group_outbound_offsets.data(), snap.node_ts_group_outbound_offsets.size());
        data.node_ts_group_inbound_offsets.copy_to_host_async(
            snap.node_ts_group_inbound_offsets.data(), snap.node_ts_group_inbound_offsets.size());
        data.outbound_forward_cumulative_weights_exponential.copy_to_host_async(
            snap.outbound_forward_cumulative_weights_exponential.data(),
            snap.outbound_forward_cumulative_weights_exponential.size());
        data.outbound_backward_cumulative_weights_exponential.copy_to_host_async(
            snap.outbound_backward_cumulative_weights_exponential.data(),
            snap.outbound_backward_cumulative_weights_exponential.size());
        data.inbound_backward_cumulative_weights_exponential.copy_to_host_async(
            snap.inbound_backward_cumulative_weights_exponential.data(),
            snap.inbound_backward_cumulative_weights_exponential.size());
#endif

        return snap;
    }

}

#endif // NODE_EDGE_INDEX_CUH
