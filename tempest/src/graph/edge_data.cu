#include "edge_data.cuh"

#include <algorithm>
#include <cmath>
#include <vector>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <tbb/parallel_scan.h>
#include <tbb/parallel_sort.h>

#ifdef HAS_CUDA

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>

#include "../common/cuda_scan.cuh"

#endif

#include "../utils/omp_utils.cuh"

HOST DEVICE size_t edge_data::size(const TemporalGraphData& data) {
    return data.timestamps.size();
}

HOST void edge_data::set_size(TemporalGraphData& data, const size_t size) {
    data.sources.resize(size);
    data.targets.resize(size);
    data.timestamps.resize(size);

    if (data.feature_dim > 0) {
        data.edge_features.resize(size * data.feature_dim);
    }
}

HOST bool edge_data::empty(const TemporalGraphData& data) {
    return data.timestamps.size() == 0;
}

HOST void edge_data::add_edges(
    TemporalGraphData& data,
    const int* sources,
    const int* targets,
    const int64_t* timestamps,
    const size_t num_new_edges) {
    edge_data::add_edges(data, sources, targets, timestamps, num_new_edges, nullptr, 0);
}

HOST void edge_data::add_edges(
    TemporalGraphData& data,
    const int* sources,
    const int* targets,
    const int64_t* timestamps,
    const size_t num_new_edges,
    const float* edge_features,
    const size_t feature_dim) {
    if (edge_features != nullptr && feature_dim == 0) {
        throw std::runtime_error("edge_features provided but feature_dim is 0");
    }

    if (data.sources.size() != 0 && feature_dim != data.feature_dim) {
        throw std::runtime_error("feature_dim mismatch with existing data.feature_dim");
    }

    if (data.sources.size() == 0) {
        data.feature_dim = feature_dim;
    }

    if (edge_features == nullptr && feature_dim != 0) {
        throw std::runtime_error("feature_dim must be 0 when edge_features is not provided");
    }

    if (data.feature_dim > 0 && edge_features == nullptr && num_new_edges > 0) {
        throw std::runtime_error("edge features enabled; non-empty ingestion must provide edge_features");
    }

    data.sources.append_from_host(sources, num_new_edges);
    data.targets.append_from_host(targets, num_new_edges);
    data.timestamps.append_from_host(timestamps, num_new_edges);

    if (data.feature_dim > 0 && edge_features != nullptr && num_new_edges > 0) {
        const size_t feature_values = num_new_edges * data.feature_dim;
        data.edge_features.append_from_host(edge_features, feature_values);
    }
}

HOST std::vector<Edge> edge_data::get_edges(const TemporalGraphData& data) {
    const size_t n = data.timestamps.size();
    std::vector<Edge> result(n);

    if (n == 0) {
        return result;
    }

    std::vector<int> host_sources;
    std::vector<int> host_targets;
    std::vector<int64_t> host_timestamps;

    const int* sources = data.sources.data();
    const int* targets = data.targets.data();
    const int64_t* timestamps = data.timestamps.data();

    #ifdef HAS_CUDA
    if (data.use_gpu) {
        host_sources = data.sources.to_host_vector();
        host_targets = data.targets.to_host_vector();
        host_timestamps = data.timestamps.to_host_vector();

        sources = host_sources.data();
        targets = host_targets.data();
        timestamps = host_timestamps.data();
    }
    #endif

    for (size_t i = 0; i < n; i++) {
        const float* edge_features = nullptr;
        int edge_feature_dim = 0;

        if (data.feature_dim > 0 && data.edge_features.data() != nullptr) {
            edge_features = data.edge_features.data() + (i * data.feature_dim);
            edge_feature_dim = static_cast<int>(data.feature_dim);
        }

        result[i] = Edge{sources[i], targets[i], timestamps[i], edge_features, edge_feature_dim};
    }

    return result;
}

HOST std::vector<int> edge_data::get_active_node_ids(const TemporalGraphData& data) {
    const size_t n = data.active_node_ids.size();
    size_t active_count = 0;

    #ifdef HAS_CUDA
    if (data.use_gpu) {
        const thrust::device_ptr<int> d_active_nodes(
            const_cast<int*>(data.active_node_ids.data()));
        active_count = thrust::count(
            DEVICE_EXECUTION_POLICY,
            d_active_nodes,
            d_active_nodes + static_cast<long>(n),
            1
        );
        CUDA_KERNEL_CHECK("After thrust count in get_active_node_ids");
    } else
    #endif
    {
        for (size_t i = 0; i < n; i++) {
            if (data.active_node_ids.data()[i] == 1) {
                active_count++;
            }
        }
    }

    std::vector<int> result(active_count);
    if (active_count == 0) {
        return result;
    }

    #ifdef HAS_CUDA
    if (data.use_gpu) {
        Buffer<int> d_result(true);
        d_result.resize(active_count);

        const thrust::device_ptr<int> d_active_nodes(
            const_cast<int*>(data.active_node_ids.data()));
        const thrust::device_ptr<int> d_result_ptr(d_result.data());

        thrust::copy_if(
            DEVICE_EXECUTION_POLICY,
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(static_cast<int>(n)),
            d_active_nodes,
            d_result_ptr,
            [] __device__ (const int val) { return val == 1; }
        );
        CUDA_KERNEL_CHECK("After thrust copy_if in get_active_node_ids");

        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            result.data(), d_result.data(),
            active_count * sizeof(int), cudaMemcpyDeviceToHost));
    } else
    #endif
    {
        size_t index = 0;
        for (size_t i = 0; i < n; i++) {
            if (data.active_node_ids.data()[i] == 1) {
                result[index++] = static_cast<int>(i);
            }
        }
    }

    return result;
}

HOST size_t edge_data::active_node_count(const TemporalGraphData& data) {
    const size_t n = data.active_node_ids.size();
    size_t count = 0;

    #ifdef HAS_CUDA
    if (data.use_gpu) {
        const thrust::device_ptr<int> d_active_nodes(
            const_cast<int*>(data.active_node_ids.data()));
        count = thrust::count(
            DEVICE_EXECUTION_POLICY,
            d_active_nodes,
            d_active_nodes + static_cast<long>(n),
            1
        );
        CUDA_KERNEL_CHECK("After thrust count in active_node_count");
    } else
    #endif
    {
        for (size_t i = 0; i < n; i++) {
            if (data.active_node_ids.data()[i] == 1) {
                count++;
            }
        }
    }

    return count;
}

HOST bool edge_data::is_node_active(const TemporalGraphData& data, const int node_id) {
    if (node_id < 0 || static_cast<size_t>(node_id) >= data.active_node_ids.size()) {
        return false;
    }

    #ifdef HAS_CUDA
    if (data.use_gpu) {
        int is_active;
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            &is_active, data.active_node_ids.data() + node_id,
            sizeof(int), cudaMemcpyDeviceToHost));
        return is_active == 1;
    }
    #endif

    return data.active_node_ids.data()[node_id] == 1;
}

HOST void edge_data::populate_active_nodes_std(TemporalGraphData& data) {
    const size_t num_edges = size(data);
    if (num_edges == 0) {
        data.max_node_id = -1;
        data.active_node_ids.shrink_to_fit_empty();
        return;
    }

    const int* sources = data.sources.data();
    const int* targets = data.targets.data();

    int max_source = -1;
    #pragma omp parallel for reduction(max:max_source)
    for (size_t i = 0; i < num_edges; ++i) {
        max_source = std::max(max_source, sources[i]);
    }

    int max_target = -1;
    #pragma omp parallel for reduction(max:max_target)
    for (size_t i = 0; i < num_edges; ++i) {
        max_target = std::max(max_target, targets[i]);
    }

    const int max_node_id = std::max(max_source, max_target);
    data.max_node_id = max_node_id;

    data.active_node_ids.resize(max_node_id + 1);
    data.active_node_ids.fill(0);

    int* active = data.active_node_ids.data();

    #pragma omp parallel for
    for (size_t i = 0; i < num_edges; ++i) {
        active[sources[i]] = 1;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < num_edges; ++i) {
        active[targets[i]] = 1;
    }
}

HOST void edge_data::build_node_adjacency_csr_std(TemporalGraphData& data) {
    const size_t n = data.active_node_ids.size();
    const size_t m = size(data);

    data.node_adj_offsets.resize(n + 1);
    data.node_adj_neighbors.resize(2 * m);

    if (n == 0 || m == 0) {
        size_t* offsets = data.node_adj_offsets.data();
        #pragma omp parallel for
        for (size_t i = 0; i < n + 1; ++i) {
            offsets[i] = 0;
        }
        data.node_adj_neighbors.resize(0);
        return;
    }

    const int* sources = data.sources.data();
    const int* targets = data.targets.data();
    size_t* offsets = data.node_adj_offsets.data();
    int* neighbors = data.node_adj_neighbors.data();

    std::vector<std::atomic<size_t>> degree(n);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        degree[i].store(0, std::memory_order_relaxed);
    }

    #pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        const int u = sources[i];
        const int v = targets[i];

        degree[static_cast<size_t>(u)].fetch_add(1, std::memory_order_relaxed);
        degree[static_cast<size_t>(v)].fetch_add(1, std::memory_order_relaxed);
    }

    offsets[0] = 0;

    tbb::parallel_scan(
        tbb::blocked_range<size_t>(0, n),
        static_cast<size_t>(0),
        [&](const tbb::blocked_range<size_t>& r, size_t sum, bool is_final) -> size_t {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const size_t d = degree[i].load(std::memory_order_relaxed);
                if (is_final) {
                    offsets[i + 1] = sum + d;
                }
                sum += d;
            }
            return sum;
        },
        std::plus<>()
    );

    std::vector<std::atomic<size_t>> cursor(n);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        cursor[i].store(offsets[i], std::memory_order_relaxed);
    }

    #pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        const int u = sources[i];
        const int v = targets[i];

        const size_t u_pos =
            cursor[static_cast<size_t>(u)].fetch_add(1, std::memory_order_relaxed);
        const size_t v_pos =
            cursor[static_cast<size_t>(v)].fetch_add(1, std::memory_order_relaxed);

        neighbors[u_pos] = v;
        neighbors[v_pos] = u;
    }

    // segmented sort via (owner_node, neighbor) pair sort
    const size_t nnz = 2 * m;
    std::vector<std::pair<int, int>> key_neighbor(nnz);

    #pragma omp parallel for
    for (size_t i = 0; i < nnz; ++i) {
        const auto it  = std::upper_bound(offsets, offsets + n + 1, i);
        const int owner = static_cast<int>((it - offsets) - 1);
        key_neighbor[i] = {owner, neighbors[i]};
    }

    tbb::parallel_sort(key_neighbor.begin(), key_neighbor.end());

    #pragma omp parallel for
    for (size_t i = 0; i < nnz; ++i) {
        neighbors[i] = key_neighbor[i].second;
    }
}

HOST void edge_data::update_timestamp_groups_std(TemporalGraphData& data) {
    if (data.timestamps.size() == 0) {
        data.timestamp_group_offsets.shrink_to_fit_empty();
        data.unique_timestamps.shrink_to_fit_empty();

        data.max_node_id = -1;
        data.active_node_ids.shrink_to_fit_empty();
        return;
    }

    const size_t n = data.timestamps.size();
    const int64_t* timestamps = data.timestamps.data();

    std::vector<int> flags(n, 0);
    flags[0] = 1;

    #pragma omp parallel for
    for (size_t i = 1; i < n; ++i) {
        flags[i] = (timestamps[i] != timestamps[i - 1]) ? 1 : 0;
    }

    std::vector<int> prefix_sum(n);
    parallel_prefix_sum(flags.data(), prefix_sum.data(), n);

    const int num_groups = prefix_sum[n - 1] + flags[n - 1];

    data.timestamp_group_offsets.resize(num_groups + 1);
    data.unique_timestamps.resize(num_groups);

    size_t* group_offsets = data.timestamp_group_offsets.data();
    int64_t* unique_ts = data.unique_timestamps.data();

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        if (flags[i]) {
            const int idx = prefix_sum[i];
            group_offsets[idx] = i;
            unique_ts[idx] = timestamps[i];
        }
    }

    group_offsets[num_groups] = n;

    populate_active_nodes_std(data);
}

HOST void edge_data::update_temporal_weights_std(
    TemporalGraphData& data, const double timescale_bound) {
    if (data.timestamps.size() == 0) {
        data.forward_cumulative_weights_exponential.shrink_to_fit_empty();
        data.backward_cumulative_weights_exponential.shrink_to_fit_empty();
        return;
    }

    const int64_t* timestamps = data.timestamps.data();
    const int64_t min_timestamp = timestamps[0];
    const int64_t max_timestamp = timestamps[data.timestamps.size() - 1];
    const auto time_diff = static_cast<double>(max_timestamp - min_timestamp);
    const double time_scale = (timescale_bound > 0 && time_diff > 0) ? timescale_bound / time_diff : 1.0;

    const size_t num_groups = get_timestamp_group_count(data);

    data.forward_cumulative_weights_exponential.resize(num_groups);
    data.backward_cumulative_weights_exponential.resize(num_groups);

    auto* forward = data.forward_cumulative_weights_exponential.data();
    auto* backward = data.backward_cumulative_weights_exponential.data();
    const auto* offsets = data.timestamp_group_offsets.data();

    // Pivot so that the largest weight in each direction lands at exp(0) = 1
    // instead of exp(span·scale), which would overflow on long-span unix-
    // timestamp data with timescale_bound = -1. The per-group exponent here
    // is now ≤ 0, so weights live in (0, group_size]. The probability
    // distribution is preserved by softmax shift-invariance — every weight
    // is scaled by the same per-direction constant, which cancels in the
    // normalization step below.
    //
    //   Forward picker biases toward the earliest-in-time group ⇒ pivot on
    //   min_timestamp so the smallest group_timestamp gives the largest
    //   weight (exp((min − ts) · scale) = exp(0) at ts = min).
    //   Backward picker biases toward the latest-in-time group ⇒ pivot on
    //   max_timestamp so the largest group_timestamp gives the largest
    //   weight (exp((ts − max) · scale) = exp(0) at ts = max).
    #pragma omp parallel for
    for (size_t group = 0; group < num_groups; ++group) {
        const size_t start = offsets[group];
        const size_t group_size = offsets[group + 1] - offsets[group];

        const int64_t ts = timestamps[start];

        const auto t_fwd = static_cast<double>(min_timestamp - ts);  // ≤ 0
        const auto t_bwd = static_cast<double>(ts - max_timestamp);  // ≤ 0

        const double fwd_scaled = (timescale_bound > 0) ? t_fwd * time_scale : t_fwd;
        const double bwd_scaled = (timescale_bound > 0) ? t_bwd * time_scale : t_bwd;

        forward[group]  = static_cast<double>(group_size) * std::exp(fwd_scaled);
        backward[group] = static_cast<double>(group_size) * std::exp(bwd_scaled);
    }

    double forward_sum = 0.0;
    #pragma omp parallel for reduction(+:forward_sum)
    for (size_t group = 0; group < num_groups; ++group) {
        forward_sum += forward[group];
    }

    double backward_sum = 0.0;
    #pragma omp parallel for reduction(+:backward_sum)
    for (size_t group = 0; group < num_groups; ++group) {
        backward_sum += backward[group];
    }

    #pragma omp parallel for
    for (size_t group = 0; group < num_groups; ++group) {
        forward[group]  /= forward_sum;
        backward[group] /= backward_sum;
    }

    parallel_inclusive_scan(forward, num_groups);
    parallel_inclusive_scan(backward, num_groups);
}

#ifdef HAS_CUDA

HOST void edge_data::populate_active_nodes_cuda(TemporalGraphData& data) {
    const size_t num_edges = size(data);
    if (num_edges == 0) {
        data.max_node_id = -1;
        data.active_node_ids.shrink_to_fit_empty();
        return;
    }

    const thrust::device_ptr<int> d_sources(data.sources.data());
    const thrust::device_ptr<int> d_targets(data.targets.data());

    const int max_source = thrust::reduce(
        DEVICE_EXECUTION_POLICY,
        d_sources,
        d_sources + static_cast<long>(num_edges),
        0,
        thrust::maximum<int>()
    );

    const int max_target = thrust::reduce(
        DEVICE_EXECUTION_POLICY,
        d_targets,
        d_targets + static_cast<long>(num_edges),
        0,
        thrust::maximum<int>()
    );

    const int max_node_id = std::max(max_source, max_target);
    data.max_node_id = max_node_id;

    data.active_node_ids.resize(max_node_id + 1);
    data.active_node_ids.fill(0);

    thrust::device_ptr<int> d_active_nodes(data.active_node_ids.data());

    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        d_sources,
        d_sources + static_cast<long>(num_edges),
        [d_active_nodes] __device__ (const int source_id) {
            d_active_nodes[source_id] = 1;
        }
    );
    CUDA_KERNEL_CHECK("After thrust for_each sources in populate_active_nodes_cuda");

    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        d_targets,
        d_targets + static_cast<long>(num_edges),
        [d_active_nodes] __device__ (int target_id) {
            d_active_nodes[target_id] = 1;
        }
    );
    CUDA_KERNEL_CHECK("After thrust for_each targets in populate_active_nodes_cuda");
}

HOST void edge_data::build_node_adjacency_csr_cuda(TemporalGraphData& data) {
    const size_t n = data.active_node_ids.size();
    const size_t m = size(data);

    data.node_adj_offsets.resize(n + 1);
    data.node_adj_neighbors.resize(2 * m);

    if (n == 0 || m == 0) {
        thrust::fill_n(
            DEVICE_EXECUTION_POLICY,
            thrust::device_pointer_cast(data.node_adj_offsets.data()),
            static_cast<long>(n + 1),
            static_cast<size_t>(0)
        );
        CUDA_KERNEL_CHECK("After thrust fill_n zero offsets in build_node_adjacency_csr_cuda");
        return;
    }

    const auto d_sources   = thrust::device_pointer_cast(data.sources.data());
    const auto d_targets   = thrust::device_pointer_cast(data.targets.data());
    const auto d_offsets   = thrust::device_pointer_cast(data.node_adj_offsets.data());
    const auto d_neighbors = thrust::device_pointer_cast(data.node_adj_neighbors.data());

    // unsigned int for atomicAdd compatibility
    thrust::device_vector<unsigned int> degree(n, 0u);

    unsigned int* d_degree_raw = thrust::raw_pointer_cast(degree.data());

    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(m),
        [d_sources, d_targets, d_degree_raw] __device__ (const size_t i) {
            const int u = d_sources[static_cast<long>(i)];
            const int v = d_targets[static_cast<long>(i)];
            atomicAdd(d_degree_raw + u, 1u);
            atomicAdd(d_degree_raw + v, 1u);
        }
    );
    CUDA_KERNEL_CHECK("After thrust for_each degree counting in build_node_adjacency_csr_cuda");

    thrust::exclusive_scan(
        DEVICE_EXECUTION_POLICY,
        degree.begin(),
        degree.end(),
        d_offsets,
        static_cast<size_t>(0)
    );
    CUDA_KERNEL_CHECK("After thrust exclusive_scan offsets in build_node_adjacency_csr_cuda");

    thrust::fill_n(
        DEVICE_EXECUTION_POLICY,
        d_offsets + static_cast<long>(n),
        1,
        static_cast<size_t>(2 * m)
    );
    CUDA_KERNEL_CHECK("After writing offsets[n] in build_node_adjacency_csr_cuda");

    thrust::device_vector<unsigned int> cursor(n, 0u);

    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        d_offsets,
        d_offsets + static_cast<long>(n),
        cursor.begin(),
        [] __device__ (const size_t x) { return static_cast<unsigned int>(x); }
    );
    CUDA_KERNEL_CHECK("After cursor init transform in build_node_adjacency_csr_cuda");

    unsigned int* d_cursor_raw = thrust::raw_pointer_cast(cursor.data());

    thrust::for_each(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(m),
        [d_sources, d_targets, d_cursor_raw, d_neighbors] __device__ (const size_t i) {
            const int u = d_sources[static_cast<long>(i)];
            const int v = d_targets[static_cast<long>(i)];

            const unsigned int u_pos = atomicAdd(d_cursor_raw + u, 1u);
            const unsigned int v_pos = atomicAdd(d_cursor_raw + v, 1u);

            d_neighbors[static_cast<long>(static_cast<size_t>(u_pos))] = v;
            d_neighbors[static_cast<long>(static_cast<size_t>(v_pos))] = u;
        }
    );
    CUDA_KERNEL_CHECK("After thrust for_each neighbor fill in build_node_adjacency_csr_cuda");

    const auto nnz = static_cast<size_t>(2 * m);

    thrust::device_vector<int> node_ids(nnz);

    thrust::upper_bound(
        DEVICE_EXECUTION_POLICY,
        d_offsets,
        d_offsets + static_cast<long>(n + 1),
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(nnz),
        node_ids.begin()
    );
    CUDA_KERNEL_CHECK("After thrust upper_bound in build_node_adjacency_csr_cuda");

    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        node_ids.begin(),
        node_ids.end(),
        node_ids.begin(),
        [] __device__ (int x) { return x - 1; }
    );
    CUDA_KERNEL_CHECK("After node_ids subtract-one transform in build_node_adjacency_csr_cuda");

    const auto zipped_begin = thrust::make_zip_iterator(
        thrust::make_tuple(
            node_ids.begin(),
            d_neighbors
        )
    );

    thrust::sort(
        DEVICE_EXECUTION_POLICY,
        zipped_begin,
        zipped_begin + static_cast<long>(nnz)
    );
    CUDA_KERNEL_CHECK("After thrust segmented sort (key,neighbor) in build_node_adjacency_csr_cuda");
}

HOST void edge_data::update_timestamp_groups_cuda(TemporalGraphData& data) {
    if (data.timestamps.size() == 0) {
        data.timestamp_group_offsets.shrink_to_fit_empty();
        data.unique_timestamps.shrink_to_fit_empty();

        data.max_node_id = -1;
        data.active_node_ids.shrink_to_fit_empty();
        return;
    }

    const size_t n = data.timestamps.size();

    Buffer<int> d_flags(true);
    d_flags.resize(n);

    const thrust::device_ptr<int64_t> d_timestamps(data.timestamps.data());
    const thrust::device_ptr<int> d_flags_ptr(d_flags.data());

    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        d_timestamps + 1,
        d_timestamps + static_cast<long>(n),
        d_timestamps,
        d_flags_ptr + 1,
        [] HOST DEVICE (const int64_t curr, const int64_t prev) { return curr != prev ? 1 : 0; });
    CUDA_KERNEL_CHECK("After thrust transform in update_timestamp_groups_cuda");

    thrust::fill_n(d_flags_ptr, 1, 1);
    CUDA_KERNEL_CHECK("After thrust fill_n in update_timestamp_groups_cuda");

    const size_t num_groups = thrust::reduce(d_flags_ptr, d_flags_ptr + static_cast<long>(n));
    CUDA_KERNEL_CHECK("After thrust reduce in update_timestamp_groups_cuda");

    data.timestamp_group_offsets.resize(num_groups + 1);
    data.unique_timestamps.resize(num_groups);

    const thrust::device_ptr<size_t> d_group_offsets(data.timestamp_group_offsets.data());
    const thrust::device_ptr<int64_t> d_unique_timestamps(data.unique_timestamps.data());

    thrust::copy_if(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(n),
        d_flags_ptr,
        d_group_offsets,
        [] HOST DEVICE (const int flag) { return flag == 1; });
    CUDA_KERNEL_CHECK("After thrust copy_if group boundaries in update_timestamp_groups_cuda");

    thrust::fill_n(d_group_offsets + static_cast<long>(num_groups), 1, n);
    CUDA_KERNEL_CHECK("After thrust fill_n final offset in update_timestamp_groups_cuda");

    thrust::copy_if(
        DEVICE_EXECUTION_POLICY,
        d_timestamps,
        d_timestamps + static_cast<long>(n),
        d_flags_ptr,
        d_unique_timestamps,
        [] HOST DEVICE (const int flag) { return flag == 1; });
    CUDA_KERNEL_CHECK("After thrust copy_if unique timestamps in update_timestamp_groups_cuda");

    populate_active_nodes_cuda(data);
}

HOST void edge_data::update_temporal_weights_cuda(
    TemporalGraphData& data, const double timescale_bound) {
    if (data.timestamps.size() == 0) {
        data.forward_cumulative_weights_exponential.shrink_to_fit_empty();
        data.backward_cumulative_weights_exponential.shrink_to_fit_empty();
        return;
    }

    const size_t n = data.timestamps.size();

    int64_t min_timestamp, max_timestamp;
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(
        &min_timestamp, data.timestamps.data(),
        sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_CLEAR(cudaMemcpy(
        &max_timestamp, data.timestamps.data() + (n - 1),
        sizeof(int64_t), cudaMemcpyDeviceToHost));

    const auto time_diff = static_cast<double>(max_timestamp - min_timestamp);
    const double time_scale = (timescale_bound > 0 && time_diff > 0) ? timescale_bound / time_diff : 1.0;

    const size_t num_groups = get_timestamp_group_count(data);

    data.forward_cumulative_weights_exponential.resize(num_groups);
    data.backward_cumulative_weights_exponential.resize(num_groups);

    Buffer<double> d_forward_weights(true);
    d_forward_weights.resize(num_groups);
    Buffer<double> d_backward_weights(true);
    d_backward_weights.resize(num_groups);

    thrust::device_ptr<int64_t> d_timestamps(data.timestamps.data());
    thrust::device_ptr<size_t> d_offsets(data.timestamp_group_offsets.data());
    thrust::device_ptr<double> d_forward_weights_ptr(d_forward_weights.data());
    thrust::device_ptr<double> d_backward_weights_ptr(d_backward_weights.data());

    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(num_groups),
        thrust::make_zip_iterator(thrust::make_tuple(
            d_forward_weights_ptr,
            d_backward_weights_ptr
        )),
        // See edge_data::update_temporal_weights_std for the pivot rationale —
        // exponents are bounded ≤ 0 so the per-group weights stay in
        // (0, group_size] regardless of timescale_bound, while the normalized
        // distribution is unchanged by softmax shift-invariance.
        [d_offsets, d_timestamps, max_timestamp, min_timestamp, timescale_bound, time_scale]
        HOST DEVICE (const size_t group) {
            const size_t start = d_offsets[static_cast<long>(group)];
            const size_t end   = d_offsets[static_cast<long>(group + 1)];
            const double group_size = static_cast<double>(end - start);

            const int64_t group_timestamp = d_timestamps[static_cast<long>(start)];

            const auto time_diff_forward  = static_cast<double>(min_timestamp - group_timestamp);
            const auto time_diff_backward = static_cast<double>(group_timestamp - max_timestamp);

            const double forward_scaled = timescale_bound > 0 ? time_diff_forward * time_scale : time_diff_forward;
            const double backward_scaled = timescale_bound > 0
                                               ? time_diff_backward * time_scale
                                               : time_diff_backward;

            return thrust::make_tuple(
                static_cast<double>(group_size) * exp(forward_scaled),
                static_cast<double>(group_size) * exp(backward_scaled));
        }
    );
    CUDA_KERNEL_CHECK("After thrust transform weights calculation in update_temporal_weights_cuda");

    double forward_sum = thrust::reduce(
        DEVICE_EXECUTION_POLICY,
        d_forward_weights_ptr,
        d_forward_weights_ptr + static_cast<long>(num_groups)
    );
    CUDA_KERNEL_CHECK("After thrust reduce forward weights in update_temporal_weights_cuda");

    double backward_sum = thrust::reduce(
        DEVICE_EXECUTION_POLICY,
        d_backward_weights_ptr,
        d_backward_weights_ptr + static_cast<long>(num_groups)
    );
    CUDA_KERNEL_CHECK("After thrust reduce backward weights in update_temporal_weights_cuda");

    const auto norm_in = thrust::make_zip_iterator(
        thrust::make_tuple(d_forward_weights_ptr, d_backward_weights_ptr));
    thrust::transform(
        DEVICE_EXECUTION_POLICY,
        norm_in,
        norm_in + static_cast<long>(num_groups),
        norm_in,
        [forward_sum, backward_sum] HOST DEVICE (
            const thrust::tuple<double, double>& w) {
            return thrust::make_tuple(
                thrust::get<0>(w) / forward_sum,
                thrust::get<1>(w) / backward_sum);
        }
    );
    CUDA_KERNEL_CHECK("After thrust transform fused weight normalization in update_temporal_weights_cuda");

    double* d_forward_cumulative  = data.forward_cumulative_weights_exponential.data();
    double* d_backward_cumulative = data.backward_cumulative_weights_exponential.data();

    // cub specializes double scans, faster than thrust on large arrays
    cub_inclusive_sum(
        d_forward_weights_ptr,
        d_forward_cumulative,
        num_groups
    );
    CUDA_KERNEL_CHECK("After cub inclusive_sum forward weights in update_temporal_weights_cuda");

    cub_inclusive_sum(
        d_backward_weights_ptr,
        d_backward_cumulative,
        num_groups
    );
    CUDA_KERNEL_CHECK("After cub inclusive_sum backward weights in update_temporal_weights_cuda");
}

#endif

HOST size_t edge_data::get_memory_used(const TemporalGraphData& data) {
    size_t total_memory = 0;

    total_memory += data.sources.size() * sizeof(int);
    total_memory += data.targets.size() * sizeof(int);
    total_memory += data.timestamps.size() * sizeof(int64_t);

    total_memory += data.edge_features.size() * sizeof(float);

    total_memory += data.active_node_ids.size() * sizeof(int);

    total_memory += data.node_adj_offsets.size() * sizeof(size_t);
    total_memory += data.node_adj_neighbors.size() * sizeof(int);

    total_memory += data.timestamp_group_offsets.size() * sizeof(size_t);
    total_memory += data.unique_timestamps.size() * sizeof(int64_t);

    total_memory += data.forward_cumulative_weights_exponential.size() * sizeof(double);
    total_memory += data.backward_cumulative_weights_exponential.size() * sizeof(double);

    return total_memory;
}
