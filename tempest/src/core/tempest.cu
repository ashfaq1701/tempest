#include "tempest.cuh"

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>
#include <vector>

#include "tempest_cpu.cuh"
#include "../common/setup.cuh"
#include "../common/random_gen.cuh"
#include "../common/cuda_config.cuh"
#include "../common/error_handlers.cuh"
#include "../common/nvtx.cuh"
#include "../graph/edge_data.cuh"
#include "../graph/node_edge_index.cuh"
#include "../random/pickers.cuh"
#include "../utils/utils.cuh"
#include "../utils/random.cuh"

#ifdef HAS_CUDA
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include "tempest_kernels_full_walk.cuh"
#include "node_grouped/dispatch.cuh"
#include "../data/walk_set/walk_set_device.cuh"
#endif

#include "../data/walk_set/walk_set_host.cuh"
#include "../data/walk_set/walk_set_view.cuh"
#include "../data/temporal_graph_view.cuh"

core::Tempest::Tempest(
    const bool is_directed, const bool use_gpu,
    const int64_t max_time_capacity,
    const bool enable_weight_computation,
    const bool enable_temporal_node2vec,
    const double timescale_bound,
    const double node2vec_p, const double node2vec_q,
    const int walk_padding_value,
    const uint64_t global_seed,
    const bool shuffle_walk_order,
    const int cuda_device_id)
    : data_(use_gpu),
      walk_padding_value_(walk_padding_value),
      global_seed_(global_seed),
      shuffle_walk_order_(shuffle_walk_order) {

    // header default pinned use_gpu=false; rebind
    last_batch_unique_sources_ = Buffer<int>(use_gpu);
    last_batch_unique_targets_ = Buffer<int>(use_gpu);

    data_.is_directed              = is_directed;
    data_.max_time_capacity        = max_time_capacity;
    data_.timescale_bound          = timescale_bound;
    data_.node2vec_p               = node2vec_p;
    data_.node2vec_q               = node2vec_q;
    data_.inv_p                    = 1.0 / node2vec_p;
    data_.inv_q                    = 1.0 / node2vec_q;
    data_.enable_temporal_node2vec = enable_temporal_node2vec;
    data_.enable_weight_computation =
        enable_weight_computation || enable_temporal_node2vec;

#ifdef HAS_CUDA
    if (use_gpu) {
        cuda_device_id_ = cuda_device_id;
        // pin this thread to our target device for the rest of the ctor,
        // then restore on scope exit so callers (e.g. PyTorch) keep theirs.
        CudaDeviceGuard _g(cuda_device_id_);
        CUDA_CHECK_AND_CLEAR(cudaGetDeviceProperties(
            &cuda_device_prop_, cuda_device_id_));
        // non-blocking: concurrent TRW instances must not serialize on stream 0
        CUDA_CHECK_AND_CLEAR(cudaStreamCreateWithFlags(
            &stream_, cudaStreamNonBlocking));
    }
#endif
}

core::Tempest::~Tempest() {
#ifdef HAS_CUDA
    // Pin to our device before tearing down the stream and (implicitly,
    // after this body) Buffer<T> members that may issue device-scoped
    // frees. Restored on scope exit so the caller's context is preserved.
    if (data_.use_gpu) {
        CudaDeviceGuard _g(cuda_device_id_);
        if (stream_ != nullptr) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    } else if (stream_ != nullptr) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
#endif
}

core::Tempest::Tempest(Tempest&& other) noexcept
    : data_(std::move(other.data_)),
      walk_padding_value_(other.walk_padding_value_),
      global_seed_(other.global_seed_),
      shuffle_walk_order_(other.shuffle_walk_order_),
      last_batch_unique_sources_(std::move(other.last_batch_unique_sources_)),
      last_batch_unique_targets_(std::move(other.last_batch_unique_targets_))
#ifdef HAS_CUDA
    , cuda_device_prop_(other.cuda_device_prop_),
      stream_(other.stream_),
      cuda_device_id_(other.cuda_device_id_)
#endif
{
#ifdef HAS_CUDA
    other.stream_ = nullptr;
#endif
}

core::Tempest& core::Tempest::operator=(
    Tempest&& other) noexcept {
    if (this == &other) return *this;
#ifdef HAS_CUDA
    // Tear down our existing stream on our existing device, not whatever
    // device the caller's thread happens to point at.
    if (stream_ != nullptr) {
        if (data_.use_gpu) {
            CudaDeviceGuard _g(cuda_device_id_);
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        } else {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
        stream_ = nullptr;
    }
#endif
    data_                       = std::move(other.data_);
    walk_padding_value_         = other.walk_padding_value_;
    global_seed_                = other.global_seed_;
    shuffle_walk_order_         = other.shuffle_walk_order_;
    last_batch_unique_sources_  = std::move(other.last_batch_unique_sources_);
    last_batch_unique_targets_  = std::move(other.last_batch_unique_targets_);
#ifdef HAS_CUDA
    cuda_device_prop_ = other.cuda_device_prop_;
    stream_           = other.stream_;
    cuda_device_id_   = other.cuda_device_id_;
    other.stream_     = nullptr;
#endif
    return *this;
}

namespace {

void set_last_batch_unique_std(
    const int* values, const size_t n, Buffer<int>& out) {
    std::set<int> s(values, values + n);
    out.shrink_to_fit_empty();
    if (!s.empty()) {
        out.resize(s.size());
        std::copy(s.begin(), s.end(), out.data());
    }
}

#ifdef HAS_CUDA
// runs on null stream to order with add_multiple_edges_cuda
void set_last_batch_unique_cuda_device_input(
    int* values_device, const size_t n, Buffer<int>& out) {
    if (n == 0) {
        out.shrink_to_fit_empty();
        return;
    }

    thrust::device_ptr<int> s_begin(values_device);
    thrust::device_ptr<int> s_end(values_device + n);
    thrust::sort(DEVICE_EXECUTION_POLICY, s_begin, s_end);
    auto new_end = thrust::unique(DEVICE_EXECUTION_POLICY, s_begin, s_end);
    const size_t unique_count = static_cast<size_t>(new_end - s_begin);

    out.shrink_to_fit_empty();
    out.resize(unique_count);
    CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
        out.data(), values_device, unique_count * sizeof(int),
        cudaMemcpyDeviceToDevice));
}
#endif

#ifdef HAS_CUDA
DataBlock<int> get_last_batch_start_nodes_device(
    const core::Tempest* trw,
    const int num_walks_per_node) {
    const Buffer<int>& src = trw->last_batch_unique_sources();

    Buffer<int> start_device(/*use_gpu=*/true);

    if (trw->is_directed()) {
        start_device.resize(src.size());
        if (src.size() > 0) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
                start_device.data(), src.data(),
                src.size() * sizeof(int), cudaMemcpyDeviceToDevice));
        }
    } else {
        const Buffer<int>& dst = trw->last_batch_unique_targets();
        const size_t total = src.size() + dst.size();
        start_device.resize(total);
        if (src.size() > 0) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
                start_device.data(), src.data(),
                src.size() * sizeof(int), cudaMemcpyDeviceToDevice));
        }
        if (dst.size() > 0) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
                start_device.data() + src.size(), dst.data(),
                dst.size() * sizeof(int), cudaMemcpyDeviceToDevice));
        }
        if (total > 0) {
            thrust::device_ptr<int> d_begin(start_device.data());
            thrust::device_ptr<int> d_end(start_device.data() + total);
            thrust::sort(DEVICE_EXECUTION_POLICY, d_begin, d_end);
            auto new_end = thrust::unique(DEVICE_EXECUTION_POLICY, d_begin, d_end);
            const size_t unique_count = static_cast<size_t>(new_end - d_begin);
            start_device.resize(unique_count);
        }
    }

    const size_t out_size = start_device.size() * static_cast<size_t>(num_walks_per_node);
    DataBlock<int> repeated(out_size, /*use_gpu=*/true);
    if (start_device.size() > 0 && num_walks_per_node > 0) {
        const int* src_ptr = start_device.data();
        thrust::transform(
            DEVICE_EXECUTION_POLICY,
            thrust::counting_iterator<size_t>(0),
            thrust::counting_iterator<size_t>(out_size),
            thrust::device_pointer_cast(repeated.data),
            [src_ptr, num_walks_per_node] DEVICE (const size_t idx) {
                return src_ptr[idx / static_cast<size_t>(num_walks_per_node)];
            });
        CUDA_KERNEL_CHECK(
            "After thrust transform in get_last_batch_start_nodes_device");
    }
    return repeated;
}
#endif

DataBlock<int> get_last_batch_start_nodes_new(
    const core::Tempest* trw,
    const int num_walks_per_node) {
#ifdef HAS_CUDA
    if (trw->data().use_gpu) {
        return get_last_batch_start_nodes_device(trw, num_walks_per_node);
    }
#endif

    std::vector<int> start_nodes;
    if (trw->is_directed()) {
        const Buffer<int>& src = trw->last_batch_unique_sources();
        start_nodes.assign(src.data(), src.data() + src.size());
    } else {
        const Buffer<int>& src = trw->last_batch_unique_sources();
        const Buffer<int>& dst = trw->last_batch_unique_targets();
        std::set<int> u(src.data(), src.data() + src.size());
        u.insert(dst.data(), dst.data() + dst.size());
        start_nodes.assign(u.begin(), u.end());
    }

    return repeat_elements(
        start_nodes.data(), start_nodes.size(),
        num_walks_per_node, /*use_gpu=*/false);
}

} // namespace

HOST void tempest::add_multiple_edges(
    core::Tempest* trw,
    const int* sources, const int* targets, const int64_t* timestamps,
    const size_t num_edges,
    const float* edge_features, const size_t feature_dim,
    const size_t block_dim) {

    // ingestion kernels use compile-time BLOCK_DIM; param kept for API symmetry
    (void)block_dim;
    if (num_edges == 0) return;

    NVTX_RANGE_COLORED("add_multiple_edges", nvtx_colors::edge_purple);

#ifdef HAS_CUDA
    if (trw->data().use_gpu) {
        temporal_graph::add_multiple_edges_cuda(
            trw->data(), sources, targets, timestamps, num_edges,
            edge_features, feature_dim);
    } else
#endif
    {
        temporal_graph::add_multiple_edges_std(
            trw->data(), sources, targets, timestamps, num_edges,
            edge_features, feature_dim);
    }

    {
        NVTX_RANGE_COLORED("Unique sources/targets", nvtx_colors::edge_purple);
#ifdef HAS_CUDA
        if (trw->data().use_gpu) {
            Buffer<int> src_scratch(/*use_gpu=*/true);
            Buffer<int> tgt_scratch(/*use_gpu=*/true);
            src_scratch.resize(num_edges);
            tgt_scratch.resize(num_edges);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(
                src_scratch.data(), sources, num_edges * sizeof(int),
                cudaMemcpyHostToDevice));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(
                tgt_scratch.data(), targets, num_edges * sizeof(int),
                cudaMemcpyHostToDevice));

            set_last_batch_unique_cuda_device_input(
                src_scratch.data(), num_edges,
                trw->last_batch_unique_sources());
            set_last_batch_unique_cuda_device_input(
                tgt_scratch.data(), num_edges,
                trw->last_batch_unique_targets());
            return;
        }
#endif
        set_last_batch_unique_std(sources, num_edges, trw->last_batch_unique_sources());
        set_last_batch_unique_std(targets, num_edges, trw->last_batch_unique_targets());
    }
}

HOST size_t tempest::get_node_count(const core::Tempest* trw) {
    return temporal_graph::get_node_count(trw->data());
}

HOST size_t tempest::get_edge_count(const core::Tempest* trw) {
    return temporal_graph::get_total_edges(trw->data());
}

HOST std::vector<int> tempest::get_node_ids(const core::Tempest* trw) {
    return temporal_graph::get_node_ids(trw->data());
}

HOST std::vector<int64_t> tempest::get_node_degrees(
    const core::Tempest* trw,
    const int* nodes, const size_t n, const WalkDirection direction) {
    const bool forward = direction == WalkDirection::Forward_In_Time;
    return temporal_graph::get_node_degrees(trw->data(), nodes, n, forward);
}

HOST std::vector<Edge> tempest::get_edges(const core::Tempest* trw) {
    return temporal_graph::get_edges(trw->data());
}

HOST bool tempest::get_is_directed(const core::Tempest* trw) {
    return trw->data().is_directed;
}

HOST void tempest::clear(core::Tempest* trw) {
    const bool use_gpu = trw->data().use_gpu;
    TemporalGraphData fresh(use_gpu);
    fresh.is_directed               = trw->data().is_directed;
    fresh.max_time_capacity         = trw->data().max_time_capacity;
    fresh.timescale_bound           = trw->data().timescale_bound;
    fresh.node2vec_p                = trw->data().node2vec_p;
    fresh.node2vec_q                = trw->data().node2vec_q;
    fresh.inv_p                     = trw->data().inv_p;
    fresh.inv_q                     = trw->data().inv_q;
    fresh.enable_weight_computation = trw->data().enable_weight_computation;
    fresh.enable_temporal_node2vec  = trw->data().enable_temporal_node2vec;
    trw->data() = std::move(fresh);
    trw->last_batch_unique_sources().shrink_to_fit_empty();
    trw->last_batch_unique_targets().shrink_to_fit_empty();
}

HOST size_t tempest::get_memory_used(const core::Tempest* trw) {
    return temporal_graph::get_memory_used(trw->data());
}

namespace {

WalksWithEdgeFeaturesHost finalize_host_walks(
    core::Tempest* trw, WalkSetHost host_walks) {
    const int fdim = static_cast<int>(trw->data().feature_dim);
    WalksWithEdgeFeaturesHost result(std::move(host_walks), fdim);
    if (fdim > 0) {
        // edge_features is always host-resident
        result.populate_walk_edge_features(trw->data().edge_features.data());
    }
    return result;
}

} // namespace

HOST WalksWithEdgeFeaturesHost
tempest::get_random_walks_and_times_for_all_nodes_std(
    core::Tempest* trw,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction) {
    NVTX_RANGE_COLORED("Walk Sampling (all nodes, std)", nvtx_colors::walk_green);
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    std::vector<int> node_ids = temporal_graph::get_node_ids(trw->data());
    const DataBlock<int> repeated_node_ids = repeat_elements(
        node_ids.data(), node_ids.size(),
        num_walks_per_node, trw->data().use_gpu);

    if (trw->shuffle_walk_order()) {
        shuffle_vector_host<int>(repeated_node_ids.data, repeated_node_ids.size);
    }

    WalkSetHost host_walks(repeated_node_ids.size, max_walk_len,
                           trw->walk_padding_value());
    WalkSetView walk_set_view = host_walks.make_host_view();

    Buffer<double> rand_nums = generate_n_random_numbers(
        repeated_node_ids.size + repeated_node_ids.size * max_walk_len * 2, false);

    const TemporalGraphView view = make_temporal_graph_view(trw->data());

    launch_random_walk_cpu_new(
        view,
        trw->is_directed(),
        walk_set_view,
        max_walk_len,
        repeated_node_ids.data,
        repeated_node_ids.size,
        *walk_bias,
        *initial_edge_bias,
        walk_direction,
        rand_nums.data());

    return finalize_host_walks(trw, std::move(host_walks));
}

HOST WalksWithEdgeFeaturesHost
tempest::get_random_walks_and_times_for_last_batch_std(
    core::Tempest* trw,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction) {
    NVTX_RANGE_COLORED("Walk Sampling (last batch, std)", nvtx_colors::walk_green);
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    const DataBlock<int> repeated_node_ids =
        get_last_batch_start_nodes_new(trw, num_walks_per_node);

    if (trw->shuffle_walk_order()) {
        shuffle_vector_host<int>(repeated_node_ids.data, repeated_node_ids.size);
    }

    WalkSetHost host_walks(repeated_node_ids.size, max_walk_len,
                           trw->walk_padding_value());
    WalkSetView walk_set_view = host_walks.make_host_view();

    Buffer<double> rand_nums = generate_n_random_numbers(
        repeated_node_ids.size + repeated_node_ids.size * max_walk_len * 2, false);

    const TemporalGraphView view = make_temporal_graph_view(trw->data());

    launch_random_walk_cpu_new(
        view,
        trw->is_directed(),
        walk_set_view,
        max_walk_len,
        repeated_node_ids.data,
        repeated_node_ids.size,
        *walk_bias,
        *initial_edge_bias,
        walk_direction,
        rand_nums.data());

    return finalize_host_walks(trw, std::move(host_walks));
}

HOST WalksWithEdgeFeaturesHost
tempest::get_random_walks_and_times_for_nodes_std(
    core::Tempest* trw,
    const int* seed_nodes,
    const size_t num_seed_nodes,
    const int64_t* cutoff_times,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction) {
    NVTX_RANGE_COLORED("Walk Sampling (seed nodes, std)", nvtx_colors::walk_green);
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    const DataBlock<int> repeated_node_ids = repeat_elements(
        seed_nodes, num_seed_nodes,
        num_walks_per_node, trw->data().use_gpu);

    // Per-seed cutoffs fan out to per-walk with the same seed-major layout, so
    // a shared-seed co-shuffle keeps each walk paired with its seed's cutoff.
    DataBlock<int64_t> repeated_cutoffs;
    if (cutoff_times != nullptr) {
        repeated_cutoffs = repeat_elements(
            cutoff_times, num_seed_nodes, num_walks_per_node, trw->data().use_gpu);
    }

    if (trw->shuffle_walk_order()) {
        const unsigned int shuffle_seed = std::random_device{}();
        shuffle_vector_host<int>(
            repeated_node_ids.data, repeated_node_ids.size, shuffle_seed);
        if (cutoff_times != nullptr) {
            shuffle_vector_host<int64_t>(
                repeated_cutoffs.data, repeated_cutoffs.size, shuffle_seed);
        }
    }

    WalkSetHost host_walks(repeated_node_ids.size, max_walk_len,
                           trw->walk_padding_value());
    WalkSetView walk_set_view = host_walks.make_host_view();

    if (cutoff_times != nullptr) {
        std::memcpy(walk_set_view.cutoffs, repeated_cutoffs.data,
                    repeated_cutoffs.size * sizeof(int64_t));
    }

    Buffer<double> rand_nums = generate_n_random_numbers(
        repeated_node_ids.size + repeated_node_ids.size * max_walk_len * 2, false);

    const TemporalGraphView view = make_temporal_graph_view(trw->data());

    launch_random_walk_cpu_new(
        view,
        trw->is_directed(),
        walk_set_view,
        max_walk_len,
        repeated_node_ids.data,
        repeated_node_ids.size,
        *walk_bias,
        *initial_edge_bias,
        walk_direction,
        rand_nums.data());

    return finalize_host_walks(trw, std::move(host_walks));
}

HOST WalksWithEdgeFeaturesHost
tempest::get_random_walks_and_times_std(
    core::Tempest* trw,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_total,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction) {
    NVTX_RANGE_COLORED("Walk Sampling (std)", nvtx_colors::walk_green);
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    WalkSetHost host_walks(static_cast<size_t>(num_walks_total), max_walk_len,
                           trw->walk_padding_value());
    WalkSetView walk_set_view = host_walks.make_host_view();

    Buffer<double> rand_nums = generate_n_random_numbers(
        num_walks_total + num_walks_total * max_walk_len * 2, false);

    const std::vector<int> start_node_ids(num_walks_total, -1);

    const TemporalGraphView view = make_temporal_graph_view(trw->data());

    launch_random_walk_cpu_new(
        view,
        trw->is_directed(),
        walk_set_view,
        max_walk_len,
        start_node_ids.data(),
        static_cast<size_t>(num_walks_total),
        *walk_bias,
        *initial_edge_bias,
        walk_direction,
        rand_nums.data());

    return finalize_host_walks(trw, std::move(host_walks));
}

#ifdef HAS_CUDA

namespace {

uint64_t resolve_base_seed(const core::Tempest* trw) {
    return (trw->global_seed() != EMPTY_GLOBAL_SEED)
        ? trw->global_seed()
        : secure_random_seed();
}

void launch_walk_kernel_dispatch(
    const KernelLaunchType kernel_launch_type,
    const TemporalGraphView& view,
    const bool is_directed,
    const WalkSetView& walk_set_view,
    const int max_walk_len,
    const int* start_node_ids,
    const size_t num_walks,
    const bool all_starts_unconstrained,
    const RandomPickerType walk_bias,
    const RandomPickerType initial_edge_bias,
    const WalkDirection walk_direction,
    const uint64_t base_seed,
    const dim3& grid_dim,
    const dim3& block_dim,
    const int w_threshold_warp,
    const cudaStream_t stream) {
    switch (kernel_launch_type) {
        case KernelLaunchType::FULL_WALK: {
            NVTX_RANGE_COLORED("Launch walk kernel (full)", nvtx_colors::walk_green);
            tempest::launch_random_walk_kernel_full_walk(
                view, is_directed, walk_set_view, max_walk_len,
                start_node_ids, num_walks,
                walk_bias, initial_edge_bias, walk_direction,
                base_seed, grid_dim, block_dim, stream);
            break;
        }
        case KernelLaunchType::NODE_GROUPED: {
            NVTX_RANGE_COLORED("Launch walk kernel (node-grouped)", nvtx_colors::walk_green);
            tempest::dispatch_node_grouped_kernel(
                view, is_directed, walk_set_view, max_walk_len,
                start_node_ids, num_walks, all_starts_unconstrained,
                walk_bias, initial_edge_bias, walk_direction,
                base_seed, grid_dim, block_dim, stream,
                /*force_global_only=*/false,
                w_threshold_warp);
            break;
        }
        case KernelLaunchType::NODE_GROUPED_GLOBAL_ONLY: {
            NVTX_RANGE_COLORED("Launch walk kernel (node-grouped, global-only)",
                               nvtx_colors::walk_green);
            tempest::dispatch_node_grouped_kernel(
                view, is_directed, walk_set_view, max_walk_len,
                start_node_ids, num_walks, all_starts_unconstrained,
                walk_bias, initial_edge_bias, walk_direction,
                base_seed, grid_dim, block_dim, stream,
                /*force_global_only=*/true,
                w_threshold_warp);
            break;
        }
        default:
            throw std::runtime_error("Unknown KernelLaunchType");
    }
}

} // namespace

HOST WalksWithEdgeFeaturesHost
tempest::get_random_walks_and_times_for_all_nodes_cuda(
    core::Tempest* trw,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type,
    const size_t block_dim,
    const int w_threshold_warp) {
    NVTX_RANGE_COLORED("Walk Sampling (all nodes)", nvtx_colors::walk_green);
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    // non-blocking stream won't auto-sync with prior default-stream work
    CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(0));

    std::vector<int> host_node_ids = temporal_graph::get_node_ids(trw->data());
    const DataBlock<int> repeated_node_ids = repeat_elements(
        host_node_ids.data(), host_node_ids.size(),
        num_walks_per_node, trw->data().use_gpu);

    const uint64_t base_seed = resolve_base_seed(trw);

    auto [grid_dim, launch_block_dim] = get_optimal_launch_params(
        repeated_node_ids.size,
        &trw->cuda_device_prop(),
        block_dim);

    if (trw->shuffle_walk_order()) {
        shuffle_vector_device<int>(repeated_node_ids.data, repeated_node_ids.size);
        CUDA_KERNEL_CHECK(
            "After shuffle_vector_device in get_random_walks_and_times_for_all_nodes_cuda");
    }

    WalkSetDevice device_walks(repeated_node_ids.size, max_walk_len,
                               trw->walk_padding_value());
    const WalkSetView walk_set_view = device_walks.make_view();

    const TemporalGraphView view = make_temporal_graph_view(trw->data());

    CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(0));

    launch_walk_kernel_dispatch(
        kernel_launch_type, view, trw->is_directed(), walk_set_view,
        max_walk_len, repeated_node_ids.data, repeated_node_ids.size,
        /*all_starts_unconstrained=*/false,
        *walk_bias, *initial_edge_bias, walk_direction,
        base_seed, grid_dim, launch_block_dim, w_threshold_warp, trw->stream());

    CUDA_KERNEL_CHECK(
        "After generate_random_walks_kernel in get_random_walks_and_times_for_all_nodes_cuda");

    trw->sync_stream();

    WalkSetHost host_walks = std::move(device_walks).download_to_host();

    return finalize_host_walks(trw, std::move(host_walks));
}

HOST WalksWithEdgeFeaturesHost
tempest::get_random_walks_and_times_for_last_batch_cuda(
    core::Tempest* trw,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type,
    const size_t block_dim,
    const int w_threshold_warp) {
    NVTX_RANGE_COLORED("Walk Sampling (last batch)", nvtx_colors::walk_green);
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(0));

    const DataBlock<int> repeated_node_ids =
        get_last_batch_start_nodes_new(trw, num_walks_per_node);

    const uint64_t base_seed = resolve_base_seed(trw);

    auto [grid_dim, launch_block_dim] = get_optimal_launch_params(
        repeated_node_ids.size,
        &trw->cuda_device_prop(),
        block_dim);

    if (trw->shuffle_walk_order()) {
        shuffle_vector_device<int>(repeated_node_ids.data, repeated_node_ids.size);
        CUDA_KERNEL_CHECK(
            "After shuffle_vector_device in get_random_walks_and_times_for_last_batch_cuda");
    }

    WalkSetDevice device_walks(repeated_node_ids.size, max_walk_len,
                               trw->walk_padding_value());
    const WalkSetView walk_set_view = device_walks.make_view();

    const TemporalGraphView view = make_temporal_graph_view(trw->data());

    CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(0));

    launch_walk_kernel_dispatch(
        kernel_launch_type, view, trw->is_directed(), walk_set_view,
        max_walk_len, repeated_node_ids.data, repeated_node_ids.size,
        /*all_starts_unconstrained=*/false,
        *walk_bias, *initial_edge_bias, walk_direction,
        base_seed, grid_dim, launch_block_dim, w_threshold_warp, trw->stream());

    CUDA_KERNEL_CHECK(
        "After generate_random_walks_kernel in get_random_walks_and_times_for_last_batch_cuda");

    trw->sync_stream();

    WalkSetHost host_walks = std::move(device_walks).download_to_host();

    return finalize_host_walks(trw, std::move(host_walks));
}

HOST WalksWithEdgeFeaturesHost
tempest::get_random_walks_and_times_for_nodes_cuda(
    core::Tempest* trw,
    const int* seed_nodes,
    const size_t num_seed_nodes,
    const int64_t* cutoff_times,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type,
    const size_t block_dim,
    const int w_threshold_warp) {
    NVTX_RANGE_COLORED("Walk Sampling (seed nodes)", nvtx_colors::walk_green);
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(0));

    const DataBlock<int> repeated_node_ids = repeat_elements(
        seed_nodes, num_seed_nodes,
        num_walks_per_node, trw->data().use_gpu);

    // Per-seed cutoffs (host pointer) fan out to a per-walk DEVICE array with
    // the same seed-major layout as the node ids — uploaded by repeat_elements.
    DataBlock<int64_t> repeated_cutoffs;
    if (cutoff_times != nullptr) {
        repeated_cutoffs = repeat_elements(
            cutoff_times, num_seed_nodes, num_walks_per_node, trw->data().use_gpu);
    }

    const uint64_t base_seed = resolve_base_seed(trw);

    auto [grid_dim, launch_block_dim] = get_optimal_launch_params(
        repeated_node_ids.size,
        &trw->cuda_device_prop(),
        block_dim);

    if (trw->shuffle_walk_order()) {
        const unsigned int shuffle_seed =
            static_cast<unsigned int>(std::random_device{}());
        shuffle_vector_device<int>(
            repeated_node_ids.data, repeated_node_ids.size, shuffle_seed);
        if (cutoff_times != nullptr) {
            shuffle_vector_device<int64_t>(
                repeated_cutoffs.data, repeated_cutoffs.size, shuffle_seed);
        }
        CUDA_KERNEL_CHECK(
            "After shuffle_vector_device in get_random_walks_and_times_for_nodes_cuda");
    }

    WalkSetDevice device_walks(repeated_node_ids.size, max_walk_len,
                               trw->walk_padding_value());
    const WalkSetView walk_set_view = device_walks.make_view();

    if (cutoff_times != nullptr) {
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            walk_set_view.cutoffs, repeated_cutoffs.data,
            repeated_cutoffs.size * sizeof(int64_t),
            cudaMemcpyDeviceToDevice));
    }

    const TemporalGraphView view = make_temporal_graph_view(trw->data());

    CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(0));

    launch_walk_kernel_dispatch(
        kernel_launch_type, view, trw->is_directed(), walk_set_view,
        max_walk_len, repeated_node_ids.data, repeated_node_ids.size,
        /*all_starts_unconstrained=*/false,
        *walk_bias, *initial_edge_bias, walk_direction,
        base_seed, grid_dim, launch_block_dim, w_threshold_warp, trw->stream());

    CUDA_KERNEL_CHECK(
        "After generate_random_walks_kernel in get_random_walks_and_times_for_nodes_cuda");

    trw->sync_stream();

    WalkSetHost host_walks = std::move(device_walks).download_to_host();

    return finalize_host_walks(trw, std::move(host_walks));
}

HOST WalksWithEdgeFeaturesHost
tempest::get_random_walks_and_times_cuda(
    core::Tempest* trw,
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_total,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type,
    const size_t block_dim,
    const int w_threshold_warp) {
    NVTX_RANGE_COLORED("Walk Sampling", nvtx_colors::walk_green);
    if (!initial_edge_bias) {
        initial_edge_bias = walk_bias;
    }

    CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(0));

    const uint64_t base_seed = resolve_base_seed(trw);

    auto [grid_dim, launch_block_dim] = get_optimal_launch_params(
        num_walks_total,
        &trw->cuda_device_prop(),
        block_dim);

    // -1 sentinel = random start
    Buffer<int> start_node_ids(true);
    start_node_ids.resize(num_walks_total);
    start_node_ids.fill(-1);

    WalkSetDevice device_walks(num_walks_total, max_walk_len,
                               trw->walk_padding_value());
    const WalkSetView walk_set_view = device_walks.make_view();

    const TemporalGraphView view = make_temporal_graph_view(trw->data());

    CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(0));

    launch_walk_kernel_dispatch(
        kernel_launch_type, view, trw->is_directed(), walk_set_view,
        max_walk_len, start_node_ids.data(), static_cast<size_t>(num_walks_total),
        /*all_starts_unconstrained=*/true,
        *walk_bias, *initial_edge_bias, walk_direction,
        base_seed, grid_dim, launch_block_dim, w_threshold_warp, trw->stream());

    CUDA_KERNEL_CHECK(
        "After generate_random_walks_kernel in get_random_walks_and_times_cuda");

    trw->sync_stream();

    WalkSetHost host_walks = std::move(device_walks).download_to_host();

    return finalize_host_walks(trw, std::move(host_walks));
}

#endif

void core::Tempest::add_multiple_edges(
    const int* sources, const int* targets, const int64_t* timestamps,
    const size_t n, const float* edge_features, const size_t feature_dim,
    const size_t block_dim) {
#ifdef HAS_CUDA
    CudaDeviceGuard _g(data_.use_gpu ? cuda_device_id_ : -1);
#endif
    tempest::add_multiple_edges(
        this, sources, targets, timestamps, n, edge_features, feature_dim,
        block_dim);
}

void core::Tempest::add_multiple_edges(
    const std::vector<std::tuple<int, int, int64_t>>& edges,
    const float* edge_features, const size_t feature_dim,
    const size_t block_dim) {
    // The other overload below acquires its own CudaDeviceGuard; no need
    // to double-guard here.
    std::vector<int> sources; sources.reserve(edges.size());
    std::vector<int> targets; targets.reserve(edges.size());
    std::vector<int64_t> timestamps; timestamps.reserve(edges.size());
    for (const auto& e : edges) {
        sources.push_back(std::get<0>(e));
        targets.push_back(std::get<1>(e));
        timestamps.push_back(std::get<2>(e));
    }
    add_multiple_edges(sources.data(), targets.data(), timestamps.data(),
                       timestamps.size(), edge_features, feature_dim, block_dim);
}

WalksWithEdgeFeaturesHost
core::Tempest::get_random_walks_and_times_for_all_nodes(
    const int max_walk_len, const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type,
    const size_t block_dim,
    const int w_threshold_warp) {
#ifdef HAS_CUDA
    CudaDeviceGuard _g(data_.use_gpu ? cuda_device_id_ : -1);
    if (data_.use_gpu) {
        return tempest::get_random_walks_and_times_for_all_nodes_cuda(
            this, max_walk_len, walk_bias, num_walks_per_node,
            initial_edge_bias, walk_direction, kernel_launch_type,
            block_dim, w_threshold_warp);
    }
#endif
    (void)kernel_launch_type;
    (void)block_dim;
    (void)w_threshold_warp;
    return tempest::get_random_walks_and_times_for_all_nodes_std(
        this, max_walk_len, walk_bias, num_walks_per_node,
        initial_edge_bias, walk_direction);
}

WalksWithEdgeFeaturesHost
core::Tempest::get_random_walks_and_times_for_last_batch(
    const int max_walk_len, const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type,
    const size_t block_dim,
    const int w_threshold_warp) {
#ifdef HAS_CUDA
    CudaDeviceGuard _g(data_.use_gpu ? cuda_device_id_ : -1);
    if (data_.use_gpu) {
        return tempest::get_random_walks_and_times_for_last_batch_cuda(
            this, max_walk_len, walk_bias, num_walks_per_node,
            initial_edge_bias, walk_direction, kernel_launch_type,
            block_dim, w_threshold_warp);
    }
#endif
    (void)kernel_launch_type;
    (void)block_dim;
    (void)w_threshold_warp;
    return tempest::get_random_walks_and_times_for_last_batch_std(
        this, max_walk_len, walk_bias, num_walks_per_node,
        initial_edge_bias, walk_direction);
}

WalksWithEdgeFeaturesHost
core::Tempest::get_random_walks_and_times_for_nodes(
    const int* seed_nodes,
    const size_t num_seed_nodes,
    const int64_t* cutoff_times,
    const int max_walk_len, const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type,
    const size_t block_dim,
    const int w_threshold_warp) {
#ifdef HAS_CUDA
    CudaDeviceGuard _g(data_.use_gpu ? cuda_device_id_ : -1);
    if (data_.use_gpu) {
        return tempest::get_random_walks_and_times_for_nodes_cuda(
            this, seed_nodes, num_seed_nodes, cutoff_times,
            max_walk_len, walk_bias, num_walks_per_node,
            initial_edge_bias, walk_direction, kernel_launch_type,
            block_dim, w_threshold_warp);
    }
#endif
    (void)kernel_launch_type;
    (void)block_dim;
    (void)w_threshold_warp;
    return tempest::get_random_walks_and_times_for_nodes_std(
        this, seed_nodes, num_seed_nodes, cutoff_times,
        max_walk_len, walk_bias, num_walks_per_node,
        initial_edge_bias, walk_direction);
}

WalksWithEdgeFeaturesHost
core::Tempest::get_random_walks_and_times(
    const int max_walk_len, const RandomPickerType* walk_bias,
    const int num_walks_total,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const KernelLaunchType kernel_launch_type,
    const size_t block_dim,
    const int w_threshold_warp) {
#ifdef HAS_CUDA
    CudaDeviceGuard _g(data_.use_gpu ? cuda_device_id_ : -1);
    if (data_.use_gpu) {
        return tempest::get_random_walks_and_times_cuda(
            this, max_walk_len, walk_bias, num_walks_total,
            initial_edge_bias, walk_direction, kernel_launch_type,
            block_dim, w_threshold_warp);
    }
#endif
    (void)kernel_launch_type;
    (void)block_dim;
    (void)w_threshold_warp;
    return tempest::get_random_walks_and_times_std(
        this, max_walk_len, walk_bias, num_walks_total,
        initial_edge_bias, walk_direction);
}

void core::Tempest::set_node_features(
    const int* node_ids, const size_t num_nodes,
    const float* node_features_src, const size_t feature_dim) {
#ifdef HAS_CUDA
    CudaDeviceGuard _g(data_.use_gpu ? cuda_device_id_ : -1);
#endif
    node_features::set_node_features(
        data_, data_.max_node_id, node_ids, num_nodes,
        node_features_src, feature_dim);
}

size_t core::Tempest::get_node_count() const {
    return tempest::get_node_count(this);
}
size_t core::Tempest::get_edge_count() const {
    return tempest::get_edge_count(this);
}
std::vector<int> core::Tempest::get_node_ids() const {
#ifdef HAS_CUDA
    CudaDeviceGuard _g(data_.use_gpu ? cuda_device_id_ : -1);
#endif
    return tempest::get_node_ids(this);
}
std::vector<int64_t> core::Tempest::get_node_degrees(
    const int* nodes, const size_t n, const WalkDirection direction) const {
#ifdef HAS_CUDA
    CudaDeviceGuard _g(data_.use_gpu ? cuda_device_id_ : -1);
#endif
    return tempest::get_node_degrees(this, nodes, n, direction);
}
std::vector<Edge> core::Tempest::get_edges() const {
#ifdef HAS_CUDA
    CudaDeviceGuard _g(data_.use_gpu ? cuda_device_id_ : -1);
#endif
    return tempest::get_edges(this);
}
void core::Tempest::clear() {
#ifdef HAS_CUDA
    // reassigning data_ runs Buffer<T> dtors which call cudaFree
    CudaDeviceGuard _g(data_.use_gpu ? cuda_device_id_ : -1);
#endif
    tempest::clear(this);
}
size_t core::Tempest::get_memory_used() const {
    return tempest::get_memory_used(this);
}
