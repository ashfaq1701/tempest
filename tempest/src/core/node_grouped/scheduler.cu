#include "scheduler.cuh"

#ifdef HAS_CUDA

#include <cstddef>
#include <cstdint>

#include "../../common/cuda_sort.cuh"
#include "../../common/cuda_scan.cuh"
#include "../../common/error_handlers.cuh"
#include "../../common/nvtx.cuh"
#include "../../common/cuda_config.cuh"
#include "../../random/pickers.cuh"

namespace tempest {

namespace {

__global__ void iota_int_kernel(int* __restrict__ out, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = i;
}

__global__ void walk_alive_flags_kernel(
    WalkSetView walk_set,
    const int step_number,
    const int max_walk_len,
    const std::size_t num_walks,
    uint8_t* __restrict__ alive_flags) {

    const std::size_t walk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (walk_idx >= num_walks) return;

    const std::size_t offset = walk_idx * static_cast<std::size_t>(max_walk_len)
                               + static_cast<std::size_t>(step_number);
    alive_flags[walk_idx] =
        (walk_set.nodes[offset] != walk_set.walk_padding_value)
            ? uint8_t{1} : uint8_t{0};
}

__global__ void gather_last_nodes_kernel(
    WalkSetView walk_set,
    const int* __restrict__ active_walk_idx,
    const int* __restrict__ num_active_ptr,
    const int step_number,
    const int max_walk_len,
    int* __restrict__ last_nodes_out) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_active = *num_active_ptr;
    if (i >= num_active) return;

    const int walk_idx = active_walk_idx[i];
    const std::size_t offset = static_cast<std::size_t>(walk_idx)
                               * static_cast<std::size_t>(max_walk_len)
                               + static_cast<std::size_t>(step_number);
    last_nodes_out[i] = walk_set.nodes[offset];
}

__global__ void partition_by_w_kernel(
    const int* __restrict__ unique_nodes,
    const int* __restrict__ run_starts,
    const int* __restrict__ run_lengths,
    const int* __restrict__ num_runs_ptr,
    const int* __restrict__ sorted_walk_idx,
    const int t_warp,
    const int t_block,
    int* __restrict__ solo_walks,
    int* __restrict__ warp_nodes,
    int* __restrict__ warp_walk_starts,
    int* __restrict__ warp_walk_counts,
    int* __restrict__ block_nodes,
    int* __restrict__ block_walk_starts,
    int* __restrict__ block_walk_counts,
    int* __restrict__ counters) {

    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= *num_runs_ptr) return;

    const int W = run_lengths[r];
    const int start = run_starts[r];
    const int node = unique_nodes[r];

    if (W <= t_warp) {
        // reserve W slots so all walks at this node land in solo
        const int idx = atomicAdd(&counters[0], W);
        for (int k = 0; k < W; ++k) {
            solo_walks[idx + k] = sorted_walk_idx[start + k];
        }
    } else if (W <= t_block) {
        const int idx = atomicAdd(&counters[1], 1);
        warp_nodes[idx]       = node;
        warp_walk_starts[idx] = start;
        warp_walk_counts[idx] = W;
    } else {
        const int idx = atomicAdd(&counters[2], 1);
        block_nodes[idx]       = node;
        block_walk_starts[idx] = start;
        block_walk_counts[idx] = W;
    }
}

// splits a tier's task list into (smem-fit, global-fallback) by G
__global__ void partition_by_g_kernel(
    const int* __restrict__ tier_nodes,
    const int* __restrict__ tier_walk_starts,
    const int* __restrict__ tier_walk_counts,
    const int* __restrict__ num_tier_tasks_ptr,
    const std::size_t* __restrict__ count_ts_group_per_node,
    const int g_cap,
    int* __restrict__ smem_nodes,
    int* __restrict__ smem_walk_starts,
    int* __restrict__ smem_walk_counts,
    int* __restrict__ global_nodes,
    int* __restrict__ global_walk_starts,
    int* __restrict__ global_walk_counts,
    int* __restrict__ g_counters) {

    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= *num_tier_tasks_ptr) return;

    const int node_id    = tier_nodes[t];
    const int walk_start = tier_walk_starts[t];
    const int walk_count = tier_walk_counts[t];

    const std::size_t group_begin = count_ts_group_per_node[node_id];
    const std::size_t group_end   = count_ts_group_per_node[node_id + 1];
    const int G = static_cast<int>(group_end - group_begin);

    if (G <= g_cap) {
        const int idx = atomicAdd(&g_counters[0], 1);
        smem_nodes[idx]        = node_id;
        smem_walk_starts[idx]  = walk_start;
        smem_walk_counts[idx]  = walk_count;
    } else {
        const int idx = atomicAdd(&g_counters[1], 1);
        global_nodes[idx]        = node_id;
        global_walk_starts[idx]  = walk_start;
        global_walk_counts[idx]  = walk_count;
    }
}

// split mega-hubs (W > cap) into sub-tasks so no block monopolizes an SM
__global__ void expand_block_tasks_kernel(
    const int* __restrict__ in_nodes,
    const int* __restrict__ in_walk_starts,
    const int* __restrict__ in_walk_counts,
    const int* __restrict__ in_num_tasks_ptr,
    const int block_walk_cap,
    int* __restrict__ out_nodes,
    int* __restrict__ out_walk_starts,
    int* __restrict__ out_walk_counts,
    int* __restrict__ out_num_tasks) {

    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= *in_num_tasks_ptr) return;

    const int node  = in_nodes[t];
    const int start = in_walk_starts[t];
    const int count = in_walk_counts[t];

    const int num_sub_tasks =
        (count + block_walk_cap - 1) / block_walk_cap;

    // single atomicAdd reserves the whole sub-task range
    const int base_idx = atomicAdd(out_num_tasks, num_sub_tasks);

    for (int k = 0; k < num_sub_tasks; ++k) {
        const int sub_start = start + k * block_walk_cap;
        const int remaining = count - k * block_walk_cap;
        const int sub_count = (remaining < block_walk_cap)
            ? remaining
            : block_walk_cap;

        out_nodes[base_idx + k]        = node;
        out_walk_starts[base_idx + k]  = sub_start;
        out_walk_counts[base_idx + k]  = sub_count;
    }
}

}  // namespace

NodeGroupedScheduler::NodeGroupedScheduler(
    const std::size_t num_walks,
    const dim3 block_dim,
    const int w_threshold_warp,
    const cudaStream_t stream)
    : num_walks_(num_walks),
      num_walks_int_(static_cast<int>(num_walks)),
      block_dim_(block_dim),
      w_threshold_warp_(w_threshold_warp),
      stream_(stream),
      arena_(/*use_gpu=*/true),
      iota_src_(/*use_gpu=*/true) {

    iota_src_.resize(num_walks);

    // reused as cub_partition_flagged input every step
    const std::size_t iota_blocks =
        (num_walks + block_dim_.x - 1) / block_dim_.x;
    iota_int_kernel<<<dim3(static_cast<unsigned>(iota_blocks)),
                      block_dim_, 0, stream_>>>(
        iota_src_.data(), num_walks_int_);
}

NodeGroupedScheduler::StepOutputs NodeGroupedScheduler::run_step(
    WalkSetView walk_set_view,
    const int step_number,
    const int max_walk_len,
    const std::size_t* count_ts_group_per_node,
    const RandomPickerType edge_picker_type,
    const bool force_global_only) {

    NVTX_RANGE_COLORED("NG step", nvtx_colors::walk_green);

    arena_.reset();

    uint8_t* alive_flags       = arena_.acquire<uint8_t>(num_walks_);
    int*     active_walk_idx   = arena_.acquire<int>(num_walks_);
    int*     step_num_active   = arena_.acquire<int>(1);
    int*     last_nodes_active = arena_.acquire<int>(num_walks_);
    int*     sorted_last_nodes = arena_.acquire<int>(num_walks_);
    int*     sorted_active_idx = arena_.acquire<int>(num_walks_);
    int*     unique_last_nodes = arena_.acquire<int>(num_walks_);
    int*     step_run_lengths  = arena_.acquire<int>(num_walks_);
    int*     step_run_starts   = arena_.acquire<int>(num_walks_);
    int*     step_num_runs     = arena_.acquire<int>(1);

    int*     solo_walks         = arena_.acquire<int>(num_walks_);
    int*     warp_nodes_w       = arena_.acquire<int>(num_walks_);
    int*     warp_walk_starts_w = arena_.acquire<int>(num_walks_);
    int*     warp_walk_counts_w = arena_.acquire<int>(num_walks_);
    int*     block_nodes_w      = arena_.acquire<int>(num_walks_);
    int*     block_walk_starts_w= arena_.acquire<int>(num_walks_);
    int*     block_walk_counts_w= arena_.acquire<int>(num_walks_);

    int*     warp_smem_nodes    = arena_.acquire<int>(num_walks_);
    int*     warp_smem_starts   = arena_.acquire<int>(num_walks_);
    int*     warp_smem_counts   = arena_.acquire<int>(num_walks_);
    int*     warp_global_nodes  = arena_.acquire<int>(num_walks_);
    int*     warp_global_starts = arena_.acquire<int>(num_walks_);
    int*     warp_global_counts = arena_.acquire<int>(num_walks_);

    int*     block_smem_pre_nodes    = arena_.acquire<int>(num_walks_);
    int*     block_smem_pre_starts   = arena_.acquire<int>(num_walks_);
    int*     block_smem_pre_counts   = arena_.acquire<int>(num_walks_);
    int*     block_global_pre_nodes  = arena_.acquire<int>(num_walks_);
    int*     block_global_pre_starts = arena_.acquire<int>(num_walks_);
    int*     block_global_pre_counts = arena_.acquire<int>(num_walks_);

    int*     block_smem_nodes    = arena_.acquire<int>(num_walks_);
    int*     block_smem_starts   = arena_.acquire<int>(num_walks_);
    int*     block_smem_counts   = arena_.acquire<int>(num_walks_);
    int*     block_global_nodes  = arena_.acquire<int>(num_walks_);
    int*     block_global_starts = arena_.acquire<int>(num_walks_);
    int*     block_global_counts = arena_.acquire<int>(num_walks_);

    // contiguous so end-of-step readback is a single D2H
    constexpr int kCounterSlots = 9;
    int* tier_counters = arena_.acquire<int>(kCounterSlots);

    // caps=-1 forces every coop task to *_global (G >= 1 always)
    const bool is_index_picker =
        random_pickers::is_index_based_picker(edge_picker_type);
    const int g_cap_warp = force_global_only
        ? -1
        : (is_index_picker ? G_THRESHOLD_WARP_INDEX : G_THRESHOLD_WARP_WEIGHT);
    const int g_cap_block = force_global_only
        ? -1
        : (is_index_picker ? G_THRESHOLD_BLOCK_INDEX : G_THRESHOLD_BLOCK_WEIGHT);

    const std::size_t flag_blocks =
        (num_walks_ + block_dim_.x - 1) / block_dim_.x;

    {
        NVTX_RANGE_COLORED("NG filter alive", nvtx_colors::io_grey);
        walk_alive_flags_kernel
            <<<dim3(static_cast<unsigned>(flag_blocks)),
               block_dim_, 0, stream_>>>(
                walk_set_view, step_number, max_walk_len, num_walks_,
                alive_flags);
    }

    {
        NVTX_RANGE_COLORED("NG compact", nvtx_colors::io_grey);
        cub_partition_flagged(
            iota_src_.data(), alive_flags,
            active_walk_idx, step_num_active,
            num_walks_, stream_);
    }

    // CUB needs host num_items; blocking D2H
    int host_num_active = 0;
    {
        NVTX_RANGE_COLORED("NG num_active readback", nvtx_colors::io_grey);
        CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
            &host_num_active, step_num_active, sizeof(int),
            cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(stream_));
    }

    CUDA_CHECK_AND_CLEAR(cudaMemsetAsync(
        tier_counters, 0, kCounterSlots * sizeof(int), stream_));

    auto empty_tier = [&](const int device_counter_slot) {
        return TierTaskList{
            nullptr, nullptr, nullptr,
            &tier_counters[device_counter_slot], 0};
    };

    if (host_num_active <= 0) {
        return StepOutputs{
            nullptr, 0,
            solo_walks, &tier_counters[0], 0,
            empty_tier(3), empty_tier(4),
            empty_tier(7), empty_tier(8)};
    }

    const std::size_t active_items =
        static_cast<std::size_t>(host_num_active);
    const std::size_t active_blocks =
        (active_items + block_dim_.x - 1) / block_dim_.x;
    const dim3        active_grid(static_cast<unsigned>(active_blocks));

    {
        NVTX_RANGE_COLORED("NG gather", nvtx_colors::io_grey);
        gather_last_nodes_kernel
            <<<active_grid, block_dim_, 0, stream_>>>(
                walk_set_view,
                active_walk_idx, step_num_active,
                step_number, max_walk_len,
                last_nodes_active);
    }

    {
        NVTX_RANGE_COLORED("NG sort", nvtx_colors::index_blue);
        cub_sort_pairs(
            last_nodes_active, sorted_last_nodes,
            active_walk_idx, sorted_active_idx,
            active_items, stream_);
    }

    {
        NVTX_RANGE_COLORED("NG RLE+scan", nvtx_colors::index_blue);
        CUDA_CHECK_AND_CLEAR(cudaMemsetAsync(
            step_run_lengths, 0, active_items * sizeof(int), stream_));
        cub_run_length_encode(
            sorted_last_nodes, unique_last_nodes,
            step_run_lengths, step_num_runs,
            active_items, stream_);
        cub_exclusive_sum(
            step_run_lengths, step_run_starts,
            active_items, stream_);
    }

    {
        NVTX_RANGE_COLORED("NG W-partition", nvtx_colors::weight_orange);
        // warp/block W boundary tracks runtime block_dim_.x
        partition_by_w_kernel<<<active_grid, block_dim_, 0, stream_>>>(
            unique_last_nodes, step_run_starts, step_run_lengths,
            step_num_runs, sorted_active_idx,
            w_threshold_warp_,
            static_cast<int>(block_dim_.x),
            solo_walks,
            warp_nodes_w,  warp_walk_starts_w,  warp_walk_counts_w,
            block_nodes_w, block_walk_starts_w, block_walk_counts_w,
            tier_counters);
    }

    {
        NVTX_RANGE_COLORED("NG G-partition (warp)", nvtx_colors::weight_orange);
        partition_by_g_kernel<<<active_grid, block_dim_, 0, stream_>>>(
            warp_nodes_w, warp_walk_starts_w, warp_walk_counts_w,
            &tier_counters[1],
            count_ts_group_per_node,
            g_cap_warp,
            warp_smem_nodes,  warp_smem_starts,  warp_smem_counts,
            warp_global_nodes, warp_global_starts, warp_global_counts,
            &tier_counters[3]);
    }
    {
        NVTX_RANGE_COLORED("NG G-partition (block)", nvtx_colors::weight_orange);
        partition_by_g_kernel<<<active_grid, block_dim_, 0, stream_>>>(
            block_nodes_w, block_walk_starts_w, block_walk_counts_w,
            &tier_counters[2],
            count_ts_group_per_node,
            g_cap_block,
            block_smem_pre_nodes,  block_smem_pre_starts,  block_smem_pre_counts,
            block_global_pre_nodes, block_global_pre_starts, block_global_pre_counts,
            &tier_counters[5]);
    }

    {
        NVTX_RANGE_COLORED("NG block-task expansion (smem)",
                           nvtx_colors::weight_orange);
        expand_block_tasks_kernel<<<active_grid, block_dim_, 0, stream_>>>(
            block_smem_pre_nodes, block_smem_pre_starts, block_smem_pre_counts,
            &tier_counters[5],
            W_THRESHOLD_MULTI_BLOCK,
            block_smem_nodes, block_smem_starts, block_smem_counts,
            &tier_counters[7]);
    }
    {
        NVTX_RANGE_COLORED("NG block-task expansion (global)",
                           nvtx_colors::weight_orange);
        expand_block_tasks_kernel<<<active_grid, block_dim_, 0, stream_>>>(
            block_global_pre_nodes, block_global_pre_starts, block_global_pre_counts,
            &tier_counters[6],
            W_THRESHOLD_MULTI_BLOCK,
            block_global_nodes, block_global_starts, block_global_counts,
            &tier_counters[8]);
    }

    int counts_host[kCounterSlots] = {};
    {
        NVTX_RANGE_COLORED("NG tier-count readback", nvtx_colors::io_grey);
        CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
            counts_host, tier_counters, kCounterSlots * sizeof(int),
            cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK_AND_CLEAR(cudaStreamSynchronize(stream_));
    }

    return StepOutputs{
        sorted_active_idx, host_num_active,
        solo_walks, &tier_counters[0], counts_host[0],
        TierTaskList{warp_smem_nodes, warp_smem_starts, warp_smem_counts,
                     &tier_counters[3], counts_host[3]},
        TierTaskList{warp_global_nodes, warp_global_starts, warp_global_counts,
                     &tier_counters[4], counts_host[4]},
        TierTaskList{block_smem_nodes, block_smem_starts, block_smem_counts,
                     &tier_counters[7], counts_host[7]},
        TierTaskList{block_global_nodes, block_global_starts, block_global_counts,
                     &tier_counters[8], counts_host[8]}};
}

}  // namespace tempest

#endif  // HAS_CUDA
