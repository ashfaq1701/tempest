#ifndef NODE_GROUPED_DISPATCH_CUH
#define NODE_GROUPED_DISPATCH_CUH

#include "kernels.cuh"
#include "scheduler.cuh"
#include "../../common/picker_dispatch.cuh"
#include "../../common/nvtx.cuh"
#include "../../common/cuda_config.cuh"

namespace tempest {

#ifdef HAS_CUDA

inline void dispatch_node_grouped_kernel(
    const TemporalGraphView& view,
    const bool is_directed,
    WalkSetView walk_set_view,
    const int max_walk_len,
    const int* start_node_ids,
    const size_t num_walks,
    const bool all_starts_unconstrained,
    const RandomPickerType edge_picker_type,
    const RandomPickerType start_picker_type,
    const WalkDirection walk_direction,
    const uint64_t base_seed,
    const dim3& grid_dim,
    const dim3& block_dim,
    const cudaStream_t stream = 0,
    const bool force_global_only = false,
    const int w_threshold_warp = W_THRESHOLD_WARP) {

    if (num_walks == 0) return;

    dim3 grid = grid_dim;
    if (grid.x == 0) {
        grid.x = (num_walks + block_dim.x - 1) / block_dim.x;
    }

    const bool should_walk_forward = get_should_walk_forward(walk_direction);

    const bool starts_constrained = !all_starts_unconstrained;

    dispatch_bool(is_directed, [&](auto dir_tag) {
        constexpr bool kDir = decltype(dir_tag)::value;
        dispatch_bool(should_walk_forward, [&](auto fwd_tag) {
            constexpr bool kFwd = decltype(fwd_tag)::value;

            // unconstrained: per-walk start (no coop benefit at step 0).
            // constrained: prepop slot 0 so coop pipeline handles step 0.
            int first_coop_step;
            if (!starts_constrained) {
                NVTX_RANGE_COLORED("NG step0 pick", nvtx_colors::walk_green);
                dispatch_start_edges_kernel<kDir, kFwd>(
                    view, walk_set_view, start_node_ids,
                    /*constrained=*/starts_constrained,
                    max_walk_len, num_walks, start_picker_type,
                    base_seed, grid, block_dim, stream);
                first_coop_step = 1;
            } else {
                NVTX_RANGE_COLORED("NG step0 prepop", nvtx_colors::walk_green);
                dispatch_prepopulate_start_slot_kernel<kFwd, kDir>(
                    view, walk_set_view, start_node_ids, num_walks,
                    grid, block_dim, stream);
                first_coop_step = 0;
            }

            // loop bound max_walk_len-1: last iter would no-op anyway via kernel guards
            NodeGroupedScheduler scheduler(num_walks, block_dim, w_threshold_warp, stream);

            const std::size_t* count_ts_group_per_node =
                count_ts_group_per_node_for_dir<kDir, kFwd>(view);

            const unsigned warps_per_block = block_dim.x / 32u;

            for (int step_number = first_coop_step; step_number < max_walk_len - 1; ++step_number) {
                // step 0 uses start-picker (samples from start node)
                const RandomPickerType picker_for_step =
                    (step_number == 0) ? start_picker_type : edge_picker_type;

                auto step_outs = scheduler.run_step(
                    walk_set_view, step_number, max_walk_len,
                    count_ts_group_per_node, picker_for_step,
                    force_global_only);

                // padding is absorbing
                if (step_outs.num_active_host <= 0) break;

                NVTX_RANGE_COLORED("NG pick", nvtx_colors::walk_green);

                if (step_outs.num_solo_walks_host > 0) {
                    const size_t solo_blocks =
                        (static_cast<size_t>(step_outs.num_solo_walks_host)
                         + block_dim.x - 1) / block_dim.x;
                    const dim3 solo_grid(static_cast<unsigned>(solo_blocks));
                    dispatch_node_grouped_solo_kernel<kDir, kFwd>(
                        view, walk_set_view,
                        step_outs.solo_walks,
                        step_outs.num_solo_walks_device,
                        step_number, max_walk_len,
                        picker_for_step, base_seed,
                        solo_grid, block_dim, stream);
                }

                auto launch_coop_tier = [&](auto dispatcher,
                                            const NodeGroupedScheduler::TierTaskList& tier,
                                            const bool is_warp_tier) {
                    if (tier.num_tasks_host <= 0) return;
                    dim3 tier_grid;
                    if (is_warp_tier) {
                        const size_t blocks =
                            (static_cast<size_t>(tier.num_tasks_host) + warps_per_block - 1)
                            / warps_per_block;
                        tier_grid = dim3(static_cast<unsigned>(blocks));
                    } else {
                        tier_grid = dim3(static_cast<unsigned>(tier.num_tasks_host));
                    }
                    dispatcher(view, walk_set_view,
                               step_outs.sorted_walk_idx,
                               tier.nodes, tier.walk_starts, tier.walk_counts,
                               tier.num_tasks_device,
                               step_number, max_walk_len,
                               picker_for_step, base_seed,
                               tier_grid, block_dim, stream);
                };

                launch_coop_tier(dispatch_node_grouped_warp_smem_kernel<kDir, kFwd>,    step_outs.warp_smem,    true);
                launch_coop_tier(dispatch_node_grouped_warp_global_kernel<kDir, kFwd>,  step_outs.warp_global,  true);
                launch_coop_tier(dispatch_node_grouped_block_smem_kernel<kDir, kFwd>,   step_outs.block_smem,   false);
                launch_coop_tier(dispatch_node_grouped_block_global_kernel<kDir, kFwd>, step_outs.block_global, false);
            }

            if constexpr (!kFwd) {
                NVTX_RANGE_COLORED("NG reverse", nvtx_colors::edge_purple);
                reverse_walks_kernel<<<grid, block_dim, 0, stream>>>(
                    walk_set_view, num_walks);
            }
        });
    });
}

#endif // HAS_CUDA

} // namespace tempest

#endif // NODE_GROUPED_DISPATCH_CUH
