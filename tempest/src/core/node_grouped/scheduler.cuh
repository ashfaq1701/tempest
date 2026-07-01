#ifndef NODE_GROUPED_SCHEDULER_CUH
#define NODE_GROUPED_SCHEDULER_CUH

#include <cstddef>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

#include "../../data/buffer.cuh"
#include "../../data/device_arena.cuh"
#include "../../data/walk_set/walk_set_view.cuh"
#include "../../data/enums.cuh"

namespace tempest {

#ifdef HAS_CUDA

// StepOutputs pointers are valid only until the next run_step call
class NodeGroupedScheduler {
public:
    struct TierTaskList {
        int* nodes;
        int* walk_starts;
        int* walk_counts;
        int* num_tasks_device;
        int  num_tasks_host;
    };

    struct StepOutputs {
        int* sorted_walk_idx;
        int  num_active_host;

        int* solo_walks;
        int* num_solo_walks_device;
        int  num_solo_walks_host;

        TierTaskList warp_smem;
        TierTaskList warp_global;
        TierTaskList block_smem;
        TierTaskList block_global;
    };

    NodeGroupedScheduler(std::size_t num_walks,
                         dim3 block_dim,
                         int w_threshold_warp,
                         cudaStream_t stream);

    // two blocking D2Hs per call: num_active and tier counts
    StepOutputs run_step(WalkSetView walk_set_view,
                         int step_number,
                         int max_walk_len,
                         const std::size_t* count_ts_group_per_node,
                         RandomPickerType edge_picker_type,
                         bool force_global_only = false);

private:
    std::size_t  num_walks_;
    int          num_walks_int_;
    dim3         block_dim_;
    int          w_threshold_warp_;
    cudaStream_t stream_;

    DeviceArena arena_;
    Buffer<int> iota_src_;
};

#endif  // HAS_CUDA

}  // namespace tempest

#endif  // NODE_GROUPED_SCHEDULER_CUH
