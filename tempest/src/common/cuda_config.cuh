#ifndef CUDA_CONST_H
#define CUDA_CONST_H

#include <cstddef>

// visible on CPU-only builds: used as a default-parameter value in public headers.
constexpr size_t BLOCK_DIM = 256;

// W <= W_THRESHOLD_WARP -> solo; W <= BLOCK_DIM -> warp; W > BLOCK_DIM -> block.
constexpr int W_THRESHOLD_WARP = 4;

#ifdef HAS_CUDA

#include <thrust/execution_policy.h>

constexpr auto DEVICE_EXECUTION_POLICY = thrust::device;

// above this, a block task splits into ceil(W/cap) disjoint block-tasks.
constexpr int W_THRESHOLD_MULTI_BLOCK = 8192;

// G-fit thresholds per tier × picker class. per-group bytes: index=16, weight=24.
constexpr int G_THRESHOLD_BLOCK_INDEX  = 2800;
constexpr int G_THRESHOLD_BLOCK_WEIGHT = 1800;
constexpr int G_THRESHOLD_WARP_INDEX   = 340;
constexpr int G_THRESHOLD_WARP_WEIGHT  = 220;

#endif  // HAS_CUDA

#endif  // CUDA_CONST_H
