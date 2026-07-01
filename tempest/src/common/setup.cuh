#ifndef CUDA_SETUP_H
#define CUDA_SETUP_H

#include <cstddef>

#ifdef HAS_CUDA
#include <curand_kernel.h>
#include "cuda_config.cuh"
#endif

#include "macros.cuh"

#ifdef HAS_CUDA

HOST inline std::pair<size_t, size_t> get_optimal_launch_params(
    const size_t data_size,
    const cudaDeviceProp* device_prop,
    size_t block_dim=BLOCK_DIM) {

    size_t grid_dim = (data_size + block_dim - 1) / block_dim;
    const size_t min_grid_size = 2 * device_prop->multiProcessorCount;
    grid_dim = std::max(grid_dim, min_grid_size);
    return {grid_dim, block_dim};
}

#endif

#endif //CUDA_SETUP_H
