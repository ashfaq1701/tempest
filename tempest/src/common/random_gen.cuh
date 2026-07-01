#ifndef RANDOM_GEN_H
#define RANDOM_GEN_H

#ifdef HAS_CUDA
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#endif

#include <chrono>
#include <omp.h>
#include <random>

#include "error_handlers.cuh"
#include "../data/buffer.cuh"

inline uint64_t secure_random_seed() {
    std::random_device rd;
    uint64_t seed = 0;

    seed |= static_cast<uint64_t>(rd()) << 32;
    seed |= static_cast<uint64_t>(rd());

    return seed;
}

inline Buffer<double> generate_n_random_numbers_cpu(const size_t n) {
    Buffer<double> random_numbers(n, false);

    std::random_device rd;
    double* out = random_numbers.data();

    #pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        std::mt19937 gen(rd() + thread_id);
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        #pragma omp for
        for (size_t i = 0; i < n; ++i) {
            out[i] = dis(gen);
        }
    }

    return random_numbers;
}

#ifdef HAS_CUDA

// init once per thread; draw_u01_philox steps the counter per draw.
using PhiloxState = curandStatePhilox4_32_10_t;

DEVICE __forceinline__ void init_philox_state(
    PhiloxState& state,
    const uint64_t base_seed,
    const uint64_t walk_idx,
    const uint64_t offset = 0ULL) {
    // subsequence=walk_idx gives each walk its own stream; offset avoids
    // counter reuse across successive step-kernel launches.
    curand_init(base_seed, walk_idx, offset, &state);
}

DEVICE __forceinline__ double draw_u01_philox(PhiloxState& state) {
    return curand_uniform_double(&state);
}

inline Buffer<double> generate_n_random_numbers_gpu(const size_t n) {
    Buffer<double> d_random_numbers(n, true);

    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));

    const auto seed = secure_random_seed();
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));

    CHECK_CURAND(curandGenerateUniformDouble(gen, d_random_numbers.data(), n));
    CHECK_CURAND(curandDestroyGenerator(gen));

    return d_random_numbers;
}

#endif

inline Buffer<double> generate_n_random_numbers(const size_t n, const bool use_gpu) {
    #ifdef HAS_CUDA
    if (use_gpu) {
        return generate_n_random_numbers_gpu(n);
    }
    else
    #endif
    {
        return generate_n_random_numbers_cpu(n);
    }
}

#endif // RANDOM_GEN_H
