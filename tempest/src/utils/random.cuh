#ifndef UTILS_RANDOM_H
#define UTILS_RANDOM_H

#include <random>
#include <ctime>
#include <algorithm>
#include "../common/error_handlers.cuh"

#ifdef HAS_CUDA
#include <thrust/device_ptr.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#endif

#include "../common/macros.cuh"
#include "../common/cuda_config.cuh"

thread_local static std::mt19937 thread_local_gen{std::random_device{}()};

HOST DEVICE inline int generate_random_number_bounded_by(const int max_bound, const double rand_number) {
    return static_cast<int>(rand_number * max_bound);
}

HOST DEVICE inline bool generate_random_boolean(const double rand_number) {
    return rand_number >= 0.5;
}

HOST DEVICE inline int pick_random_number(const int a, const int b, const double rand_number) {
    return generate_random_boolean(rand_number) ? a : b;
}

// Seeded overload: std::shuffle's swap sequence depends only on the RNG and
// the range length, not the element type — so calling this with the SAME seed
// and SAME size on two parallel arrays applies the identical permutation to
// both (used to co-shuffle walk seeds and their per-walk cutoffs).
template <typename T>
HOST void shuffle_vector_host(T* vec, size_t size, unsigned int seed) {
    std::mt19937 rng(seed);
    std::shuffle(vec, vec + size, rng);
}

template <typename T>
HOST void shuffle_vector_host(T* vec, size_t size) {
    std::random_device rd;
    shuffle_vector_host(vec, size, rd());
}

#ifdef HAS_CUDA

// Seeded overload — same-seed/same-size ⇒ identical permutation across arrays,
// regardless of element type (parallel-array co-shuffle on device).
template <typename T>
HOST void shuffle_vector_device(T* data, size_t size, unsigned int seed) {
    thrust::device_ptr<T> d_data(data);
    thrust::default_random_engine random_engine(seed);
    thrust::shuffle(DEVICE_EXECUTION_POLICY, d_data, d_data + size, random_engine);
    CUDA_KERNEL_CHECK("After thrust shuffle in shuffle_vector_device (seeded)");
}

template <typename T>
HOST void shuffle_vector_device(T* data, size_t size) {
    shuffle_vector_device(data, size, static_cast<unsigned int>(std::time(nullptr)));
}

#endif

#endif // UTILS_RANDOM_H
