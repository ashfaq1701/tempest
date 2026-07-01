#ifndef STRUCTS_H
#define STRUCTS_H

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <utility>
#include "../common/error_handlers.cuh"
#include "../common/macros.cuh"

struct Edge {
    int u;
    int i;
    int64_t ts;

    const float* features;
    int feature_dim;

    HOST DEVICE Edge(): u(-1), i(-1), ts(-1), features(nullptr), feature_dim(0) {}

    HOST DEVICE explicit Edge(const int u, const int i, const int64_t ts, const float* features, const int feature_dim)
        : u(u), i(i), ts(ts), features(features), feature_dim(feature_dim) {}

    HOST DEVICE explicit Edge(const int u, const int i, const int64_t ts)
        : u(u), i(i), ts(ts), features(nullptr), feature_dim(0) {}

    HOST DEVICE Edge& operator=(const Edge& other) {
        if (this != &other) {
            u = other.u;
            i = other.i;
            ts = other.ts;
            features = other.features;
            feature_dim = other.feature_dim;
        }
        return *this;
    }
};

struct InternalEdge : Edge {
    int64_t edge_id;

    HOST DEVICE InternalEdge()
        : Edge(), edge_id(-1) {}

    HOST DEVICE InternalEdge(const int u, const int i, const int64_t ts, const int64_t edge_id)
        : Edge(u, i, ts), edge_id(edge_id) {}
};

struct SizeRange {
    size_t from;
    size_t to;

    HOST DEVICE SizeRange(): from(0), to(0) {}

    HOST DEVICE explicit SizeRange(const size_t f, const size_t t) : from(f), to(t) {}

    HOST DEVICE SizeRange& operator=(const SizeRange& other)
    {
        if (this != &other)
        {
            from = other.from;
            to = other.to;
        }
        return *this;
    }
};

template <typename T>
struct DataBlock {
    T* data = nullptr;
    size_t size = 0;
    bool use_gpu = false;

    DataBlock() = default;

    HOST DataBlock(const size_t size, const bool use_gpu) : size(size), use_gpu(use_gpu) {
        if (size == 0) {
            data = nullptr;
        }
        #ifdef HAS_CUDA
        else if (use_gpu) {
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&data, size * sizeof(T)));
        }
        #endif
        else {
            data = static_cast<T *>(malloc(sizeof(T) * size));
        }
    }

    DataBlock(const DataBlock&) = delete;
    DataBlock& operator=(const DataBlock&) = delete;

    // explicit moves so the moved-from pointer is nulled (no double-free).
    HOST DataBlock(DataBlock&& other) noexcept
        : data(other.data), size(other.size), use_gpu(other.use_gpu) {
        other.data = nullptr;
        other.size = 0;
    }

    HOST DataBlock& operator=(DataBlock&& other) noexcept {
        if (this != &other) {
            release();
            data    = other.data;
            size    = other.size;
            use_gpu = other.use_gpu;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }

    HOST ~DataBlock() noexcept { release(); }

private:
    // bare cudaFree + cudaGetLastError so destructor stays noexcept.
    HOST void release() noexcept {
        if (!data) return;
        #ifdef HAS_CUDA
        if (use_gpu) {
            cudaFree(data);
            cudaGetLastError();
        } else
        #endif
        {
            free(data);
        }
        data = nullptr;
    }
};

#endif // STRUCTS_H
