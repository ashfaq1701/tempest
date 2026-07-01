#ifndef DATA_DEVICE_ARENA_CUH
#define DATA_DEVICE_ARENA_CUH

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../common/macros.cuh"
#include "../common/error_handlers.cuh"

// chunked bump allocator. acquire returns 256-byte-aligned pointers that stay
// valid until reset() or destruction. not stream-aware.
class DeviceArena {
public:
    DeviceArena() noexcept : use_gpu_(false) {}

    explicit DeviceArena(const bool use_gpu) noexcept : use_gpu_(use_gpu) {}

    DeviceArena(const bool use_gpu, const size_t initial_capacity_bytes)
        : use_gpu_(use_gpu) {
        if (initial_capacity_bytes > 0) {
            allocate_chunk(initial_capacity_bytes);
        }
    }

    ~DeviceArena() {
        destroy_all();
    }

    DeviceArena(const DeviceArena&)            = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    DeviceArena(DeviceArena&& other) noexcept
        : chunks_(std::move(other.chunks_)),
          cur_chunk_(other.cur_chunk_),
          offset_in_cur_(other.offset_in_cur_),
          use_gpu_(other.use_gpu_) {
        other.cur_chunk_     = 0;
        other.offset_in_cur_ = 0;
    }

    DeviceArena& operator=(DeviceArena&& other) noexcept {
        if (this != &other) {
            destroy_all();
            chunks_        = std::move(other.chunks_);
            cur_chunk_     = other.cur_chunk_;
            offset_in_cur_ = other.offset_in_cur_;
            use_gpu_       = other.use_gpu_;
            other.cur_chunk_     = 0;
            other.offset_in_cur_ = 0;
        }
        return *this;
    }

    template <typename T>
    T* acquire(const size_t n) {
        if (n == 0) return nullptr;

        constexpr size_t kAlign = 256;
        const size_t bytes_needed = n * sizeof(T);

        if (cur_chunk_ < chunks_.size()) {
            const size_t aligned =
                (offset_in_cur_ + kAlign - 1) & ~(kAlign - 1);
            const Chunk& cur = chunks_[cur_chunk_];
            if (aligned + bytes_needed <= cur.capacity) {
                offset_in_cur_ = aligned + bytes_needed;
                return reinterpret_cast<T*>(cur.base + aligned);
            }
        }

        // reuse a later chunk left behind on an earlier peak step.
        const size_t start = chunks_.empty() ? 0 : cur_chunk_ + 1;
        for (size_t c = start; c < chunks_.size(); ++c) {
            if (chunks_[c].capacity >= bytes_needed) {
                cur_chunk_     = c;
                offset_in_cur_ = bytes_needed;
                return reinterpret_cast<T*>(chunks_[c].base);
            }
        }

        // grow geometrically so the arena converges on repeat calls.
        const size_t prev_cap =
            chunks_.empty() ? size_t{0} : chunks_.back().capacity;
        size_t new_cap = prev_cap * 2;
        if (new_cap < bytes_needed) new_cap = bytes_needed;

        allocate_chunk(new_cap);
        cur_chunk_     = chunks_.size() - 1;
        offset_in_cur_ = bytes_needed;
        return reinterpret_cast<T*>(chunks_.back().base);
    }

    void reset() noexcept {
        cur_chunk_     = 0;
        offset_in_cur_ = 0;
    }

    size_t capacity_bytes() const noexcept {
        size_t total = 0;
        for (const auto& c : chunks_) total += c.capacity;
        return total;
    }

    size_t used_bytes() const noexcept {
        // loose metric — earlier chunks may not have been fully consumed.
        size_t total = 0;
        for (size_t i = 0; i < cur_chunk_ && i < chunks_.size(); ++i) {
            total += chunks_[i].capacity;
        }
        total += offset_in_cur_;
        return total;
    }

    bool is_gpu() const noexcept { return use_gpu_; }

private:
    struct Chunk {
        char*  base;
        size_t capacity;
    };

    void allocate_chunk(size_t capacity) {
        char* base = nullptr;
#ifdef HAS_CUDA
        if (use_gpu_) {
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&base, capacity));
        } else
#endif
        {
            base = static_cast<char*>(std::malloc(capacity));
            if (!base) throw std::bad_alloc();
        }
        chunks_.push_back(Chunk{base, capacity});
    }

    void destroy_all() noexcept {
        for (auto& c : chunks_) {
            if (!c.base) continue;
#ifdef HAS_CUDA
            if (use_gpu_) {
                cudaFree(c.base);
                cudaGetLastError();
            } else
#endif
            {
                std::free(c.base);
            }
        }
        chunks_.clear();
        cur_chunk_     = 0;
        offset_in_cur_ = 0;
    }

    std::vector<Chunk> chunks_;
    size_t cur_chunk_     = 0;
    size_t offset_in_cur_ = 0;
    bool   use_gpu_       = false;
};

#endif // DATA_DEVICE_ARENA_CUH
