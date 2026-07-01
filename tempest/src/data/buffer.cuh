#ifndef DATA_BUFFER_CUH
#define DATA_BUFFER_CUH

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../common/macros.cuh"
#include "../common/error_handlers.cuh"

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

// host-side single-element read that picks cudaMemcpy or direct load.
template <typename T>
HOST inline T read_one_host_safe(const T* p, const bool use_gpu) {
#ifdef HAS_CUDA
    if (use_gpu) {
        T out;
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(&out, p, sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }
#else
    (void)use_gpu;
#endif
    return *p;
}

// POD returned by Buffer<T>::release_host() so the downstream deleter can free it.
struct HostRelease {
    void*  ptr   = nullptr;
    size_t bytes = 0;
};

template <typename T>
class Buffer {
public:
    Buffer() noexcept
        : data_(nullptr), size_(0), capacity_(0), use_gpu_(false) {}

    explicit Buffer(const bool use_gpu) noexcept
        : data_(nullptr), size_(0), capacity_(0), use_gpu_(use_gpu) {}

    Buffer(const size_t n, const bool use_gpu) : Buffer(use_gpu) {
        reserve(n);
        size_ = n;
    }

    ~Buffer() {
        deallocate();
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_),
          use_gpu_(other.use_gpu_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            deallocate();
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            use_gpu_ = other.use_gpu_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    HOST DEVICE T*       data()       noexcept { return data_; }
    HOST DEVICE const T* data() const noexcept { return data_; }
    HOST DEVICE size_t   size()     const noexcept { return size_; }
    HOST DEVICE size_t   capacity() const noexcept { return capacity_; }
    HOST DEVICE bool     empty()    const noexcept { return size_ == 0; }
    HOST DEVICE bool     is_gpu()   const noexcept { return use_gpu_; }

    void reserve(size_t n) {
        if (n <= capacity_) return;
        size_t new_cap = capacity_ == 0 ? n : std::max(capacity_ * 2, n);
        T* const   old_data       = data_;
        const size_t old_capacity = capacity_;
        const bool   old_use_gpu  = use_gpu_;

        T* new_ptr = raw_allocate(new_cap);
        if (old_data && size_ > 0) {
            raw_copy_device_to_device_or_host_to_host(new_ptr, old_data, size_);
        }
        deallocate_ptr(old_data, old_capacity, old_use_gpu);
        data_ = new_ptr;
        capacity_ = new_cap;
    }

    void resize(size_t n) {
        if (n > capacity_) reserve(n);
        size_ = n;
    }

    void clear() noexcept { size_ = 0; }

    void shrink_to_fit_empty() {
        deallocate();
        size_ = 0;
        capacity_ = 0;
    }

#ifdef HAS_CUDA
    void append_from_host(const T* src, const size_t n, cudaStream_t stream) {
        if (n == 0) return;
        const size_t new_size = size_ + n;
        reserve(new_size);
        if (use_gpu_) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
                data_ + size_, src, n * sizeof(T),
                cudaMemcpyHostToDevice, stream));
        } else {
            std::memcpy(data_ + size_, src, n * sizeof(T));
        }
        size_ = new_size;
    }
#endif

    void append_from_host(const T* src, const size_t n) {
        if (n == 0) return;
        const size_t new_size = size_ + n;
        reserve(new_size);
#ifdef HAS_CUDA
        if (use_gpu_) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(
                data_ + size_, src, n * sizeof(T), cudaMemcpyHostToDevice));
        } else
#endif
        {
            std::memcpy(data_ + size_, src, n * sizeof(T));
        }
        size_ = new_size;
    }

    void append_from_device(const T* src, const size_t n) {
#ifdef HAS_CUDA
        if (!use_gpu_) {
            throw std::runtime_error(
                "Buffer::append_from_device called on host buffer");
        }
        if (n == 0) return;
        const size_t new_size = size_ + n;
        reserve(new_size);
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            data_ + size_, src, n * sizeof(T), cudaMemcpyDeviceToDevice));
        size_ = new_size;
#else
        (void)src; (void)n;
        throw std::runtime_error("CUDA not compiled in");
#endif
    }

    void fill(const T& value);

    // caller must sync the stream before reading dst.
    void copy_to_host_async(T* dst, size_t n
#ifdef HAS_CUDA
                            , cudaStream_t stream = 0
#endif
                            ) const {
        if (n == 0) return;
#ifdef HAS_CUDA
        if (use_gpu_) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpyAsync(
                dst, data_, n * sizeof(T),
                cudaMemcpyDeviceToHost, stream));
            return;
        }
#endif
        std::memcpy(dst, data_, n * sizeof(T));
    }

    void drop_front(size_t n) {
        if (n == 0 || data_ == nullptr) return;
        if (n >= size_) { size_ = 0; return; }
        const size_t remaining = size_ - n;
#ifdef HAS_CUDA
        if (use_gpu_) {
            // D2D cudaMemcpy doesn't guarantee non-overlap; bounce via scratch.
            Buffer<T> scratch(true);
            scratch.resize(remaining);
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(
                scratch.data(), data_ + n, remaining * sizeof(T),
                cudaMemcpyDeviceToDevice));
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(
                data_, scratch.data(), remaining * sizeof(T),
                cudaMemcpyDeviceToDevice));
        } else
#endif
        {
            std::memmove(data_, data_ + n, remaining * sizeof(T));
        }
        size_ = remaining;
    }

    std::vector<T> to_host_vector() const;

    HostRelease release_host() {
        if (use_gpu_) {
            throw std::logic_error(
                "Buffer::release_host called on a device buffer");
        }
        HostRelease r{ data_, capacity_ * sizeof(T) };
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
        return r;
    }

private:
    T*     data_;
    size_t size_;
    size_t capacity_;
    bool   use_gpu_;

    T* raw_allocate(size_t n) {
        if (n == 0) return nullptr;
        T* p = nullptr;
#ifdef HAS_CUDA
        if (use_gpu_) {
            CUDA_CHECK_AND_CLEAR(cudaMalloc(&p, n * sizeof(T)));
            return p;
        }
#endif
        p = static_cast<T*>(std::malloc(n * sizeof(T)));
        if (!p) throw std::bad_alloc();
        return p;
    }

    void raw_copy_device_to_device_or_host_to_host(
        T* dst, const T* src, size_t n) {
#ifdef HAS_CUDA
        if (use_gpu_) {
            CUDA_CHECK_AND_CLEAR(cudaMemcpy(
                dst, src, n * sizeof(T), cudaMemcpyDeviceToDevice));
            return;
        }
#endif
        std::memcpy(dst, src, n * sizeof(T));
    }

    static void deallocate_ptr(
        T* ptr, [[maybe_unused]] size_t cap,
        [[maybe_unused]] bool use_gpu) noexcept {
        if (!ptr) return;
#ifdef HAS_CUDA
        if (use_gpu) {
            cudaFree(ptr);
            cudaGetLastError();
            return;
        }
#endif
        std::free(ptr);
    }

    void deallocate() noexcept {
        deallocate_ptr(data_, capacity_, use_gpu_);
        data_ = nullptr;
    }
};

#endif // DATA_BUFFER_CUH
