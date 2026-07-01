#include "buffer.cuh"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef HAS_CUDA
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#endif

namespace buffer_detail {

// byte-pattern checks let fill() use cudaMemsetAsync for common values.
template <typename T>
inline bool is_zero_value(const T& v) {
    constexpr size_t N = sizeof(T);
    alignas(T) unsigned char zero[N] = {};
    return std::memcmp(&v, zero, N) == 0;
}

template <typename T>
inline bool is_all_0xff_value(const T& v) {
    constexpr size_t N = sizeof(T);
    alignas(T) unsigned char ffs[N];
    std::memset(ffs, 0xFF, N);
    return std::memcmp(&v, ffs, N) == 0;
}

} // namespace buffer_detail

template <typename T>
void Buffer<T>::fill(const T& value) {
    if (!data_ || size_ == 0) return;

#ifdef HAS_CUDA
    if (use_gpu_) {
        if (buffer_detail::is_zero_value(value)) {
            CUDA_CHECK_AND_CLEAR(cudaMemsetAsync(
                data_, 0, size_ * sizeof(T)));
            return;
        }
        if (buffer_detail::is_all_0xff_value(value)) {
            CUDA_CHECK_AND_CLEAR(cudaMemsetAsync(
                data_, 0xFF, size_ * sizeof(T)));
            return;
        }
        thrust::fill_n(
            thrust::device,
            thrust::device_pointer_cast(data_),
            size_,
            value);
        return;
    }
#endif
    std::fill(data_, data_ + size_, value);
}

template <typename T>
std::vector<T> Buffer<T>::to_host_vector() const {
    std::vector<T> out(size_);
    if (size_ == 0) return out;
#ifdef HAS_CUDA
    if (use_gpu_) {
        CUDA_CHECK_AND_CLEAR(cudaMemcpy(
            out.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }
#endif
    std::memcpy(out.data(), data_, size_ * sizeof(T));
    return out;
}

template class Buffer<int>;
template class Buffer<int64_t>;
template class Buffer<size_t>;
template class Buffer<float>;
template class Buffer<double>;
