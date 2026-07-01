#ifndef CUDA_ERROR_HANDLERS
#define CUDA_ERROR_HANDLERS

#ifdef HAS_CUDA

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// clears sticky prior error so a failure here is unambiguous.
#define CUDA_CHECK_AND_CLEAR(call) do { \
    cudaGetLastError(); /* clear any prior error */ \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        std::ostringstream _oss; \
        _oss << "CUDA error in " << __FILE__ << ":" << __LINE__ \
             << " code=" << _err << " (" << cudaGetErrorString(_err) << ") " \
             << "call=" << #call; \
        throw std::runtime_error(_oss.str()); \
    } \
} while(0)

// catches launch-config errors only. async errors surface at next sync.
#define CUDA_KERNEL_CHECK_ASYNC(msg) do { \
    cudaError_t _err = cudaGetLastError(); \
    if (_err != cudaSuccess) { \
        std::ostringstream _oss; \
        _oss << "CUDA launch error in " << __FILE__ << ":" << __LINE__ \
             << " code=" << _err << " (" << cudaGetErrorString(_err) << ") " \
             << "msg=" << (msg); \
        throw std::runtime_error(_oss.str()); \
    } \
} while(0)

// syncs the device — never use in the streaming hot path.
#define CUDA_KERNEL_CHECK_SYNC(msg) do { \
    cudaError_t _err = cudaGetLastError(); \
    if (_err != cudaSuccess) { \
        std::ostringstream _oss; \
        _oss << "CUDA launch error in " << __FILE__ << ":" << __LINE__ \
             << " code=" << _err << " (" << cudaGetErrorString(_err) << ") " \
             << "msg=" << (msg); \
        throw std::runtime_error(_oss.str()); \
    } \
    _err = cudaDeviceSynchronize(); \
    if (_err != cudaSuccess) { \
        std::ostringstream _oss; \
        _oss << "CUDA sync error in " << __FILE__ << ":" << __LINE__ \
             << " code=" << _err << " (" << cudaGetErrorString(_err) << ") " \
             << "msg=" << (msg) << " (during synchronization)"; \
        throw std::runtime_error(_oss.str()); \
    } \
} while(0)

// debug: sync (errors surface at the offending kernel). release: async.
#ifdef NDEBUG
    #define CUDA_KERNEL_CHECK(msg) CUDA_KERNEL_CHECK_ASYNC(msg)
#else
    #define CUDA_KERNEL_CHECK(msg) CUDA_KERNEL_CHECK_SYNC(msg)
#endif

// opt-in per-step stream sync for debugging async-pipeline bugs; no-op otherwise.
#ifdef TEMPEST_DEBUG_SYNC
    #define TEMPEST_STREAM_SYNC(stream, msg) do { \
        cudaError_t _err = cudaStreamSynchronize(stream); \
        if (_err != cudaSuccess) { \
            std::ostringstream _oss; \
            _oss << "CUDA sync error in " << __FILE__ << ":" << __LINE__ \
                 << " code=" << _err << " (" << cudaGetErrorString(_err) \
                 << ") msg=" << (msg); \
            throw std::runtime_error(_oss.str()); \
        } \
    } while(0)
#else
    #define TEMPEST_STREAM_SYNC(stream, msg) do {} while(0)
#endif

#define CHECK_CURAND(call) do { \
    curandStatus_t _status = (call); \
    if (_status != CURAND_STATUS_SUCCESS) { \
        std::ostringstream _oss; \
        _oss << "cuRAND error in " << __FILE__ << ":" << __LINE__ \
             << " status=" << static_cast<int>(_status) << " " \
             << "call=" << #call; \
        throw std::runtime_error(_oss.str()); \
    } \
} while(0)

#define CUB_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        std::ostringstream _oss; \
        _oss << "CUB error in " << __FILE__ << ":" << __LINE__ \
             << " code=" << _err << " (" << cudaGetErrorString(_err) << ") " \
             << "call=" << #call; \
        throw std::runtime_error(_oss.str()); \
    } \
} while(0)

inline void clearCudaErrorState() {
    cudaGetLastError();
}

// RAII current-device pin. Snapshots the calling thread's current CUDA
// device, switches to `target_device`, restores the prior device on scope
// exit. Lets a multi-GPU process pin a single Tempest to one
// GPU while another library (e.g. PyTorch) owns a different GPU on the
// same thread.
struct CudaDeviceGuard {
    int prev_device_ = -1;
    bool active_     = false;

    explicit CudaDeviceGuard(const int target_device) {
        if (target_device < 0) return;
        if (cudaGetDevice(&prev_device_) != cudaSuccess) {
            prev_device_ = -1;
            return;
        }
        if (prev_device_ != target_device) {
            if (cudaSetDevice(target_device) != cudaSuccess) return;
        }
        active_ = true;
    }

    ~CudaDeviceGuard() {
        if (!active_) return;
        if (prev_device_ >= 0) {
            // best-effort restore; swallow errors so dtor stays noexcept
            (void)cudaSetDevice(prev_device_);
        }
    }

    CudaDeviceGuard(const CudaDeviceGuard&)            = delete;
    CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;
    CudaDeviceGuard(CudaDeviceGuard&&)                 = delete;
    CudaDeviceGuard& operator=(CudaDeviceGuard&&)      = delete;
};

#endif // HAS_CUDA

#endif // CUDA_ERROR_HANDLERS
