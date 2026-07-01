#ifndef PY_INTERFACE_BUFFER_TO_NUMPY_CUH
#define PY_INTERFACE_BUFFER_TO_NUMPY_CUH

#include <cstdlib>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../src/data/buffer.cuh"

namespace py = pybind11;

inline py::capsule make_capsule(HostRelease r) {
    auto* ctx = new HostRelease(r);
    return py::capsule(ctx, +[](void* vp) noexcept {
        auto* c = static_cast<HostRelease*>(vp);
        if (c->ptr) {
            std::free(c->ptr);
        }
        delete c;
    });
}

#endif // PY_INTERFACE_BUFFER_TO_NUMPY_CUH
