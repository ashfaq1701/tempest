#ifndef WALKS_WITH_EDGE_FEATURES_HOST_CUH
#define WALKS_WITH_EDGE_FEATURES_HOST_CUH

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <omp.h>

#include "../../common/const.cuh"
#include "../buffer.cuh"
#include "walk_set_host.cuh"

// host-resident bundle of walks plus per-edge feature rows.
struct WalksWithEdgeFeaturesHost {
    WalkSetHost   walk_set;
    Buffer<float> walk_edge_features{/*use_gpu=*/false};
    int           feature_dim = 0;

    WalksWithEdgeFeaturesHost() = default;

    WalksWithEdgeFeaturesHost(WalkSetHost ws, const int fdim)
        : walk_set(std::move(ws)), feature_dim(fdim) {
        const size_t n = walk_set.edge_ids_size();
        if (feature_dim > 0 && n > 0) {
            walk_edge_features.resize(n * static_cast<size_t>(feature_dim));
            walk_edge_features.fill(0.0f);
        }
    }

    WalksWithEdgeFeaturesHost(const WalksWithEdgeFeaturesHost&) = delete;
    WalksWithEdgeFeaturesHost& operator=(const WalksWithEdgeFeaturesHost&) = delete;
    WalksWithEdgeFeaturesHost(WalksWithEdgeFeaturesHost&&) noexcept = default;
    WalksWithEdgeFeaturesHost& operator=(WalksWithEdgeFeaturesHost&&) noexcept = default;

    size_t size() const noexcept { return walk_set.size(); }
    WalksIterator walks_begin() const { return walk_set.walks_begin(); }
    WalksIterator walks_end()   const { return walk_set.walks_end(); }

    HostRelease release_walk_edge_features() {
        return walk_edge_features.release_host();
    }

    // edge_features_src must be host-resident; empty slots stay zero.
    void populate_walk_edge_features(const float* edge_features_src) {
        if (!edge_features_src || feature_dim == 0) return;

        const int64_t* edge_ids = walk_set.edge_ids_ptr();
        float* dst = walk_edge_features.data();
        const size_t n = walk_set.edge_ids_size();
        const int fd = feature_dim;

        if (!edge_ids || !dst || n == 0) return;

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            const int64_t eid = edge_ids[i];
            if (eid == EMPTY_EDGE_ID) continue;
            const float* src = edge_features_src + (eid * fd);
            float* out = dst + (i * fd);
            std::memcpy(out, src, fd * sizeof(float));
        }
    }
};

#endif // WALKS_WITH_EDGE_FEATURES_HOST_CUH
