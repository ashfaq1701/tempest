#include "node_features.cuh"

#include <cstring>
#include <stdexcept>
#include <omp.h>

void node_features::ensure_size(
    TemporalGraphData& data,
    const int max_node_id) {

    if (data.node_feature_dim == 0) {
        return;
    }

    const size_t needed_values =
        static_cast<size_t>(max_node_id + 1) * data.node_feature_dim;
    const size_t current_values = data.node_features.size();

    if (needed_values <= current_values) return;

    data.node_features.resize(needed_values);

    std::memset(
        data.node_features.data() + current_values,
        0,
        (needed_values - current_values) * sizeof(float));
}

void node_features::set_node_features(
    TemporalGraphData& data,
    const int max_node_id,
    const int* node_ids,
    const size_t num_nodes,
    const float* node_features_src,
    const size_t feature_dim) {

    if (num_nodes == 0) {
        data.node_feature_dim = feature_dim;
        return;
    }

    if (feature_dim == 0) {
        throw std::runtime_error(
            "feature_dim must be greater than 0 when setting node features");
    }

    if (data.node_feature_dim != 0 &&
        data.node_feature_dim != feature_dim) {
        throw std::runtime_error(
            "feature_dim mismatch with existing TemporalGraphData node features");
    }

    data.node_feature_dim = feature_dim;
    ensure_size(data, max_node_id);

    float* dst_base = data.node_features.data();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_nodes; ++i) {
        const int node_id = node_ids[i];
        float* dst = dst_base +
            static_cast<size_t>(node_id) * feature_dim;
        const float* src = node_features_src + (i * feature_dim);
        std::memcpy(dst, src, feature_dim * sizeof(float));
    }
}
