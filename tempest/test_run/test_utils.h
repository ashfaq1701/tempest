#ifndef TEST_RUN_UTILS_H
#define TEST_RUN_UTILS_H

#include <iostream>
#include <filesystem>
#include <regex>
#include <algorithm>
#include <fstream>

#include "../src/data/walk_set/walk_set_host.cuh"
#include "../src/data/walk_set/walks_with_edge_features_host.cuh"
#include "../src/data/structs.cuh"

inline void print_tempests_with_times(
    const WalkSetHost& walk_set,
    const size_t n = 10)
{
    size_t count = 0;
    for (auto it = walk_set.walks_begin(); it != walk_set.walks_end() && count < n; ++it, ++count) {
        const auto& walk = *it;

        std::cout << "Length: " << walk.size() << ", Walk: ";
        for (const auto& step : walk) {
            std::cout << "(" << step.node << ", " << step.timestamp << "), ";
        }
        std::cout << std::endl;
    }
}

inline void print_tempests_with_times_and_weights(
    const WalksWithEdgeFeaturesHost& walks_with_edge_features,
    const size_t n = 10)
{
    const WalkSetHost& walk_set = walks_with_edge_features.walk_set;
    const int feature_dim = walks_with_edge_features.feature_dim;
    const int*     nodes      = walk_set.nodes_ptr();
    const int64_t* timestamps = walk_set.timestamps_ptr();
    const size_t*  walk_lens  = walk_set.walk_lens_ptr();
    const float*   edge_feats = walks_with_edge_features.walk_edge_features.data();

    size_t count = 0;
    for (size_t walk_idx = 0; walk_idx < walk_set.num_walks() && count < n; ++walk_idx, ++count) {
        const size_t walk_len = walk_lens[walk_idx];
        const size_t node_base = walk_idx * walk_set.max_len();
        const size_t edge_base = walk_idx * (walk_set.max_len() - 1);

        std::cout << "Length: " << walk_len << ", Walk: ";
        for (size_t hop = 0; hop < walk_len; ++hop) {
            const size_t hop_offset = node_base + hop;
            std::cout << "(" << nodes[hop_offset] << ", " << timestamps[hop_offset] << "), ";
        }

        std::cout << "Weights: [";
        if (feature_dim > 0 && edge_feats != nullptr) {
            for (size_t hop = 1; hop < walk_len; ++hop) {
                const size_t edge_offset = edge_base + (hop - 1);
                std::cout << "[";
                for (int feat_idx = 0; feat_idx < feature_dim; ++feat_idx) {
                    const size_t feature_offset = edge_offset * static_cast<size_t>(feature_dim) +
                                                  static_cast<size_t>(feat_idx);
                    std::cout << edge_feats[feature_offset];
                    if (feat_idx + 1 < feature_dim) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]";
                if (hop + 1 < walk_len) {
                    std::cout << ", ";
                }
            }
        }
        std::cout << "]" << std::endl;
    }
}

inline double get_average_walk_length(const WalkSetHost& walk_set) {
    size_t total_walks = 0;
    size_t total_length = 0;

    for (auto it = walk_set.walks_begin(); it != walk_set.walks_end(); ++it) {
        const auto& walk = *it;
        total_length += walk.size();
        total_walks++;
    }

    return total_walks > 0 ?
        static_cast<double>(total_length) / static_cast<double>(total_walks) : 0.0;
}

// implemented in test_utils_parquet.cpp (CXX TU isolated from CUDA headers)
std::vector<std::tuple<int, int, int64_t>>
load_edges_from_parquet(const char* path);

inline std::tuple<std::vector<int>, std::vector<int>, std::vector<int64_t>>
convert_edge_tuples_to_components(const std::vector<std::tuple<int, int, int64_t>>& edges) {
    std::vector<int> sources;
    std::vector<int> targets;
    std::vector<int64_t> timestamps;

    sources.reserve(edges.size());
    targets.reserve(edges.size());
    timestamps.reserve(edges.size());

    for (const auto& edge : edges) {
        sources.push_back(std::get<0>(edge));
        targets.push_back(std::get<1>(edge));
        timestamps.push_back(std::get<2>(edge));
    }

    return std::make_tuple(sources, targets, timestamps);
}

inline std::vector<std::string> get_sorted_data_files(const std::string& base_path) {
    std::vector<std::string> file_paths;
    std::regex pattern("processed_data_(\\d+)\\.csv");

    try {
        for (const auto& entry : std::filesystem::directory_iterator(base_path)) {
            std::string filename = entry.path().filename().string();
            std::smatch matches;

            if (std::regex_match(filename, matches, pattern)) {
                file_paths.push_back(entry.path().string());
            }
        }

        std::sort(file_paths.begin(), file_paths.end(),
            [&pattern](const std::string& a, const std::string& b) {
                std::smatch matches_a, matches_b;
                std::regex_search(a, matches_a, pattern);
                std::regex_search(b, matches_b, pattern);

                const int index_a = std::stoi(matches_a[1].str());
                const int index_b = std::stoi(matches_b[1].str());

                return index_a < index_b;
            });

    } catch (const std::exception& e) {
        std::cerr << "Error finding files: " << e.what() << std::endl;
    }

    return file_paths;
}

// RFC 4180 quoting; empty path is a no-op
inline void write_strings_to_csv(const std::string& path,
                                 const std::vector<std::string>& header,
                                 const std::vector<std::vector<std::string>>& rows) {
    if (path.empty()) {
        return;
    }
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Failed to open CSV file for write: " << path << std::endl;
        return;
    }

    auto write_field = [&out](const std::string& v) {
        const bool needs_quote =
            v.find(',') != std::string::npos
            || v.find('"') != std::string::npos
            || v.find('\n') != std::string::npos
            || v.find('\r') != std::string::npos;
        if (!needs_quote) {
            out << v;
            return;
        }
        out << '"';
        for (char c : v) {
            if (c == '"') out << "\"\"";
            else out << c;
        }
        out << '"';
    };

    auto write_row = [&](const std::vector<std::string>& row) {
        for (size_t i = 0; i < row.size(); ++i) {
            write_field(row[i]);
            if (i + 1 < row.size()) out << ',';
        }
        out << '\n';
    };

    if (!header.empty()) write_row(header);
    for (const auto& r : rows) write_row(r);
    out.close();

    std::cout << "Wrote " << rows.size() << " row(s) to " << path << std::endl;
}

inline void dump_walks_to_file(const WalkSetHost& walk_set,
                               const int max_walk_len,
                               const std::string& output_file_path)
{
    if (output_file_path.empty()) {
        return;
    }

    std::ofstream out(output_file_path);
    if (!out.is_open()) {
        std::cerr << "Failed to open walk dump file: "
                  << output_file_path << std::endl;
        return;
    }

    const int*    nodes     = walk_set.nodes_ptr();
    const size_t* walk_lens = walk_set.walk_lens_ptr();

    for (size_t i = 0; i < walk_set.num_walks(); ++i) {
        const size_t walk_len = walk_lens[i];

        for (size_t j = 0; j < walk_len; ++j) {
            const int node = nodes[i * max_walk_len + j];
            out << node;
            if (j + 1 < walk_len) {
                out << " ";
            }
        }

        out << "\n";
    }

    out.close();

    std::cout << "Walks dumped to: " << output_file_path << std::endl;
}

#endif //TEST_RUN_UTILS_H
