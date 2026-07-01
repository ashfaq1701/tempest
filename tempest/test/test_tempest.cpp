#include <gtest/gtest.h>
#include <cmath>
#include <array>
#include <unordered_map>

#include "test_utils.h"
#include "../src/proxies/Tempest.cuh"

constexpr int TEST_NODE_ID = 42;
constexpr int MAX_WALK_LEN = 20;
constexpr int64_t MAX_TIME_CAPACITY = 5;

constexpr RandomPickerType exponential_picker_type = RandomPickerType::ExponentialIndex;
constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;

#ifdef HAS_CUDA
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>,
    std::integral_constant<bool, true>
>;
#else
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>   // CPU mode only
>;
#endif

template<typename T>
class EmptyTempestTest : public ::testing::Test {
protected:
    void SetUp() override {
        tempest = std::make_unique<Tempest>(true, T::value, -1, true, false, -1);
    }

    std::unique_ptr<Tempest> tempest;
};

TYPED_TEST_SUITE(EmptyTempestTest, GPU_USAGE_TYPES);

template<typename T>
class EmptyTempestTestWithMaxCapacity : public ::testing::Test {
protected:
    void SetUp() override {
        tempest = std::make_unique<Tempest>(true, T::value, MAX_TIME_CAPACITY, true, false, -1);
    }

    std::unique_ptr<Tempest> tempest;
};

TYPED_TEST_SUITE(EmptyTempestTestWithMaxCapacity, GPU_USAGE_TYPES);

template<typename T>
class FilledDirectedTempestTest : public ::testing::Test {
protected:
    FilledDirectedTempestTest() {
        sample_edges = read_edges_from_csv(sample_data_path());
    }

    void SetUp() override {
        tempest = std::make_unique<Tempest>(true, T::value, -1, true, false, -1);
        tempest->add_multiple_edges(sample_edges);
    }

    std::vector<std::tuple<int, int, int64_t>> sample_edges;
    std::unique_ptr<Tempest> tempest;
};

TYPED_TEST_SUITE(FilledDirectedTempestTest, GPU_USAGE_TYPES);

template<typename T>
class FilledUndirectedTempestTest : public ::testing::Test {
protected:
    FilledUndirectedTempestTest() {
        sample_edges = read_edges_from_csv(sample_data_path());
    }

    void SetUp() override {
        tempest = std::make_unique<Tempest>(false, T::value, -1, true, false, -1);
        tempest->add_multiple_edges(sample_edges);
    }

    std::vector<std::tuple<int, int, int64_t>> sample_edges;
    std::unique_ptr<Tempest> tempest;
};

TYPED_TEST_SUITE(FilledUndirectedTempestTest, GPU_USAGE_TYPES);

template<typename T>
class TimescaleBoundedTempestTest : public ::testing::Test {
protected:
    void SetUp() override {
        tempest = std::make_unique<Tempest>(true, T::value, -1, true, false, 10.0);
        tempest->add_multiple_edges({
            // Node 1's outgoing edges
            {1, 2, 100},
            {1, 3, 100}, // Same timestamp
            {1, 4, 101}, // Small difference
            {1, 5, 110}, // Larger difference

            // Node 2's outgoing edges
            {2, 3, 130},
            {2, 4, 130}, // Same timestamp
            {2, 5, 160}, // Larger difference

            // Node 3's outgoing edges
            {3, 4, 200},
            {3, 5, 200}, // Same timestamp
            {3, 6, 250}, // Larger difference
        });
    }

    std::unique_ptr<Tempest> tempest;
};

TYPED_TEST_SUITE(TimescaleBoundedTempestTest, GPU_USAGE_TYPES);

TYPED_TEST(EmptyTempestTest, ConstructorTest) {
    EXPECT_NO_THROW(this->tempest = std::make_unique<Tempest>(true, TypeParam::value));
    EXPECT_EQ(this->tempest->get_node_count(), 0);
}


TYPED_TEST(EmptyTempestTest, AddEdgeTest) {
    this->tempest->add_multiple_edges({
        {1, 2, 100},
        {2, 3, 101},
        {7, 8, 102},
        {1, 7, 103},
        {3, 2, 103},
        {10, 11, 104}
    });

    EXPECT_EQ(this->tempest->get_edge_count(), 6);
    EXPECT_EQ(this->tempest->get_node_count(), 7);
}

TYPED_TEST(EmptyTempestTestWithMaxCapacity, WhenMaxTimeCapacityExceedsEdgesAreDeletedAutomatically) {
    this->tempest->add_multiple_edges({
        { 0, 2, 1 },
        { 2, 3, 3 },
        { 1, 9, 2 },
        { 2, 4, 3 },
        { 2, 4, 1 },
        { 1, 5, 4 }
    });

    EXPECT_EQ(this->tempest->get_node_count(), 7);
    EXPECT_EQ(this->tempest->get_edge_count(), 6);

    this->tempest->add_multiple_edges({
        { 5, 6, 4 },
        { 2, 5, 4 },
        { 4, 3, 5 },
    });

    EXPECT_EQ(this->tempest->get_node_count(), 8);
    EXPECT_EQ(this->tempest->get_edge_count(), 9);

    this->tempest->add_multiple_edges({
        { 1, 7, 6 }
    });

    EXPECT_EQ(this->tempest->get_node_count(), 8);
    EXPECT_EQ(this->tempest->get_edge_count(), 8);

    this->tempest->add_multiple_edges({
        { 1, 5, 7 },
        { 4, 7, 8 }
    });

    EXPECT_EQ(this->tempest->get_node_count(), 7);
    EXPECT_EQ(this->tempest->get_edge_count(), 7);
}

TYPED_TEST(FilledDirectedTempestTest, TestNodeFoundTest) {
    const auto nodes = this->tempest->get_node_ids();
    const auto it = std::find(nodes.begin(), nodes.end(), TEST_NODE_ID);
    EXPECT_NE(it, nodes.end());
}

TYPED_TEST(FilledDirectedTempestTest, NodeDegreesTest) {
    // Ground truth straight from the edge list: directed out/in degree.
    std::unordered_map<int, int64_t> expected_out;
    std::unordered_map<int, int64_t> expected_in;
    for (const auto& [src, dst, ts] : this->sample_edges) {
        expected_out[src] += 1;
        expected_in[dst]  += 1;
    }

    const auto node_ids = this->tempest->get_node_ids();

    // Query every node, then a duplicate and an out-of-range id (must be 0).
    std::vector<int> query(node_ids.begin(), node_ids.end());
    ASSERT_FALSE(query.empty());
    query.push_back(query.front());   // duplicate is allowed
    query.push_back(999999);          // inactive / out-of-range -> 0

    const auto out_degrees = this->tempest->get_node_degrees(
        query.data(), query.size(), WalkDirection::Forward_In_Time);
    const auto in_degrees = this->tempest->get_node_degrees(
        query.data(), query.size(), WalkDirection::Backward_In_Time);

    ASSERT_EQ(out_degrees.size(), query.size());
    ASSERT_EQ(in_degrees.size(), query.size());

    for (size_t i = 0; i < query.size(); ++i) {
        const int node = query[i];
        const auto out_it = expected_out.find(node);
        const auto in_it  = expected_in.find(node);
        const int64_t want_out = (out_it == expected_out.end()) ? 0 : out_it->second;
        const int64_t want_in  = (in_it  == expected_in.end())  ? 0 : in_it->second;
        EXPECT_EQ(out_degrees[i], want_out) << "out-degree mismatch for node " << node;
        EXPECT_EQ(in_degrees[i], want_in) << "in-degree mismatch for node " << node;
    }

    EXPECT_EQ(out_degrees.back(), 0);  // 999999
    EXPECT_EQ(in_degrees.back(), 0);
}

TYPED_TEST(FilledUndirectedTempestTest, NodeDegreesTest) {
    // Undirected: each edge contributes to both endpoints; direction is ignored.
    std::unordered_map<int, int64_t> expected_total;
    for (const auto& [src, dst, ts] : this->sample_edges) {
        expected_total[src] += 1;
        expected_total[dst] += 1;
    }

    const auto node_ids = this->tempest->get_node_ids();
    std::vector<int> query(node_ids.begin(), node_ids.end());
    ASSERT_FALSE(query.empty());

    const auto forward_degrees = this->tempest->get_node_degrees(
        query.data(), query.size(), WalkDirection::Forward_In_Time);
    const auto backward_degrees = this->tempest->get_node_degrees(
        query.data(), query.size(), WalkDirection::Backward_In_Time);

    ASSERT_EQ(forward_degrees.size(), query.size());
    for (size_t i = 0; i < query.size(); ++i) {
        const int64_t want = expected_total[query[i]];
        EXPECT_EQ(forward_degrees[i], want) << "degree mismatch for node " << query[i];
        // undirected -> direction makes no difference
        EXPECT_EQ(backward_degrees[i], want) << "direction should not matter (undirected), node " << query[i];
    }
}

// length-1 walks are unreachable by the walk-state machine; strict `> 1`.
TYPED_TEST(FilledDirectedTempestTest, WalkCountAndLensTest) {
    const auto walk_set = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, 10);

    int total_walk_lens = 0;

    for (auto it = walk_set.walks_begin(); it != walk_set.walks_end(); ++it) {
        const auto& walk = *it;
        EXPECT_LE(walk.size(), MAX_WALK_LEN) << "A walk exceeds the maximum length of " << MAX_WALK_LEN;
        EXPECT_GT(walk.size(), 1) << "Length-1 walks are unreachable by the walk-state machine.";

        total_walk_lens += static_cast<int>(walk.size());
    }

    auto average_walk_len = static_cast<float>(total_walk_lens) / static_cast<float>(walk_set.size());
    EXPECT_GT(average_walk_len, 1) << "System could not sample any walk of length more than 1";
}

TYPED_TEST(FilledDirectedTempestTest, WalkIncreasingTimestampTest) {
    const auto walk_set_forward = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, 10);

    for (auto it = walk_set_forward.walks_begin(); it != walk_set_forward.walks_end(); ++it) {
        const auto& walk = *it;
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }

    const auto walk_set_backward = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, 10, nullptr, WalkDirection::Backward_In_Time);
    for (auto it = walk_set_backward.walks_begin(); it != walk_set_backward.walks_end(); ++it) {
        const auto& walk = *it;
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }
}

TYPED_TEST(FilledUndirectedTempestTest, WalkIncreasingTimestampTest) {
    const auto walk_set_forward = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, 10);

    for (auto it = walk_set_forward.walks_begin(); it != walk_set_forward.walks_end(); ++it) {
        const auto& walk = *it;
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }

    const auto walk_set_backward = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, 10, nullptr, WalkDirection::Backward_In_Time);
    for (auto it = walk_set_backward.walks_begin(); it != walk_set_backward.walks_end(); ++it) {
        const auto& walk = *it;
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }
}

TYPED_TEST(FilledDirectedTempestTest, WalkValidEdgesTest) {
    std::map<std::tuple<int, int, int64_t>, bool> valid_edges;
    for (const auto& edge : this->sample_edges) {
        valid_edges[edge] = true;
    }

    const auto walk_set_forward = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, 10, nullptr, WalkDirection::Forward_In_Time);

    for (auto it = walk_set_forward.walks_begin(); it != walk_set_forward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.size() <= 1) continue;

        for (size_t i = 0; i < walk.size() - 1; i++) {
            int src = walk[i].node;
            int dst = walk[i + 1].node;
            int64_t ts = walk[i + 1].timestamp;

            bool edge_exists = valid_edges.count({src, dst, ts}) > 0;
            EXPECT_TRUE(edge_exists)
                << "Invalid forward edge in walk: (" << src << "," << dst << "," << ts << ")";
        }
    }

    const auto walk_set_backward = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, 10, nullptr, WalkDirection::Backward_In_Time);

    for (auto it = walk_set_backward.walks_begin(); it != walk_set_backward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.size() <= 1) continue;

        for (size_t i = 1; i < walk.size(); i++) {
            int src = walk[i - 1].node;
            int dst = walk[i].node;
            int64_t ts = walk[i - 1].timestamp;

            bool edge_exists = valid_edges.count({src, dst, ts}) > 0;
            EXPECT_TRUE(edge_exists)
                << "Invalid backward edge in walk: (" << src << "," << dst << "," << ts << ")";
        }
    }
}

TYPED_TEST(FilledDirectedTempestTest, WalkTerminalEdgesTest) {
    std::map<int, int64_t> max_outgoing_timestamps;
    std::map<int, int64_t> min_incoming_timestamps;

    for (const auto& [src, dst, ts] : this->sample_edges) {
        if (!max_outgoing_timestamps.count(src) || max_outgoing_timestamps[src] < ts) {
            max_outgoing_timestamps[src] = ts;
        }
        if (!min_incoming_timestamps.count(dst) || min_incoming_timestamps[dst] > ts) {
            min_incoming_timestamps[dst] = ts;
        }
    }

    const auto walk_set_forward = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, 10, nullptr, WalkDirection::Forward_In_Time);

    for (auto it = walk_set_forward.walks_begin(); it != walk_set_forward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.empty()) continue;

        // walks at max length might have terminated early
        if (walk.size() == MAX_WALK_LEN) continue;

        int last_node = walk.back().node;
        const int64_t last_ts = walk.back().timestamp;

        if (!max_outgoing_timestamps.count(last_node)) continue;

        int64_t max_ts = max_outgoing_timestamps[last_node];
        if (last_ts < max_ts) {
            for (const auto& [src, dst, ts] : this->sample_edges) {
                if (src == last_node && ts > last_ts && ts <= max_ts) {
                    FAIL() << "Forward walk incorrectly terminated:\n"
                          << "  Node: " << last_node << "\n"
                          << "  Current timestamp: " << last_ts << "\n"
                          << "  Found valid edge at timestamp: " << ts << "\n"
                          << "  Max possible timestamp: " << max_ts << "\n"
                          << "  Edge: (" << src << "," << dst << "," << ts << ")";
                }
            }
        }
    }

    const auto walk_set_backward = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, 10, nullptr, WalkDirection::Backward_In_Time);

    for (auto it = walk_set_backward.walks_begin(); it != walk_set_backward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.empty()) continue;

        if (walk.size() == MAX_WALK_LEN) continue;

        int first_node = walk.front().node;
        const int64_t first_ts = walk.front().timestamp;

        if (!min_incoming_timestamps.count(first_node)) continue;

        int64_t min_ts = min_incoming_timestamps[first_node];
        if (first_ts > min_ts) {
            for (const auto& [src, dst, ts] : this->sample_edges) {
                if (dst == first_node && ts < first_ts && ts >= min_ts) {
                    FAIL() << "Backward walk incorrectly terminated:\n"
                          << "  Node: " << first_node << "\n"
                          << "  Current timestamp: " << first_ts << "\n"
                          << "  Found valid edge at timestamp: " << ts << "\n"
                          << "  Min possible timestamp: " << min_ts << "\n"
                          << "  Edge: (" << src << "," << dst << "," << ts << ")";
                }
            }
        }
    }
}

TYPED_TEST(FilledDirectedTempestTest, WalkIncreasingTimestampWithExponentialWeightTest) {
    constexpr RandomPickerType exponential_weight_picker = RandomPickerType::ExponentialWeight;
    const auto walk_set_forward = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker, 10);

    for (auto it = walk_set_forward.walks_begin(); it != walk_set_forward.walks_end(); ++it) {
        const auto& walk = *it;
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps not increasing at index " << i
                << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }

    const auto walk_set_backward = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker, 10, nullptr, WalkDirection::Backward_In_Time);

    for (auto it = walk_set_backward.walks_begin(); it != walk_set_backward.walks_end(); ++it) {
        const auto& walk = *it;
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps not increasing in backward walk at index " << i
                << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }
}

TYPED_TEST(FilledDirectedTempestTest, WalkValidEdgesWithExponentialWeightTest) {
    constexpr RandomPickerType exponential_weight_picker = RandomPickerType::ExponentialWeight;

    std::map<std::tuple<int, int, int64_t>, bool> valid_edges;
    for (const auto& edge : this->sample_edges) {
        valid_edges[edge] = true;
    }

    const auto walk_set_forward = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker, 10, nullptr, WalkDirection::Forward_In_Time);

    for (auto it = walk_set_forward.walks_begin(); it != walk_set_forward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.size() <= 1) continue;

        for (size_t i = 0; i < walk.size() - 1; i++) {
            int src = walk[i].node;
            int dst = walk[i+1].node;
            int64_t ts = walk[i+1].timestamp;

            bool edge_exists = valid_edges.count({src, dst, ts}) > 0;
            EXPECT_TRUE(edge_exists)
                << "Invalid forward edge in exponential weight walk: ("
                << src << "," << dst << "," << ts << ")";
        }
    }

    const auto walk_set_backward = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker, 10, nullptr, WalkDirection::Backward_In_Time);

    for (auto it = walk_set_backward.walks_begin(); it != walk_set_backward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.size() <= 1) continue;

        for (size_t i = 1; i < walk.size(); i++) {
            int src = walk[i - 1].node;
            int dst = walk[i].node;
            int64_t ts = walk[i - 1].timestamp;

            bool edge_exists = valid_edges.count({src, dst, ts}) > 0;
            EXPECT_TRUE(edge_exists)
                << "Invalid backward edge in exponential weight walk: ("
                << src << "," << dst << "," << ts << ")";
        }
    }
}

TYPED_TEST(FilledDirectedTempestTest, WalkTerminalEdgesWithExponentialWeightTest) {
    constexpr RandomPickerType exponential_weight_picker = RandomPickerType::ExponentialWeight;

    std::map<int, std::vector<int64_t>> next_valid_timestamps;
    for (const auto& [src, dst, ts] : this->tempest->get_edges()) {
        next_valid_timestamps[src].push_back(ts);
    }

    for (auto& [_, timestamps] : next_valid_timestamps) {
        std::sort(timestamps.begin(), timestamps.end());
    }

    const auto walk_set_forward = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker, 100);

    for (auto it = walk_set_forward.walks_begin(); it != walk_set_forward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.empty() || walk.size() == MAX_WALK_LEN) continue;

        const int last_node = walk.back().node;
        const int64_t last_ts = walk.back().timestamp;

        auto next_ts = next_valid_timestamps.find(last_node);
        if (next_ts == next_valid_timestamps.end()) continue;

        const auto& next_timestamps = next_ts->second;
        auto next_ts_it = std::upper_bound(next_timestamps.begin(), next_timestamps.end(), last_ts);

        EXPECT_EQ(next_ts_it, next_timestamps.end())
            << "Timescale bounded walk terminated despite having valid edges from node "
            << last_node << " after timestamp " << last_ts;
    }

    std::map<int, std::vector<int64_t>> prev_valid_timestamps;
    for (const auto& [src, dst, ts] : this->sample_edges) {
        prev_valid_timestamps[dst].push_back(ts);
    }

    const auto walk_set_backward = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker, 100, nullptr, WalkDirection::Backward_In_Time);

    for (auto it = walk_set_backward.walks_begin(); it != walk_set_backward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.empty() || walk.size() == MAX_WALK_LEN) continue;
        if (walk.back().timestamp == INT64_MAX) continue;

        const int first_node = walk.front().node;
        const int64_t first_ts = walk.front().timestamp;

        auto prev_ts = prev_valid_timestamps.find(first_node);
        if (prev_ts == prev_valid_timestamps.end()) continue;

        const auto& prev_timestamps = prev_ts->second;
        auto prev_ts_it = std::lower_bound(prev_timestamps.begin(), prev_timestamps.end(), first_ts);

        EXPECT_GT(prev_ts_it, prev_timestamps.begin())
            << "Backward walk terminated despite having valid edges to node "
            << first_node << " before timestamp " << first_ts;
    }
}

TYPED_TEST(TimescaleBoundedTempestTest, ValidEdgesWithScaling) {
    constexpr RandomPickerType exponential_weight_picker = RandomPickerType::ExponentialWeight;

    std::map<std::tuple<int, int, int64_t>, bool> valid_edges;
    for (const auto& edge : this->tempest->get_edges()) {
        valid_edges[edge] = true;
    }

    const auto walk_set = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker, 1000);

    for (auto it = walk_set.walks_begin(); it != walk_set.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.size() <= 1) continue;

        for (size_t i = 0; i < walk.size() - 1; i++) {
            const auto edge = std::make_tuple(
                walk[i].node,
                walk[i+1].node,
                walk[i+1].timestamp
            );
            EXPECT_TRUE(valid_edges[edge])
                << "Invalid edge in timescale bounded walk: ("
                << std::get<0>(edge) << ","
                << std::get<1>(edge) << ","
                << std::get<2>(edge) << ")";
        }
    }
}

template<typename T>
class LastBatchDirectedTempestTest : public ::testing::Test {
protected:
    void SetUp() override {
        tempest = std::make_unique<Tempest>(true, T::value, -1, true, false, -1);

        tempest->add_multiple_edges({
            {1, 2, 100},
            {2, 3, 101},
            {3, 4, 102},
        });

        // last-batch sources used as start nodes: {5, 7}
        tempest->add_multiple_edges({
            {5, 6, 103},
            {7, 8, 104},
            {5, 9, 105},
        });
    }

    std::unique_ptr<Tempest> tempest;
};

TYPED_TEST_SUITE(LastBatchDirectedTempestTest, GPU_USAGE_TYPES);

template<typename T>
class LastBatchUndirectedTempestTest : public ::testing::Test {
protected:
    void SetUp() override {
        tempest = std::make_unique<Tempest>(false, T::value, -1, true, false, -1);

        tempest->add_multiple_edges({
            {1, 2, 100},
            {2, 3, 101},
            {3, 4, 102},
        });

        // undirected: start nodes = sources∪targets = {5, 6, 7, 8, 9}
        tempest->add_multiple_edges({
            {5, 6, 103},
            {7, 8, 104},
            {5, 9, 105},
        });
    }

    std::unique_ptr<Tempest> tempest;
};

TYPED_TEST_SUITE(LastBatchUndirectedTempestTest, GPU_USAGE_TYPES);

TYPED_TEST(LastBatchDirectedTempestTest, WalksStartFromLastBatchSourcesOnly) {
    const std::set<int> expected_start_nodes = {5, 7};

    const auto walk_set = this->tempest->get_random_walks_and_times_for_last_batch(
        MAX_WALK_LEN, &linear_picker_type, 10);

    EXPECT_GT(walk_set.size(), 0) << "No walks were generated";

    for (auto it = walk_set.walks_begin(); it != walk_set.walks_end(); ++it) {
        const auto& walk = *it;
        ASSERT_GT(walk.size(), 0);
        int start_node = walk[0].node;
        EXPECT_TRUE(expected_start_nodes.count(start_node) > 0)
            << "Walk started from node " << start_node
            << " which is not a source in the last batch";
    }
}

TYPED_TEST(LastBatchUndirectedTempestTest, WalksStartFromLastBatchSourcesUnionTargets) {
    const std::set<int> expected_start_nodes = {5, 6, 7, 8, 9};

    const auto walk_set = this->tempest->get_random_walks_and_times_for_last_batch(
        MAX_WALK_LEN, &linear_picker_type, 10);

    EXPECT_GT(walk_set.size(), 0) << "No walks were generated";

    for (auto it = walk_set.walks_begin(); it != walk_set.walks_end(); ++it) {
        const auto& walk = *it;
        ASSERT_GT(walk.size(), 0);
        int start_node = walk[0].node;
        EXPECT_TRUE(expected_start_nodes.count(start_node) > 0)
            << "Walk started from node " << start_node
            << " which is not in the source/target union of the last batch";
    }
}

TYPED_TEST(TimescaleBoundedTempestTest, TerminalEdgeValidation) {
    constexpr RandomPickerType exponential_weight_picker = RandomPickerType::ExponentialWeight;

    std::map<int, std::vector<int64_t>> next_valid_timestamps;
    std::map<int, std::vector<int64_t>> prev_valid_timestamps;

    for (const auto& [src, dst, ts] : this->tempest->get_edges()) {
        next_valid_timestamps[src].push_back(ts);
        prev_valid_timestamps[dst].push_back(ts);
    }

    for (auto& [_, timestamps] : next_valid_timestamps) {
        std::sort(timestamps.begin(), timestamps.end());
    }
    for (auto& [_, timestamps] : prev_valid_timestamps) {
        std::sort(timestamps.begin(), timestamps.end());
    }

    const auto walk_set_forward = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker, 100, nullptr, WalkDirection::Forward_In_Time);

    for (auto it = walk_set_forward.walks_begin(); it != walk_set_forward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.empty() || walk.size() == MAX_WALK_LEN) continue;
        if (walk[0].timestamp == INT64_MIN) continue;

        const int last_node = walk.back().node;
        const int64_t last_ts = walk.back().timestamp;

        auto next_ts = next_valid_timestamps.find(last_node);
        if (next_ts == next_valid_timestamps.end()) continue;

        const auto& next_timestamps = next_ts->second;
        auto next_ts_it = std::upper_bound(next_timestamps.begin(), next_timestamps.end(), last_ts);

        EXPECT_EQ(next_ts_it, next_timestamps.end())
            << "Forward walk terminated despite having valid edges from node "
            << last_node << " after timestamp " << last_ts;
    }

    const auto walk_set_backward = this->tempest->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker, 100, nullptr, WalkDirection::Backward_In_Time);

    for (auto it = walk_set_backward.walks_begin(); it != walk_set_backward.walks_end(); ++it) {
        const auto& walk = *it;
        if (walk.empty() || walk.size() == MAX_WALK_LEN) continue;
        if (walk.back().timestamp == INT64_MAX) continue;

        const int first_node = walk.front().node;
        const int64_t first_ts = walk.front().timestamp;

        auto prev_ts = prev_valid_timestamps.find(first_node);
        if (prev_ts == prev_valid_timestamps.end()) continue;

        const auto& prev_timestamps = prev_ts->second;
        auto prev_ts_it = std::lower_bound(prev_timestamps.begin(), prev_timestamps.end(), first_ts);

        EXPECT_GT(prev_ts_it, prev_timestamps.begin())
            << "Backward walk terminated despite having valid edges to node "
            << first_node << " before timestamp " << first_ts;
    }
}
