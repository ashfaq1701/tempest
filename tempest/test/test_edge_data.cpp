#include <gtest/gtest.h>

#include "../src/core/tempest.cuh"
#include "../src/graph/edge_data.cuh"

template<typename T>
class EdgeDataTest : public ::testing::Test {
protected:
    core::Tempest trw;

    EdgeDataTest() : trw(/*is_directed=*/true, /*use_gpu=*/T::value) {}

    TemporalGraphData&       data()       { return trw.data(); }
    const TemporalGraphData& data() const { return trw.data(); }

    void push_back(const int src, const int tgt, const int64_t ts) {
        edge_data::push_back(data(), src, tgt, ts);
    }

    void update_timestamp_groups() {
#ifdef HAS_CUDA
        if (data().use_gpu) {
            edge_data::update_timestamp_groups_cuda(data());
            return;
        }
#endif
        edge_data::update_timestamp_groups_std(data());
    }

    void verify_edge(const size_t index,
                     const int expected_src, const int expected_tgt,
                     const int64_t expected_ts) const {
        const auto snap = edge_data::snapshot(data());
        ASSERT_LT(index, snap.sources.size());
        EXPECT_EQ(snap.sources[index], expected_src);
        EXPECT_EQ(snap.targets[index], expected_tgt);
        EXPECT_EQ(snap.timestamps[index], expected_ts);
    }
};

#ifdef HAS_CUDA
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>,
    std::integral_constant<bool, true>
>;
#else
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<bool, false>
>;
#endif

TYPED_TEST_SUITE(EdgeDataTest, GPU_USAGE_TYPES);

TYPED_TEST(EdgeDataTest, EmptyStateTest) {
    EXPECT_TRUE(edge_data::empty(this->data()));
    EXPECT_EQ(edge_data::size(this->data()), 0u);

    const auto snap = edge_data::snapshot(this->data());
    EXPECT_TRUE(snap.timestamp_group_offsets.empty());
    EXPECT_TRUE(snap.unique_timestamps.empty());
    EXPECT_EQ(this->data().max_node_id, -1);
}

TYPED_TEST(EdgeDataTest, SingleEdgeTest) {
    this->push_back(100, 200, 100);
    EXPECT_FALSE(edge_data::empty(this->data()));
    EXPECT_EQ(edge_data::size(this->data()), 1u);
    this->verify_edge(0, 100, 200, 100);

    this->update_timestamp_groups();

    const auto snap = edge_data::snapshot(this->data());
    EXPECT_EQ(snap.unique_timestamps.size(), 1u);
    EXPECT_EQ(snap.timestamp_group_offsets.size(), 2u);
    EXPECT_EQ(snap.timestamp_group_offsets[0], 0u);
    EXPECT_EQ(snap.timestamp_group_offsets[1], 1u);
    EXPECT_EQ(this->data().max_node_id, 200);
}

TYPED_TEST(EdgeDataTest, SameTimestampEdgesTest) {
    this->push_back(100, 200, 100);
    this->push_back(200, 300, 100);
    this->push_back(300, 400, 100);

    this->update_timestamp_groups();

    const auto snap = edge_data::snapshot(this->data());
    EXPECT_EQ(snap.unique_timestamps.size(), 1u);
    EXPECT_EQ(snap.timestamp_group_offsets.size(), 2u);
    EXPECT_EQ(snap.timestamp_group_offsets[0], 0u);
    EXPECT_EQ(snap.timestamp_group_offsets[1], 3u);
}

TYPED_TEST(EdgeDataTest, DifferentTimestampEdgesTest) {
    this->push_back(1, 2, 100);
    this->push_back(2, 3, 200);
    this->push_back(3, 4, 300);

    this->update_timestamp_groups();

    const auto snap = edge_data::snapshot(this->data());
    EXPECT_EQ(snap.unique_timestamps.size(), 3u);
    EXPECT_EQ(snap.timestamp_group_offsets.size(), 4u);
    EXPECT_EQ(snap.timestamp_group_offsets[0], 0u);
    EXPECT_EQ(snap.timestamp_group_offsets[1], 1u);
    EXPECT_EQ(snap.timestamp_group_offsets[2], 2u);
    EXPECT_EQ(snap.timestamp_group_offsets[3], 3u);
    EXPECT_EQ(this->data().max_node_id, 4);
}

TYPED_TEST(EdgeDataTest, FindGroupTest) {
    this->push_back(100, 200, 100);
    this->push_back(200, 300, 200);
    this->push_back(300, 400, 300);
    this->update_timestamp_groups();

    const auto unique_ts = this->data().unique_timestamps.to_host_vector();

    auto after = [&](const int64_t t) -> size_t {
        return std::upper_bound(unique_ts.begin(), unique_ts.end(), t) - unique_ts.begin();
    };
    auto before = [&](const int64_t t) -> long {
        const auto it = std::lower_bound(unique_ts.begin(), unique_ts.end(), t);
        return (it - unique_ts.begin()) - 1;
    };

    EXPECT_EQ(after(50),  0u);
    EXPECT_EQ(after(100), 1u);
    EXPECT_EQ(after(150), 1u);
    EXPECT_EQ(after(200), 2u);
    EXPECT_EQ(after(300), 3u);
    EXPECT_EQ(after(350), 3u);

    EXPECT_EQ(before(50),  -1);
    EXPECT_EQ(before(150), 0);
    EXPECT_EQ(before(200), 0);
    EXPECT_EQ(before(300), 1);
    EXPECT_EQ(before(350), 2);
}

TYPED_TEST(EdgeDataTest, TimestampGroupRangeTest) {
    this->push_back(100, 200, 100);
    this->push_back(200, 300, 100);
    this->push_back(300, 400, 200);
    this->push_back(400, 500, 300);
    this->push_back(500, 600, 300);
    this->update_timestamp_groups();

    const auto offsets = this->data().timestamp_group_offsets.to_host_vector();
    const size_t num_groups = this->data().unique_timestamps.size();

    auto range = [&](const size_t g) -> std::pair<size_t, size_t> {
        if (g >= num_groups) return {0, 0};
        return {offsets[g], offsets[g + 1]};
    };

    const auto [start0, end0] = range(0);
    EXPECT_EQ(start0, 0u);
    EXPECT_EQ(end0, 2u);

    const auto [start1, end1] = range(1);
    EXPECT_EQ(start1, 2u);
    EXPECT_EQ(end1, 3u);

    const auto [start2, end2] = range(2);
    EXPECT_EQ(start2, 3u);
    EXPECT_EQ(end2, 5u);

    const auto [invalid_start, invalid_end] = range(3);
    EXPECT_EQ(invalid_start, 0u);
    EXPECT_EQ(invalid_end, 0u);
}

TYPED_TEST(EdgeDataTest, MaxNodeIdResetsWhenGroupsCleared) {
    this->push_back(2, 7, 10);
    this->update_timestamp_groups();
    EXPECT_EQ(this->data().max_node_id, 7);

    edge_data::set_size(this->data(), 0);
    this->update_timestamp_groups();

    EXPECT_EQ(this->data().max_node_id, -1);
}
