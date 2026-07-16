#include <gtest/gtest.h>

#include <memory>
#include <tuple>
#include <vector>

#include "../src/common/const.cuh"
#include "../src/core/tempest.cuh"
#include "../src/graph/edge_data.cuh"
#include "../src/graph/node_edge_index.cuh"

namespace {

static void do_update_timestamp_groups(TemporalGraphData& data) {
#ifdef HAS_CUDA
    if (data.use_gpu) {
        edge_data::update_timestamp_groups_cuda(data);
        return;
    }
#endif
    edge_data::update_timestamp_groups_std(data);
}

} // namespace

template<typename T>
class NodeEdgeIndexTest : public ::testing::Test {
protected:
    core::Tempest trw;

    NodeEdgeIndexTest() : trw(/*is_directed=*/true, /*use_gpu=*/T::value) {}

    TemporalGraphData&       data()       { return trw.data(); }
    const TemporalGraphData& data() const { return trw.data(); }

    core::Tempest make_simple_directed_graph() {
        core::Tempest local(/*is_directed=*/true, T::value);
        auto& d = local.data();
        edge_data::push_back(d, 10, 20, 100);
        edge_data::push_back(d, 10, 30, 100);
        edge_data::push_back(d, 10, 20, 200);
        edge_data::push_back(d, 20, 30, 300);
        edge_data::push_back(d, 20, 10, 300);
        do_update_timestamp_groups(d);
        node_edge_index::rebuild(d);
        return local;
    }

    core::Tempest make_simple_undirected_graph() {
        core::Tempest local(/*is_directed=*/false, T::value);
        auto& d = local.data();
        edge_data::push_back(d, 100, 200, 1000);
        edge_data::push_back(d, 100, 300, 1000);
        edge_data::push_back(d, 100, 200, 2000);
        edge_data::push_back(d, 200, 300, 3000);
        do_update_timestamp_groups(d);
        node_edge_index::rebuild(d);
        return local;
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

TYPED_TEST_SUITE(NodeEdgeIndexTest, GPU_USAGE_TYPES);

TYPED_TEST(NodeEdgeIndexTest, EmptyStateTest) {
    const auto idx = node_edge_index::snapshot(this->data());
    EXPECT_TRUE(idx.node_group_outbound_offsets.empty());
    EXPECT_TRUE(idx.node_ts_sorted_outbound_indices.empty());
    EXPECT_TRUE(idx.count_ts_group_per_node_outbound.empty());
    EXPECT_TRUE(idx.node_ts_group_outbound_offsets.empty());
    EXPECT_TRUE(idx.node_group_inbound_offsets.empty());
    EXPECT_TRUE(idx.node_ts_sorted_inbound_indices.empty());
    EXPECT_TRUE(idx.count_ts_group_per_node_inbound.empty());
    EXPECT_TRUE(idx.node_ts_group_inbound_offsets.empty());
}

TYPED_TEST(NodeEdgeIndexTest, DirectedEdgeRangeTest) {
    auto local = this->make_simple_directed_graph();
    const auto& data = local.data();
    const auto edges = edge_data::snapshot(data);
    const auto idx   = node_edge_index::snapshot(data);

    const auto out_start10 = idx.node_group_outbound_offsets[10];
    const auto out_end10   = idx.node_group_outbound_offsets[11];
    EXPECT_EQ(out_end10 - out_start10, 3u);
    for (size_t i = out_start10; i < out_end10; i++) {
        const size_t edge_idx = idx.node_ts_sorted_outbound_indices[i];
        EXPECT_EQ(edges.sources[edge_idx], 10);
    }

    const auto in_start20 = idx.node_group_inbound_offsets[20];
    const auto in_end20   = idx.node_group_inbound_offsets[21];
    EXPECT_EQ(in_end20 - in_start20, 2u);
    for (size_t i = in_start20; i < in_end20; i++) {
        const size_t edge_idx = idx.node_ts_sorted_inbound_indices[i];
        EXPECT_EQ(edges.targets[edge_idx], 20);
    }
}

TYPED_TEST(NodeEdgeIndexTest, DirectedTimestampGroupTest) {
    auto local = this->make_simple_directed_graph();
    const auto& data = local.data();
    const auto edges = edge_data::snapshot(data);
    const auto idx   = node_edge_index::snapshot(data);

    constexpr int node_id = 10;
    const auto grp_start = idx.count_ts_group_per_node_outbound[node_id];
    const auto grp_end   = idx.count_ts_group_per_node_outbound[node_id + 1];
    EXPECT_EQ(grp_end - grp_start, 2u);

    const auto g0_edge_start = idx.node_ts_group_outbound_offsets[grp_start];
    const auto g0_edge_end =
        (grp_start + 1 < grp_end)
            ? idx.node_ts_group_outbound_offsets[grp_start + 1]
            : idx.node_group_outbound_offsets[node_id + 1];
    EXPECT_EQ(g0_edge_end - g0_edge_start, 2u);
    for (size_t i = g0_edge_start; i < g0_edge_end; i++) {
        const size_t edge_idx = idx.node_ts_sorted_outbound_indices[i];
        EXPECT_EQ(edges.timestamps[edge_idx], 100);
        EXPECT_EQ(edges.sources[edge_idx], 10);
        EXPECT_TRUE(edges.targets[edge_idx] == 20 || edges.targets[edge_idx] == 30);
    }

    const auto g1_edge_start = idx.node_ts_group_outbound_offsets[grp_start + 1];
    const auto g1_edge_end   = idx.node_group_outbound_offsets[node_id + 1];
    EXPECT_EQ(g1_edge_end - g1_edge_start, 1u);
    const size_t edge_idx = idx.node_ts_sorted_outbound_indices[g1_edge_start];
    EXPECT_EQ(edges.timestamps[edge_idx], 200);
    EXPECT_EQ(edges.sources[edge_idx], 10);
    EXPECT_EQ(edges.targets[edge_idx], 20);
}

TYPED_TEST(NodeEdgeIndexTest, UndirectedEdgeRangeTest) {
    auto local = this->make_simple_undirected_graph();
    const auto& data = local.data();
    const auto edges = edge_data::snapshot(data);
    const auto idx   = node_edge_index::snapshot(data);

    // undirected graph stores all edges as outbound
    const auto s100 = idx.node_group_outbound_offsets[100];
    const auto e100 = idx.node_group_outbound_offsets[101];
    EXPECT_EQ(e100 - s100, 3u);
    for (size_t i = s100; i < e100; i++) {
        const size_t edge_idx = idx.node_ts_sorted_outbound_indices[i];
        EXPECT_TRUE(
            (edges.sources[edge_idx] == 100 && (edges.targets[edge_idx] == 200 || edges.targets[edge_idx] == 300)) ||
            (edges.targets[edge_idx] == 100 && (edges.sources[edge_idx] == 200 || edges.sources[edge_idx] == 300))
        );
    }

    const auto s200 = idx.node_group_outbound_offsets[200];
    const auto e200 = idx.node_group_outbound_offsets[201];
    EXPECT_EQ(e200 - s200, 3u);
}

TYPED_TEST(NodeEdgeIndexTest, UndirectedTimestampGroupTest) {
    auto local = this->make_simple_undirected_graph();
    const auto& data = local.data();
    const auto edges = edge_data::snapshot(data);
    const auto idx   = node_edge_index::snapshot(data);

    constexpr int node_id = 100;
    const auto grp_start = idx.count_ts_group_per_node_outbound[node_id];
    const auto grp_end   = idx.count_ts_group_per_node_outbound[node_id + 1];
    EXPECT_EQ(grp_end - grp_start, 2u);

    const auto g0_edge_start = idx.node_ts_group_outbound_offsets[grp_start];
    const auto g0_edge_end =
        (grp_start + 1 < grp_end)
            ? idx.node_ts_group_outbound_offsets[grp_start + 1]
            : idx.node_group_outbound_offsets[node_id + 1];
    EXPECT_EQ(g0_edge_end - g0_edge_start, 2u);
    EXPECT_EQ(edges.timestamps[idx.node_ts_sorted_outbound_indices[g0_edge_start]], 1000);
    for (size_t i = g0_edge_start; i < g0_edge_end; i++) {
        const size_t edge_idx = idx.node_ts_sorted_outbound_indices[i];
        EXPECT_EQ(edges.timestamps[edge_idx], 1000);
    }

    const auto g1_edge_start = idx.node_ts_group_outbound_offsets[grp_start + 1];
    const auto g1_edge_end   = idx.node_group_outbound_offsets[node_id + 1];
    EXPECT_EQ(g1_edge_end - g1_edge_start, 1u);
    const size_t edge_idx = idx.node_ts_sorted_outbound_indices[g1_edge_start];
    EXPECT_EQ(edges.timestamps[edge_idx], 2000);
    EXPECT_TRUE(
        (edges.sources[edge_idx] == 100 && edges.targets[edge_idx] == 200) ||
        (edges.targets[edge_idx] == 100 && edges.sources[edge_idx] == 200)
    );
}

TYPED_TEST(NodeEdgeIndexTest, EdgeCasesTest) {
    auto local = this->make_simple_directed_graph();
    auto& data = local.data();

    EXPECT_EQ(node_edge_index::get_timestamp_group_count(data, -1, /*forward=*/true), 0u);

    const SizeRange r =
        node_edge_index::get_timestamp_group_range(data, 1, /*group_idx=*/999, true);
    EXPECT_EQ(r.from, 0u);
    EXPECT_EQ(r.to,   0u);

    edge_data::push_back(data, 4, 5, 400);
    do_update_timestamp_groups(data);
    node_edge_index::rebuild(data);

    EXPECT_EQ(node_edge_index::get_timestamp_group_count(data, 4, /*forward=*/false), 0u);
}

// Directed graph: 10->20@100, 10->30@100, 10->20@200, 20->30@300, 20->10@300.
TYPED_TEST(NodeEdgeIndexTest, LatestEventsForNodes) {
    auto local = this->make_simple_directed_graph();
    const std::vector<int> nodes = {10, 20, 30, 999};   // 30 = no outbound, 999 = inactive

    // No cutoff -> each node's latest OUTBOUND event: (partner, timestamp).
    std::vector<int64_t> partners, latest;
    std::tie(partners, latest) = local.get_latest_events_for_nodes(
        nodes.data(), nodes.size(), /*cutoff_times=*/nullptr, WalkDirection::Forward_In_Time);
    ASSERT_EQ(latest.size(), nodes.size());
    ASSERT_EQ(partners.size(), nodes.size());
    EXPECT_EQ(latest[0], 200);                                 // node 10: latest 10->20@200
    EXPECT_EQ(partners[0], 20);                                //   partner 20 (unambiguous, single edge @200)
    EXPECT_EQ(latest[1], 300);                                 // node 20: latest @300
    EXPECT_TRUE(partners[1] == 30 || partners[1] == 10);       //   tie (20->30, 20->10 both @300)
    EXPECT_EQ(latest[2], -1);                                  // node 30: no outbound
    EXPECT_EQ(partners[2], -1);
    EXPECT_EQ(latest[3], -1);                                  // node 999: inactive
    EXPECT_EQ(partners[3], -1);

    // Per-node EXCLUSIVE cutoffs.
    const std::vector<int64_t> cutoffs = {200, 300, 1000, 1000};
    std::tie(partners, latest) = local.get_latest_events_for_nodes(
        nodes.data(), nodes.size(), cutoffs.data(), WalkDirection::Forward_In_Time);
    EXPECT_EQ(latest[0], 100);                                 // node 10, t < 200 -> 100
    EXPECT_TRUE(partners[0] == 20 || partners[0] == 30);       //   tie (10->20, 10->30 both @100)
    EXPECT_EQ(latest[1], -1);                                  // node 20, t < 300 -> none
    EXPECT_EQ(partners[1], -1);

    // Cutoff at/below the earliest edge -> none.
    const std::vector<int64_t> tight = {100, 100, 100, 100};
    std::tie(partners, latest) = local.get_latest_events_for_nodes(
        nodes.data(), nodes.size(), tight.data(), WalkDirection::Forward_In_Time);
    EXPECT_EQ(latest[0], -1);                                  // node 10, t < 100 -> none
    EXPECT_EQ(partners[0], -1);

    // Backward uses INBOUND edges: node 20 inbound = 10->20@{100,200} -> latest 200, partner 10.
    std::vector<int64_t> partners_in, latest_in;
    std::tie(partners_in, latest_in) = local.get_latest_events_for_nodes(
        nodes.data(), nodes.size(), nullptr, WalkDirection::Backward_In_Time);
    EXPECT_EQ(latest_in[1], 200);
    EXPECT_EQ(partners_in[1], 10);                             //   unambiguous: 10->20@200
}

TYPED_TEST(NodeEdgeIndexTest, NodeParticipationCounts) {
    auto local = this->make_simple_directed_graph();
    const std::vector<int> nodes = {10, 20, 30, 999};

    // No cutoff -> outbound degree.
    auto counts = local.get_node_participation_counts(
        nodes.data(), nodes.size(), /*cutoff_times=*/nullptr, WalkDirection::Forward_In_Time);
    ASSERT_EQ(counts.size(), nodes.size());
    EXPECT_EQ(counts[0], 3);   // node 10: 3 outbound edges
    EXPECT_EQ(counts[1], 2);   // node 20: 2 outbound edges (both @300)
    EXPECT_EQ(counts[2], 0);   // node 30: no outbound
    EXPECT_EQ(counts[3], 0);   // inactive

    const std::vector<int64_t> cutoffs = {200, 300, 1000, 1000};
    counts = local.get_node_participation_counts(
        nodes.data(), nodes.size(), cutoffs.data(), WalkDirection::Forward_In_Time);
    EXPECT_EQ(counts[0], 2);   // node 10, t < 200 -> two @100
    EXPECT_EQ(counts[1], 0);   // node 20, t < 300 -> none

    const std::vector<int64_t> tight = {100, 100, 100, 100};
    counts = local.get_node_participation_counts(
        nodes.data(), nodes.size(), tight.data(), WalkDirection::Forward_In_Time);
    EXPECT_EQ(counts[0], 0);   // node 10, t < 100 -> none

    // Backward -> inbound degree: node 20 inbound = 10->20@{100,200} -> 2.
    const auto counts_in = local.get_node_participation_counts(
        nodes.data(), nodes.size(), nullptr, WalkDirection::Backward_In_Time);
    EXPECT_EQ(counts_in[1], 2);
}
