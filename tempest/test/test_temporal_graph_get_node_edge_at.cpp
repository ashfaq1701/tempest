#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <vector>

#include "../src/core/tempest.cuh"
#include "test_temporal_graph_utils.h"

template<typename T>
class TemporalGraphGetNodeEdgeAtTest : public ::testing::Test {
protected:
    std::unique_ptr<core::Tempest> graph;

    void SetUp() override {
        graph = std::make_unique<core::Tempest>(/*is_directed=*/true, T::value);
    }

    TemporalGraphData&       data()       { return graph->data(); }
    const TemporalGraphData& data() const { return graph->data(); }

    static void verify_edge(const Edge& edge,
                            const int expected_src, const int expected_tgt,
                            const int64_t expected_ts) {
        EXPECT_EQ(edge.u, expected_src);
        EXPECT_EQ(edge.i, expected_tgt);
        EXPECT_EQ(edge.ts, expected_ts);
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

TYPED_TEST_SUITE(TemporalGraphGetNodeEdgeAtTest, GPU_USAGE_TYPES);

TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, ForwardWalkTest) {
    test_util::add_edges(*this->graph, {
        Edge{10, 20, 100},
        Edge{10, 30, 100},
        Edge{10, 40, 102},
        Edge{10, 50, 104},
        Edge{20, 10, 101},
        Edge{30, 10, 103}
    });

    const auto& d = this->data();

    auto edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, -1, -1, true);
    EXPECT_EQ(edge.ts, 100);
    EXPECT_EQ(edge.u, 10);

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_LAST, -1, -1, true);
    EXPECT_EQ(edge.ts, 104);
    EXPECT_EQ(edge.u, 10);

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, 102, -1, true);
    EXPECT_EQ(edge.ts, 104);

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, 103, -1, true);
    EXPECT_EQ(edge.ts, 104);

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, 104, -1, true);
    this->verify_edge(edge, -1, -1, -1);
}

TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, BackwardWalkTest) {
    test_util::add_edges(*this->graph, {
        Edge{20, 10, 100},
        Edge{30, 10, 101},
        Edge{40, 10, 102},
        Edge{50, 10, 103},
        Edge{10, 20, 101},
        Edge{10, 30, 102}
    });

    const auto& d = this->data();

    auto edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, -1, -1, false);
    this->verify_edge(edge, 20, 10, 100);

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_LAST, -1, -1, false);
    this->verify_edge(edge, 50, 10, 103);

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, 104, -1, false);
    this->verify_edge(edge, 20, 10, 100);

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, 103, -1, false);
    this->verify_edge(edge, 20, 10, 100);

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, 102, -1, false);
    this->verify_edge(edge, 20, 10, 100);

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, 101, -1, false);
    this->verify_edge(edge, 20, 10, 100);

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_LAST, 104, -1, false);
    this->verify_edge(edge, 50, 10, 103);

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_LAST, 103, -1, false);
    this->verify_edge(edge, 40, 10, 102);

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_LAST, 102, -1, false);
    this->verify_edge(edge, 30, 10, 101);

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, 101, -1, false);
    this->verify_edge(edge, 20, 10, 100);

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, 100, -1, false);
    this->verify_edge(edge, -1, -1, -1);

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, 99, -1, false);
    this->verify_edge(edge, -1, -1, -1);
}

TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, EdgeCasesTest) {
    test_util::add_edges(*this->graph, {
        Edge{10, 20, 100},
        Edge{10, 30, 101}
    });

    const auto& d = this->data();

    auto edge = test_util::get_node_edge_at(d, -1, RandomPickerType::TEST_FIRST, -1, -1, true);
    this->verify_edge(edge, -1, -1, -1);

    edge = test_util::get_node_edge_at(d, 999, RandomPickerType::TEST_FIRST, -1, -1, true);
    this->verify_edge(edge, -1, -1, -1);

    edge = test_util::get_node_edge_at(d, 20, RandomPickerType::TEST_FIRST, -1, -1, true);
    this->verify_edge(edge, -1, -1, -1);

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, -1, -1, false);
    this->verify_edge(edge, -1, -1, -1);

    this->graph = std::make_unique<core::Tempest>(true, TypeParam::value);
    edge = test_util::get_node_edge_at(this->data(), 10, RandomPickerType::TEST_FIRST, -1, -1, true);
    this->verify_edge(edge, -1, -1, -1);
}

TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, RandomSelectionTest) {
    test_util::add_edges(*this->graph, {
        Edge{10, 20, 100},
        Edge{10, 30, 100},
        Edge{10, 40, 100},
        Edge{10, 50, 101}
    });

    const auto& d = this->data();

    std::set<int> seen_targets;
    constexpr int NUM_TRIES = 50;

    for (int i = 0; i < NUM_TRIES; i++) {
        auto edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, -1, -1, true);
        EXPECT_EQ(edge.ts, 100);
        seen_targets.insert(edge.i);
    }
    EXPECT_GT(seen_targets.size(), 1u);
}

TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, ExactTimestampTest) {
    test_util::add_edges(*this->graph, {
        Edge{20, 10, 100},
        Edge{30, 10, 101},
        Edge{40, 10, 102},
        Edge{10, 50, 100},
        Edge{10, 60, 101},
        Edge{10, 70, 102}
    });

    const auto& d = this->data();

    auto edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, 100, -1, true);
    this->verify_edge(edge, 10, 60, 101);

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, 101, -1, false);
    this->verify_edge(edge, 20, 10, 100);
}

TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, ExactTimestampUndirectedTest) {
    this->graph = std::make_unique<core::Tempest>(
        /*is_directed=*/false, TypeParam::value);

    test_util::add_edges(*this->graph, {
        Edge{10, 20, 100},
        Edge{30, 10, 101},
        Edge{10, 40, 102},
        Edge{50, 10, 100},
        Edge{10, 60, 101},
        Edge{70, 10, 102},
        Edge{20, 30, 104}
    });

    const auto& d = this->data();

    auto edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, 100, -1, true);
    EXPECT_EQ(edge.ts, 101);
    EXPECT_TRUE((edge.u == 30 && edge.i == 10) || (edge.u == 10 && edge.i == 60));

    edge = test_util::get_node_edge_at(d, 10, RandomPickerType::TEST_FIRST, 101, -1, false);
    EXPECT_EQ(edge.ts, 100);
    EXPECT_TRUE((edge.u == 10 && edge.i == 20) || (edge.u == 50 && edge.i == 10));

    edge = test_util::get_node_edge_at(d, 20, RandomPickerType::TEST_FIRST, 100, -1, true);
    this->verify_edge(edge, 20, 30, 104);
}
