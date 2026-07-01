#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <set>
#include <vector>

#include "../src/common/const.cuh"
#include "../src/core/tempest.cuh"
#include "../src/graph/edge_data.cuh"
#include "test_temporal_graph_utils.h"

template<typename T>
class TemporalGraphTest : public ::testing::Test {
protected:
    std::unique_ptr<core::Tempest> graph;

    void SetUp() override {
        graph = std::make_unique<core::Tempest>(/*is_directed=*/true, T::value);
    }

    TemporalGraphData&       data()       { return graph->data(); }
    const TemporalGraphData& data() const { return graph->data(); }
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

TYPED_TEST_SUITE(TemporalGraphTest, GPU_USAGE_TYPES);

TYPED_TEST(TemporalGraphTest, EmptyStateTest) {
    EXPECT_EQ(this->graph->get_node_ids().size(), 0u);
    EXPECT_TRUE(this->graph->get_edges().empty());
}

TYPED_TEST(TemporalGraphTest, BasicEdgeAdditionTest) {
    std::vector<Edge> edges = {
        Edge {1, 2, 100},
        Edge {2, 3, 200},
        Edge {3, 1, 300}
    };
    test_util::add_edges(*this->graph, edges);

    EXPECT_EQ(this->graph->get_edges().size(), 3u);
    EXPECT_EQ(this->graph->get_node_ids().size(), 3u);
}

TYPED_TEST(TemporalGraphTest, MaintainSortedOrderTest) {
    test_util::add_edges(*this->graph, {
        Edge {10, 20, 200},
        Edge {20, 30, 100}
    });

    auto sorted_edges = this->graph->get_edges();
    EXPECT_EQ(sorted_edges[0].ts, 100);
    EXPECT_EQ(sorted_edges[1].ts, 200);

    test_util::add_edges(*this->graph, {
        Edge {30, 40, 150},
        Edge {40, 50, 250}
    });

    sorted_edges = this->graph->get_edges();
    EXPECT_EQ(sorted_edges.size(), 4u);
    EXPECT_EQ(sorted_edges[0].ts, 100);
    EXPECT_EQ(sorted_edges[1].ts, 150);
    EXPECT_EQ(sorted_edges[2].ts, 200);
    EXPECT_EQ(sorted_edges[3].ts, 250);

    test_util::add_edges(*this->graph, {
        Edge {50, 60, 150},
        Edge {60, 70, 200},
        Edge {70, 80, 175}
    });

    sorted_edges = this->graph->get_edges();
    EXPECT_EQ(sorted_edges.size(), 7u);
    EXPECT_EQ(sorted_edges[0].ts, 100);
    EXPECT_EQ(sorted_edges[1].ts, 150);
    EXPECT_EQ(sorted_edges[2].ts, 150);
    EXPECT_EQ(sorted_edges[3].ts, 175);
    EXPECT_EQ(sorted_edges[4].ts, 200);
    EXPECT_EQ(sorted_edges[5].ts, 200);
    EXPECT_EQ(sorted_edges[6].ts, 250);

    EXPECT_EQ(test_util::count_node_timestamps_greater_than(this->data(), 30, 50),  1u);
    EXPECT_EQ(test_util::count_node_timestamps_less_than(this->data(), 30, 200),    1u);
}

TYPED_TEST(TemporalGraphTest, TimeWindowTest) {
    this->graph = std::make_unique<core::Tempest>(
        /*is_directed=*/true, TypeParam::value, /*max_time_capacity=*/100);

    test_util::add_edges(*this->graph, {
        Edge {1, 2, 100},
        Edge {2, 3, 150},
        Edge {3, 4, 249}
    });
    const auto remaining_edges = this->graph->get_edges();

    EXPECT_EQ(remaining_edges.size(), 2u);
    EXPECT_EQ(remaining_edges[0].ts, 150);
    EXPECT_EQ(remaining_edges[1].ts, 249);
}

TYPED_TEST(TemporalGraphTest, EdgeAdditionEdgeCasesTest) {
    test_util::add_edges(*this->graph, {});
    EXPECT_TRUE(this->graph->get_edges().empty());

    test_util::add_edges(*this->graph, {Edge {1, 2, 100}});
    EXPECT_EQ(this->graph->get_edges().size(), 1u);

    test_util::add_edges(*this->graph, {
        Edge {1, 2, 100},
        Edge {2, 3, 100},
        Edge {3, 4, 100}
    });
    EXPECT_EQ(this->graph->get_edges().size(), 4u);

    test_util::add_edges(*this->graph, {Edge {1, 2, INT64_MAX}});
    EXPECT_EQ(this->graph->get_edges().size(), 5u);
}

TYPED_TEST(TemporalGraphTest, NodeDeletionTest) {
    this->graph = std::make_unique<core::Tempest>(
        /*is_directed=*/true, TypeParam::value, /*max_time_capacity=*/100);

    test_util::add_edges(*this->graph, {
        Edge {1, 2, 100},
        Edge {2, 3, 100},
        Edge {3, 1, 100}
    });
    EXPECT_EQ(this->graph->get_node_ids().size(), 3u);

    test_util::add_edges(*this->graph, {Edge {4, 5, 250}});

    auto remaining_nodes = this->graph->get_node_ids();
    EXPECT_EQ(remaining_nodes.size(), 2u);
    EXPECT_TRUE(std::find(remaining_nodes.begin(), remaining_nodes.end(), 4) != remaining_nodes.end());
    EXPECT_TRUE(std::find(remaining_nodes.begin(), remaining_nodes.end(), 5) != remaining_nodes.end());
}

TYPED_TEST(TemporalGraphTest, UndirectedGraphEdgeAdditionTest) {
    this->graph = std::make_unique<core::Tempest>(
        /*is_directed=*/false, TypeParam::value);

    test_util::add_edges(*this->graph, {
        Edge {2, 1, 100},
        Edge {3, 1, 200}
    });
    const auto stored_edges = this->graph->get_edges();

    EXPECT_EQ(stored_edges[0].u, 2);
    EXPECT_EQ(stored_edges[0].i, 1);
    EXPECT_EQ(stored_edges[1].u, 3);
    EXPECT_EQ(stored_edges[1].i, 1);
}

TYPED_TEST(TemporalGraphTest, NodeAdjacencyCsrBuildsWhenTemporalNode2VecEnabled) {
    this->graph = std::make_unique<core::Tempest>(
        /*is_directed=*/true,
        TypeParam::value,
        /*max_time_capacity=*/-1,
        /*enable_weight_computation=*/false,
        /*enable_temporal_node2vec=*/true,
        /*timescale_bound=*/-1,
        DEFAULT_NODE2VEC_P,
        DEFAULT_NODE2VEC_Q);

    test_util::add_edges(*this->graph, {
        Edge{0, 1, 100},
        Edge{1, 2, 200},
        Edge{2, 3, 300}
    });

    const auto& d = this->data();
    const size_t active = edge_data::active_node_count(d);
    EXPECT_EQ(d.node_adj_offsets.size(), active + 1);
    EXPECT_EQ(d.node_adj_neighbors.size(), 2 * edge_data::size(d));
}

TYPED_TEST(TemporalGraphTest, NodeAdjacencyCsrNotBuiltWhenTemporalNode2VecDisabled) {
    test_util::add_edges(*this->graph, {
        Edge{0, 1, 100},
        Edge{1, 2, 200},
        Edge{2, 3, 300}
    });

    const auto& d = this->data();
    EXPECT_EQ(d.node_adj_offsets.size(), 0u);
    EXPECT_EQ(d.node_adj_neighbors.size(), 0u);
}

TYPED_TEST(TemporalGraphTest, CountTimestampsTest) {
    test_util::add_edges(*this->graph, {
        Edge {1, 2, 100},
        Edge {2, 3, 100},
        Edge {1, 3, 200},
        Edge {2, 4, 300},
        Edge {3, 4, 300},
        Edge {4, 1, 400}
    });

    const auto& d = this->data();

    EXPECT_EQ(test_util::count_timestamps_less_than(d, 50),  0u);
    EXPECT_EQ(test_util::count_timestamps_less_than(d, 100), 0u);
    EXPECT_EQ(test_util::count_timestamps_less_than(d, 150), 1u);
    EXPECT_EQ(test_util::count_timestamps_less_than(d, 200), 1u);
    EXPECT_EQ(test_util::count_timestamps_less_than(d, 300), 2u);
    EXPECT_EQ(test_util::count_timestamps_less_than(d, 400), 3u);
    EXPECT_EQ(test_util::count_timestamps_less_than(d, 500), 4u);

    EXPECT_EQ(test_util::count_timestamps_greater_than(d, 50),  4u);
    EXPECT_EQ(test_util::count_timestamps_greater_than(d, 100), 3u);
    EXPECT_EQ(test_util::count_timestamps_greater_than(d, 150), 3u);
    EXPECT_EQ(test_util::count_timestamps_greater_than(d, 200), 2u);
    EXPECT_EQ(test_util::count_timestamps_greater_than(d, 300), 1u);
    EXPECT_EQ(test_util::count_timestamps_greater_than(d, 400), 0u);
    EXPECT_EQ(test_util::count_timestamps_greater_than(d, 500), 0u);

    this->graph = std::make_unique<core::Tempest>(true, TypeParam::value);
    EXPECT_EQ(test_util::count_timestamps_less_than(this->data(), 100), 0u);
    EXPECT_EQ(test_util::count_timestamps_greater_than(this->data(), 100), 0u);
}

TYPED_TEST(TemporalGraphTest, CountNodeTimestampsDirectedTest) {
    test_util::add_edges(*this->graph, {
        Edge {1, 2, 100},
        Edge {1, 3, 100},
        Edge {2, 1, 200},
        Edge {1, 2, 300},
        Edge {3, 1, 300},
        Edge {1, 4, 400}
    });

    const auto& d = this->data();

    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 1, 50),  3u);
    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 1, 100), 2u);
    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 1, 200), 2u);
    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 1, 300), 1u);
    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 1, 400), 0u);
    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 1, 500), 0u);

    EXPECT_EQ(test_util::count_node_timestamps_less_than(d, 1, 50),  0u);
    EXPECT_EQ(test_util::count_node_timestamps_less_than(d, 1, 200), 0u);
    EXPECT_EQ(test_util::count_node_timestamps_less_than(d, 1, 250), 1u);
    EXPECT_EQ(test_util::count_node_timestamps_less_than(d, 1, 400), 2u);
    EXPECT_EQ(test_util::count_node_timestamps_less_than(d, 1, 500), 2u);

    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 2, 50),  1u);
    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 2, 200), 0u);
    EXPECT_EQ(test_util::count_node_timestamps_less_than(d, 2, 400),   2u);

    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 5, 100), 0u);
    EXPECT_EQ(test_util::count_node_timestamps_less_than(d, 5, 100),    0u);

    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, -1, 100), 0u);
    EXPECT_EQ(test_util::count_node_timestamps_less_than(d, -1, 100),    0u);
}

TYPED_TEST(TemporalGraphTest, CountNodeTimestampsUndirectedTest) {
    this->graph = std::make_unique<core::Tempest>(
        /*is_directed=*/false, TypeParam::value);

    test_util::add_edges(*this->graph, {
        Edge {2, 1, 100},
        Edge {3, 1, 100},
        Edge {1, 2, 200},
        Edge {4, 1, 300},
        Edge {1, 3, 300}
    });

    const auto& d = this->data();

    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 1, 50),  3u);
    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 1, 100), 2u);
    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 1, 200), 1u);
    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 1, 300), 0u);
    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 1, 400), 0u);

    EXPECT_EQ(test_util::count_node_timestamps_less_than(d, 1, 50),  0u);
    EXPECT_EQ(test_util::count_node_timestamps_less_than(d, 1, 100), 0u);
    EXPECT_EQ(test_util::count_node_timestamps_less_than(d, 1, 150), 1u);
    EXPECT_EQ(test_util::count_node_timestamps_less_than(d, 1, 400), 3u);

    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 2, 50),  2u);
    EXPECT_EQ(test_util::count_node_timestamps_less_than(d, 2, 250),   2u);
}

TYPED_TEST(TemporalGraphTest, CountNodeTimestampsDuplicatesTest) {
    test_util::add_edges(*this->graph, {
        Edge {1, 2, 100},
        Edge {1, 3, 100},
        Edge {1, 4, 100},
        Edge {2, 1, 200},
        Edge {3, 1, 200}
    });

    const auto& d = this->data();

    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 1, 50),  1u);
    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 1, 100), 0u);

    EXPECT_EQ(test_util::count_node_timestamps_less_than(d, 1, 250),    1u);
    EXPECT_EQ(test_util::count_node_timestamps_less_than(d, 1, 150),    0u);

    EXPECT_EQ(test_util::count_node_timestamps_less_than(d, 2, 150),    1u);
    EXPECT_EQ(test_util::count_node_timestamps_greater_than(d, 2, 50),  1u);
}

TYPED_TEST(TemporalGraphTest, GetEdgeAtTest) {
    test_util::add_edges(*this->graph, {
        Edge {10, 20, 100},
        Edge {30, 40, 100},
        Edge {50, 60, 200},
        Edge {70, 80, 300},
        Edge {90, 100, 300},
        Edge {110, 120, 400}
    });

    const auto& d = this->data();

    auto edge1 = test_util::get_edge_at(d, RandomPickerType::TEST_FIRST, -1, true);
    EXPECT_EQ(edge1.ts, 100);

    auto edge2 = test_util::get_edge_at(d, RandomPickerType::TEST_LAST, -1, true);
    EXPECT_EQ(edge2.ts, 400);

    auto edge3 = test_util::get_edge_at(d, RandomPickerType::TEST_FIRST, 100, true);
    EXPECT_EQ(edge3.ts, 200);

    auto edge4 = test_util::get_edge_at(d, RandomPickerType::TEST_FIRST, 300, true);
    EXPECT_EQ(edge4.ts, 400);

    auto edge5 = test_util::get_edge_at(d, RandomPickerType::TEST_FIRST, 400, false);
    EXPECT_EQ(edge5.ts, 100);

    auto edge6 = test_util::get_edge_at(d, RandomPickerType::TEST_LAST, 250, false);
    EXPECT_EQ(edge6.ts, 200);

    auto edge7 = test_util::get_edge_at(d, RandomPickerType::TEST_FIRST, 500, true);
    EXPECT_EQ(edge7.ts, -1);

    auto edge8 = test_util::get_edge_at(d, RandomPickerType::TEST_FIRST, 50, false);
    EXPECT_EQ(edge8.ts, -1);

    this->graph = std::make_unique<core::Tempest>(true, TypeParam::value);
    auto edge9 = test_util::get_edge_at(this->data(), RandomPickerType::TEST_FIRST, 100, true);
    EXPECT_EQ(edge9.ts, -1);
}

TYPED_TEST(TemporalGraphTest, GetEdgeAtDuplicateTimestampsTest) {
    test_util::add_edges(*this->graph, {
        Edge {10, 20, 100},
        Edge {30, 40, 100},
        Edge {50, 60, 100},
        Edge {70, 80, 200},
        Edge {90, 100, 200},
        Edge {110, 120, 300}
    });

    const auto& d = this->data();

    auto edge1 = test_util::get_edge_at(d, RandomPickerType::TEST_FIRST, 50, true);
    EXPECT_EQ(edge1.ts, 100);
    EXPECT_TRUE((edge1.u == 10 && edge1.i == 20) ||
                (edge1.u == 30 && edge1.i == 40) ||
                (edge1.u == 50 && edge1.i == 60));

    auto edge2 = test_util::get_edge_at(d, RandomPickerType::TEST_FIRST, 250, false);
    EXPECT_EQ(edge2.ts, 100);
    EXPECT_TRUE((edge2.u == 10 && edge2.i == 20) ||
                (edge2.u == 30 && edge2.i == 40) ||
                (edge2.u == 50 && edge2.i == 60));
}

TYPED_TEST(TemporalGraphTest, GetEdgeAtBoundaryConditionsTest) {
    test_util::add_edges(*this->graph, {
        Edge {10, 20, 100},
        Edge {30, 40, 200},
        Edge {50, 60, 300}
    });

    const auto& d = this->data();

    auto edge1 = test_util::get_edge_at(d, RandomPickerType::TEST_FIRST, 100, true);
    EXPECT_EQ(edge1.ts, 200);

    auto edge2 = test_util::get_edge_at(d, RandomPickerType::TEST_FIRST, 300, true);
    EXPECT_EQ(edge2.ts, -1);

    auto edge3 = test_util::get_edge_at(d, RandomPickerType::TEST_FIRST, 200, false);
    EXPECT_EQ(edge3.ts, 100);

    auto edge4 = test_util::get_edge_at(d, RandomPickerType::TEST_FIRST, 100, false);
    EXPECT_EQ(edge4.ts, -1);
}

TYPED_TEST(TemporalGraphTest, GetEdgeAtRandomSelectionTest) {
    test_util::add_edges(*this->graph, {
        Edge {1, 2, 100},
        Edge {3, 4, 100},
        Edge {5, 6, 100}
    });

    const auto& d = this->data();

    std::set<std::pair<int, int>> seen_edges;
    constexpr int NUM_TRIES = 50;

    for (int i = 0; i < NUM_TRIES; i++) {
        auto edge = test_util::get_edge_at(d, RandomPickerType::TEST_FIRST, 50, true);
        EXPECT_EQ(edge.ts, 100);
        seen_edges.insert({edge.u, edge.i});
    }

    EXPECT_GT(seen_edges.size(), 1u);
}
