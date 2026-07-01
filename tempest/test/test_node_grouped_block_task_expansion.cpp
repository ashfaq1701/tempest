// sub-task output order is non-deterministic across source tasks; assertions
// group by node_id and sort by walk_start.

#ifdef HAS_CUDA

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include "../src/common/const.cuh"
#include "../src/common/cuda_config.cuh"
#include "../src/common/cuda_config.cuh"
#include "../src/core/node_grouped/scheduler.cuh"
#include "../src/data/enums.cuh"
#include "../src/data/walk_set/walk_set_view.cuh"

namespace {

using tempest::NodeGroupedScheduler;

constexpr int PAD = EMPTY_NODE_VALUE;
constexpr int CAP = W_THRESHOLD_MULTI_BLOCK;

struct DeviceWalkSet {
    int*         nodes              = nullptr;
    int64_t*     timestamps         = nullptr;
    std::size_t* walk_lens          = nullptr;
    int64_t*     edge_ids           = nullptr;
    std::size_t  num_walks          = 0;
    std::size_t  max_len            = 0;
    int          walk_padding_value = EMPTY_NODE_VALUE;

    ~DeviceWalkSet() {
        if (nodes)      cudaFree(nodes);
        if (timestamps) cudaFree(timestamps);
        if (walk_lens)  cudaFree(walk_lens);
        if (edge_ids)   cudaFree(edge_ids);
    }

    DeviceWalkSet() = default;
    DeviceWalkSet(const DeviceWalkSet&)            = delete;
    DeviceWalkSet& operator=(const DeviceWalkSet&) = delete;
};

std::unique_ptr<DeviceWalkSet> make_walk_set(
    const std::vector<int>& last_nodes_at_step,
    const int step_number,
    const std::size_t max_walk_len,
    const int padding_value) {

    auto ws = std::make_unique<DeviceWalkSet>();
    ws->num_walks          = last_nodes_at_step.size();
    ws->max_len            = max_walk_len;
    ws->walk_padding_value = padding_value;

    const std::size_t total_nodes = ws->num_walks * max_walk_len;
    std::vector<int> h_nodes(total_nodes, padding_value);
    for (std::size_t w = 0; w < ws->num_walks; ++w) {
        h_nodes[w * max_walk_len + static_cast<std::size_t>(step_number)] =
            last_nodes_at_step[w];
    }

    cudaMalloc(&ws->nodes, total_nodes * sizeof(int));
    cudaMemcpy(ws->nodes, h_nodes.data(),
               total_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&ws->timestamps, total_nodes * sizeof(int64_t));
    cudaMalloc(&ws->walk_lens,  ws->num_walks * sizeof(std::size_t));
    const std::size_t edge_total =
        ws->num_walks * (max_walk_len > 0 ? max_walk_len - 1 : 0);
    if (edge_total > 0) {
        cudaMalloc(&ws->edge_ids, edge_total * sizeof(int64_t));
    }
    return ws;
}

WalkSetView make_view(const DeviceWalkSet& ws) {
    WalkSetView v{};
    v.nodes              = ws.nodes;
    v.timestamps         = ws.timestamps;
    v.walk_lens          = ws.walk_lens;
    v.edge_ids           = ws.edge_ids;
    v.num_walks          = ws.num_walks;
    v.max_len            = ws.max_len;
    v.walk_padding_value = ws.walk_padding_value;
    return v;
}

struct DeviceG {
    std::size_t* ptr = nullptr;
    ~DeviceG() { if (ptr) cudaFree(ptr); }
};

std::unique_ptr<DeviceG> make_count_ts_group_per_node(
    const std::vector<int>& g_per_node) {
    auto g = std::make_unique<DeviceG>();
    const std::size_t n = g_per_node.size() + 1;
    std::vector<std::size_t> h(n, 0);
    for (std::size_t node = 0; node < g_per_node.size(); ++node) {
        h[node + 1] = h[node] + static_cast<std::size_t>(g_per_node[node]);
    }
    cudaMalloc(&g->ptr, n * sizeof(std::size_t));
    cudaMemcpy(g->ptr, h.data(), n * sizeof(std::size_t),
               cudaMemcpyHostToDevice);
    return g;
}

void set_g(std::vector<int>& g_per_node, int node_id, int g) {
    if (static_cast<int>(g_per_node.size()) <= node_id) {
        g_per_node.resize(node_id + 1, 0);
    }
    g_per_node[node_id] = g;
}

struct ExpResult {
    int num_active_host        = 0;
    int num_solo_walks_host    = 0;
    int num_warp_smem_host     = 0;
    int num_warp_global_host   = 0;
    int num_block_smem_host    = 0;
    int num_block_global_host  = 0;

    std::vector<int> warp_smem_nodes;
    std::vector<int> warp_smem_walk_counts;

    std::vector<int> block_smem_nodes;
    std::vector<int> block_smem_walk_starts;
    std::vector<int> block_smem_walk_counts;

    std::vector<int> block_global_nodes;
    std::vector<int> block_global_walk_starts;
    std::vector<int> block_global_walk_counts;
};

std::vector<int> download_ints(const int* device_ptr, const std::size_t n) {
    std::vector<int> h(n);
    if (n > 0 && device_ptr != nullptr) {
        cudaMemcpy(h.data(), device_ptr,
                   n * sizeof(int), cudaMemcpyDeviceToHost);
    }
    return h;
}

ExpResult run_and_materialize(
    const std::vector<int>& last_nodes_at_step,
    const std::vector<int>& g_per_node,
    const RandomPickerType picker,
    const dim3 block_dim,
    const cudaStream_t stream) {

    constexpr int STEP_NUMBER  = 1;
    constexpr int MAX_WALK_LEN = 4;

    auto ws = make_walk_set(last_nodes_at_step, STEP_NUMBER,
                            MAX_WALK_LEN, PAD);
    const WalkSetView view = make_view(*ws);
    auto g = make_count_ts_group_per_node(g_per_node);

    NodeGroupedScheduler scheduler(ws->num_walks, block_dim, /*w_threshold_warp=*/1, stream);
    auto outs = scheduler.run_step(
        view, STEP_NUMBER, MAX_WALK_LEN, g->ptr, picker);
    cudaStreamSynchronize(stream);

    ExpResult r;
    r.num_active_host        = outs.num_active_host;
    r.num_solo_walks_host    = outs.num_solo_walks_host;
    r.num_warp_smem_host     = outs.warp_smem.num_tasks_host;
    r.num_warp_global_host   = outs.warp_global.num_tasks_host;
    r.num_block_smem_host    = outs.block_smem.num_tasks_host;
    r.num_block_global_host  = outs.block_global.num_tasks_host;

    r.warp_smem_nodes = download_ints(
        outs.warp_smem.nodes, static_cast<std::size_t>(r.num_warp_smem_host));
    r.warp_smem_walk_counts = download_ints(
        outs.warp_smem.walk_counts, static_cast<std::size_t>(r.num_warp_smem_host));

    r.block_smem_nodes = download_ints(
        outs.block_smem.nodes, static_cast<std::size_t>(r.num_block_smem_host));
    r.block_smem_walk_starts = download_ints(
        outs.block_smem.walk_starts, static_cast<std::size_t>(r.num_block_smem_host));
    r.block_smem_walk_counts = download_ints(
        outs.block_smem.walk_counts, static_cast<std::size_t>(r.num_block_smem_host));

    r.block_global_nodes = download_ints(
        outs.block_global.nodes, static_cast<std::size_t>(r.num_block_global_host));
    r.block_global_walk_starts = download_ints(
        outs.block_global.walk_starts, static_cast<std::size_t>(r.num_block_global_host));
    r.block_global_walk_counts = download_ints(
        outs.block_global.walk_counts, static_cast<std::size_t>(r.num_block_global_host));

    return r;
}

struct SubTask { int walk_start; int walk_count; };

std::vector<SubTask> sub_tasks_for_node(
    const std::vector<int>& nodes,
    const std::vector<int>& starts,
    const std::vector<int>& counts,
    const int node_id) {
    std::vector<SubTask> out;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i] == node_id) {
            out.push_back({starts[i], counts[i]});
        }
    }
    std::sort(out.begin(), out.end(),
              [](const SubTask& a, const SubTask& b) {
                  return a.walk_start < b.walk_start;
              });
    return out;
}

}  // namespace

class GpuBlockTaskExpansionTest : public ::testing::Test {
protected:
    void SetUp() override {
        block_dim_ = dim3(static_cast<unsigned>(BLOCK_DIM));
        cudaStreamCreate(&stream_);
    }

    void TearDown() override {
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    dim3 block_dim_{};
    cudaStream_t stream_ = nullptr;
};

TEST_F(GpuBlockTaskExpansionTest, BlockBelowCap_NoExpansion) {
    const int NODE = 7;
    std::vector<int> last_nodes(300, NODE);
    std::vector<int> g; set_g(g, NODE, 10);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    ASSERT_EQ(r.num_block_smem_host, 1);
    EXPECT_EQ(r.block_smem_nodes[0], NODE);
    EXPECT_EQ(r.block_smem_walk_counts[0], 300);
}

TEST_F(GpuBlockTaskExpansionTest, BlockExactlyAtCap_StillSingleTask) {
    const int NODE = 7;
    std::vector<int> last_nodes(CAP, NODE);
    std::vector<int> g; set_g(g, NODE, 10);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    ASSERT_EQ(r.num_block_smem_host, 1);
    EXPECT_EQ(r.block_smem_nodes[0], NODE);
    EXPECT_EQ(r.block_smem_walk_counts[0], CAP);
    EXPECT_EQ(r.block_smem_walk_starts[0], 0);
}

TEST_F(GpuBlockTaskExpansionTest, BlockJustOverCap_SplitsIntoTwo) {
    const int NODE = 7;
    const int W = CAP + 1;
    std::vector<int> last_nodes(W, NODE);
    std::vector<int> g; set_g(g, NODE, 10);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    ASSERT_EQ(r.num_block_smem_host, 2);

    const auto subs = sub_tasks_for_node(r.block_smem_nodes,
                                         r.block_smem_walk_starts,
                                         r.block_smem_walk_counts,
                                         NODE);
    ASSERT_EQ(subs.size(), 2u);
    EXPECT_EQ(subs[0].walk_count, CAP);
    EXPECT_EQ(subs[1].walk_count, 1);
    EXPECT_EQ(subs[0].walk_start, 0);
    EXPECT_EQ(subs[1].walk_start, CAP);
}

TEST_F(GpuBlockTaskExpansionTest, BlockThreeTimesCap_SplitsIntoThree) {
    const int NODE = 7;
    const int W = 3 * CAP;
    std::vector<int> last_nodes(W, NODE);
    std::vector<int> g; set_g(g, NODE, 10);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    ASSERT_EQ(r.num_block_smem_host, 3);
    const auto subs = sub_tasks_for_node(r.block_smem_nodes,
                                         r.block_smem_walk_starts,
                                         r.block_smem_walk_counts,
                                         NODE);
    ASSERT_EQ(subs.size(), 3u);
    for (const auto& s : subs) EXPECT_EQ(s.walk_count, CAP);
    EXPECT_EQ(subs[0].walk_start, 0);
    EXPECT_EQ(subs[1].walk_start, CAP);
    EXPECT_EQ(subs[2].walk_start, 2 * CAP);
}

TEST_F(GpuBlockTaskExpansionTest, BlockThreeCapPlusRemainder_SplitsIntoFour) {
    const int NODE = 7;
    const int REMAINDER = 5;
    const int W = 3 * CAP + REMAINDER;
    std::vector<int> last_nodes(W, NODE);
    std::vector<int> g; set_g(g, NODE, 10);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    ASSERT_EQ(r.num_block_smem_host, 4);
    const auto subs = sub_tasks_for_node(r.block_smem_nodes,
                                         r.block_smem_walk_starts,
                                         r.block_smem_walk_counts,
                                         NODE);
    ASSERT_EQ(subs.size(), 4u);
    EXPECT_EQ(subs[0].walk_count, CAP);
    EXPECT_EQ(subs[1].walk_count, CAP);
    EXPECT_EQ(subs[2].walk_count, CAP);
    EXPECT_EQ(subs[3].walk_count, REMAINDER);
    EXPECT_EQ(subs[0].walk_start, 0);
    EXPECT_EQ(subs[1].walk_start, CAP);
    EXPECT_EQ(subs[2].walk_start, 2 * CAP);
    EXPECT_EQ(subs[3].walk_start, 3 * CAP);
}

TEST_F(GpuBlockTaskExpansionTest, SubTasksTileOriginalRangeExactly) {
    const int NODE = 42;
    const int W = 5 * CAP + 123;
    std::vector<int> last_nodes(W, NODE);
    std::vector<int> g; set_g(g, NODE, 10);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    ASSERT_EQ(r.num_block_smem_host, 6);
    const auto subs = sub_tasks_for_node(r.block_smem_nodes,
                                         r.block_smem_walk_starts,
                                         r.block_smem_walk_counts,
                                         NODE);
    ASSERT_EQ(subs.size(), 6u);

    for (std::size_t k = 0; k + 1 < subs.size(); ++k) {
        EXPECT_EQ(subs[k].walk_start + subs[k].walk_count,
                  subs[k + 1].walk_start);
    }
    EXPECT_EQ(subs.back().walk_start + subs.back().walk_count, W);

    int total = 0;
    for (const auto& s : subs) total += s.walk_count;
    EXPECT_EQ(total, W);
}

TEST_F(GpuBlockTaskExpansionTest, NodeIdPreservedAcrossSubTasks) {
    const int NODE = 12345;
    const int W = 2 * CAP + 50;
    std::vector<int> last_nodes(W, NODE);
    std::vector<int> g; set_g(g, NODE, 10);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    ASSERT_EQ(r.num_block_smem_host, 3);
    for (int got : r.block_smem_nodes) {
        EXPECT_EQ(got, NODE);
    }
}

TEST_F(GpuBlockTaskExpansionTest, MultipleMegaHubs_EachExpandsIndependently) {
    const int NODE_A = 10;
    const int NODE_B = 20;
    const int NODE_C = 30;
    const int W_A = CAP + 1;
    const int W_B = 3 * CAP;
    const int W_C = CAP / 2;

    std::vector<int> last_nodes;
    for (int i = 0; i < W_A; ++i) last_nodes.push_back(NODE_A);
    for (int i = 0; i < W_B; ++i) last_nodes.push_back(NODE_B);
    for (int i = 0; i < W_C; ++i) last_nodes.push_back(NODE_C);

    std::vector<int> g;
    set_g(g, NODE_A, 10);
    set_g(g, NODE_B, 10);
    set_g(g, NODE_C, 10);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_block_smem_host, 2 + 3 + 1);

    const auto subs_a = sub_tasks_for_node(r.block_smem_nodes,
                                           r.block_smem_walk_starts,
                                           r.block_smem_walk_counts, NODE_A);
    const auto subs_b = sub_tasks_for_node(r.block_smem_nodes,
                                           r.block_smem_walk_starts,
                                           r.block_smem_walk_counts, NODE_B);
    const auto subs_c = sub_tasks_for_node(r.block_smem_nodes,
                                           r.block_smem_walk_starts,
                                           r.block_smem_walk_counts, NODE_C);

    ASSERT_EQ(subs_a.size(), 2u);
    ASSERT_EQ(subs_b.size(), 3u);
    ASSERT_EQ(subs_c.size(), 1u);

    int sum_a = 0; for (const auto& s : subs_a) sum_a += s.walk_count;
    int sum_b = 0; for (const auto& s : subs_b) sum_b += s.walk_count;
    int sum_c = 0; for (const auto& s : subs_c) sum_c += s.walk_count;
    EXPECT_EQ(sum_a, W_A);
    EXPECT_EQ(sum_b, W_B);
    EXPECT_EQ(sum_c, W_C);
}

TEST_F(GpuBlockTaskExpansionTest, BlockGlobal_IsAlsoExpanded) {
    const int NODE = 7;
    const int W = 2 * CAP + 10;
    const int G_LARGE = G_THRESHOLD_BLOCK_INDEX + 1;
    std::vector<int> last_nodes(W, NODE);
    std::vector<int> g; set_g(g, NODE, G_LARGE);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_block_smem_host, 0);
    ASSERT_EQ(r.num_block_global_host, 3);

    const auto subs = sub_tasks_for_node(r.block_global_nodes,
                                         r.block_global_walk_starts,
                                         r.block_global_walk_counts,
                                         NODE);
    ASSERT_EQ(subs.size(), 3u);
    EXPECT_EQ(subs[0].walk_count, CAP);
    EXPECT_EQ(subs[1].walk_count, CAP);
    EXPECT_EQ(subs[2].walk_count, 10);
    EXPECT_EQ(subs[0].walk_start, 0);
    EXPECT_EQ(subs[1].walk_start, CAP);
    EXPECT_EQ(subs[2].walk_start, 2 * CAP);
}

TEST_F(GpuBlockTaskExpansionTest, WarpTier_NotExpanded_EvenAtWarpUpperBound) {
    const int NODE = 5;
    const int W = static_cast<int>(BLOCK_DIM);
    std::vector<int> last_nodes(W, NODE);
    std::vector<int> g; set_g(g, NODE, 10);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_block_smem_host, 0);
    EXPECT_EQ(r.num_block_global_host, 0);
    ASSERT_EQ(r.num_warp_smem_host, 1);
    EXPECT_EQ(r.warp_smem_nodes[0], NODE);
    EXPECT_EQ(r.warp_smem_walk_counts[0], W);
}

TEST_F(GpuBlockTaskExpansionTest, CountIdentity_WithMegaHubExpansion) {
    const int NODE_SOLO   = 101;
    const int NODE_WARP   = 11;
    const int NODE_BLOCK  = 13;
    const int NODE_MEGA   = 14;

    const int W_WARP  = 10;
    const int W_BLOCK = CAP / 2;
    const int W_MEGA  = 3 * CAP + 17;

    std::vector<int> last_nodes;
    last_nodes.push_back(NODE_SOLO);
    for (int i = 0; i < W_WARP;  ++i) last_nodes.push_back(NODE_WARP);
    for (int i = 0; i < W_BLOCK; ++i) last_nodes.push_back(NODE_BLOCK);
    for (int i = 0; i < W_MEGA;  ++i) last_nodes.push_back(NODE_MEGA);
    const int expected_active = 1 + W_WARP + W_BLOCK + W_MEGA;

    std::vector<int> g;
    set_g(g, NODE_SOLO,  50);
    set_g(g, NODE_WARP,  100);
    set_g(g, NODE_BLOCK, 100);
    set_g(g, NODE_MEGA,  100);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, expected_active);
    EXPECT_EQ(r.num_solo_walks_host,  1);
    EXPECT_EQ(r.num_warp_smem_host,   1);
    EXPECT_EQ(r.num_warp_global_host, 0);
    const int expected_mega_subs = (W_MEGA + CAP - 1) / CAP;
    EXPECT_EQ(r.num_block_smem_host, 1 + expected_mega_subs);
    EXPECT_EQ(r.num_block_global_host, 0);

    int total = r.num_solo_walks_host;
    total += std::accumulate(r.warp_smem_walk_counts.begin(),
                             r.warp_smem_walk_counts.end(), 0);
    total += std::accumulate(r.block_smem_walk_counts.begin(),
                             r.block_smem_walk_counts.end(), 0);
    total += std::accumulate(r.block_global_walk_counts.begin(),
                             r.block_global_walk_counts.end(), 0);
    EXPECT_EQ(total, expected_active);
}

#endif  // HAS_CUDA
