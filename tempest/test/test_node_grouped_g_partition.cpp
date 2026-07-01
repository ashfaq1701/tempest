#ifdef HAS_CUDA

#include <gtest/gtest.h>

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

// builds prefix-sum array: count_ts_group_per_node[node+1] - [node] == G_node.
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

struct GResult {
    int num_active_host        = 0;
    int num_solo_walks_host    = 0;
    int num_warp_smem_host     = 0;
    int num_warp_global_host   = 0;
    int num_block_smem_host    = 0;
    int num_block_global_host  = 0;

    std::vector<int> solo_walks;

    std::vector<int> warp_smem_nodes;
    std::vector<int> warp_smem_walk_counts;
    std::vector<int> warp_global_nodes;
    std::vector<int> warp_global_walk_counts;

    std::vector<int> block_smem_nodes;
    std::vector<int> block_smem_walk_counts;
    std::vector<int> block_global_nodes;
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

GResult run_and_materialize(
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

    GResult r;
    r.num_active_host        = outs.num_active_host;
    r.num_solo_walks_host    = outs.num_solo_walks_host;
    r.num_warp_smem_host     = outs.warp_smem.num_tasks_host;
    r.num_warp_global_host   = outs.warp_global.num_tasks_host;
    r.num_block_smem_host    = outs.block_smem.num_tasks_host;
    r.num_block_global_host  = outs.block_global.num_tasks_host;

    r.solo_walks = download_ints(
        outs.solo_walks, static_cast<std::size_t>(r.num_solo_walks_host));

    r.warp_smem_nodes = download_ints(
        outs.warp_smem.nodes, static_cast<std::size_t>(r.num_warp_smem_host));
    r.warp_smem_walk_counts = download_ints(
        outs.warp_smem.walk_counts, static_cast<std::size_t>(r.num_warp_smem_host));

    r.warp_global_nodes = download_ints(
        outs.warp_global.nodes, static_cast<std::size_t>(r.num_warp_global_host));
    r.warp_global_walk_counts = download_ints(
        outs.warp_global.walk_counts, static_cast<std::size_t>(r.num_warp_global_host));

    r.block_smem_nodes = download_ints(
        outs.block_smem.nodes, static_cast<std::size_t>(r.num_block_smem_host));
    r.block_smem_walk_counts = download_ints(
        outs.block_smem.walk_counts, static_cast<std::size_t>(r.num_block_smem_host));

    r.block_global_nodes = download_ints(
        outs.block_global.nodes, static_cast<std::size_t>(r.num_block_global_host));
    r.block_global_walk_counts = download_ints(
        outs.block_global.walk_counts, static_cast<std::size_t>(r.num_block_global_host));

    return r;
}

void set_g(std::vector<int>& g_per_node, int node_id, int g) {
    if (static_cast<int>(g_per_node.size()) <= node_id) {
        g_per_node.resize(node_id + 1, 0);
    }
    g_per_node[node_id] = g;
}

}  // namespace

class GpuGPartitionTest : public ::testing::Test {
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

TEST_F(GpuGPartitionTest, WarpTier_SmallG_RoutesToWarpSmem) {
    // W=10, G=100 < 340 -> warp_smem.
    const int NODE = 5;
    std::vector<int> last_nodes(10, NODE);
    std::vector<int> g; set_g(g, NODE, 100);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_warp_smem_host,    1);
    EXPECT_EQ(r.num_warp_global_host,  0);
    EXPECT_EQ(r.num_block_smem_host,   0);
    EXPECT_EQ(r.num_block_global_host, 0);
    ASSERT_EQ(r.warp_smem_walk_counts.size(), 1u);
    EXPECT_EQ(r.warp_smem_nodes[0], NODE);
    EXPECT_EQ(r.warp_smem_walk_counts[0], 10);
}

TEST_F(GpuGPartitionTest, WarpTier_LargeG_RoutesToWarpGlobal) {
    // W=10, G=400 > 340 -> warp_global.
    const int NODE = 5;
    std::vector<int> last_nodes(10, NODE);
    std::vector<int> g; set_g(g, NODE, 400);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_warp_smem_host,    0);
    EXPECT_EQ(r.num_warp_global_host,  1);
    EXPECT_EQ(r.num_block_smem_host,   0);
    EXPECT_EQ(r.num_block_global_host, 0);
    ASSERT_EQ(r.warp_global_walk_counts.size(), 1u);
    EXPECT_EQ(r.warp_global_nodes[0], NODE);
    EXPECT_EQ(r.warp_global_walk_counts[0], 10);
}

TEST_F(GpuGPartitionTest, BlockTier_SmallG_RoutesToBlockSmem) {
    // W=300, G=1000 < 2800 -> block_smem.
    const int NODE = 7;
    std::vector<int> last_nodes(300, NODE);
    std::vector<int> g; set_g(g, NODE, 1000);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_warp_smem_host,    0);
    EXPECT_EQ(r.num_warp_global_host,  0);
    EXPECT_EQ(r.num_block_smem_host,   1);
    EXPECT_EQ(r.num_block_global_host, 0);
    EXPECT_EQ(r.block_smem_nodes[0],  NODE);
    EXPECT_EQ(r.block_smem_walk_counts[0], 300);
}

TEST_F(GpuGPartitionTest, BlockTier_LargeG_RoutesToBlockGlobal) {
    // W=300, G=3000 > 2800 -> block_global.
    const int NODE = 7;
    std::vector<int> last_nodes(300, NODE);
    std::vector<int> g; set_g(g, NODE, 3000);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_warp_smem_host,    0);
    EXPECT_EQ(r.num_warp_global_host,  0);
    EXPECT_EQ(r.num_block_smem_host,   0);
    EXPECT_EQ(r.num_block_global_host, 1);
    EXPECT_EQ(r.block_global_nodes[0], NODE);
    EXPECT_EQ(r.block_global_walk_counts[0], 300);
}

TEST_F(GpuGPartitionTest, BoundaryWarpIndex_GEqCap_IsSmem_GCapPlus1_IsGlobal) {
    const int cap    = G_THRESHOLD_WARP_INDEX;
    const int NODE_A = 5;
    const int NODE_B = 6;
    std::vector<int> last_nodes;
    for (int i = 0; i < 10; ++i) last_nodes.push_back(NODE_A);
    for (int i = 0; i < 10; ++i) last_nodes.push_back(NODE_B);

    std::vector<int> g;
    set_g(g, NODE_A, cap);
    set_g(g, NODE_B, cap + 1);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_warp_smem_host,   1);
    EXPECT_EQ(r.num_warp_global_host, 1);
    EXPECT_EQ(r.warp_smem_nodes[0],   NODE_A);
    EXPECT_EQ(r.warp_global_nodes[0], NODE_B);
}

TEST_F(GpuGPartitionTest, BoundaryWarpWeighted_GEqCap_IsSmem_GCapPlus1_IsGlobal) {
    const int cap    = G_THRESHOLD_WARP_WEIGHT;
    const int NODE_A = 5;
    const int NODE_B = 6;
    std::vector<int> last_nodes;
    for (int i = 0; i < 10; ++i) last_nodes.push_back(NODE_A);
    for (int i = 0; i < 10; ++i) last_nodes.push_back(NODE_B);

    std::vector<int> g;
    set_g(g, NODE_A, cap);
    set_g(g, NODE_B, cap + 1);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::ExponentialWeight,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_warp_smem_host,   1);
    EXPECT_EQ(r.num_warp_global_host, 1);
    EXPECT_EQ(r.warp_smem_nodes[0],   NODE_A);
    EXPECT_EQ(r.warp_global_nodes[0], NODE_B);
}

TEST_F(GpuGPartitionTest, BoundaryBlockIndex_GEqCap_IsSmem_GCapPlus1_IsGlobal) {
    const int cap    = G_THRESHOLD_BLOCK_INDEX;
    const int NODE_A = 5;
    const int NODE_B = 6;
    std::vector<int> last_nodes;
    for (int i = 0; i < 300; ++i) last_nodes.push_back(NODE_A);
    for (int i = 0; i < 300; ++i) last_nodes.push_back(NODE_B);

    std::vector<int> g;
    set_g(g, NODE_A, cap);
    set_g(g, NODE_B, cap + 1);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_block_smem_host,   1);
    EXPECT_EQ(r.num_block_global_host, 1);
    EXPECT_EQ(r.block_smem_nodes[0],   NODE_A);
    EXPECT_EQ(r.block_global_nodes[0], NODE_B);
}

TEST_F(GpuGPartitionTest, BoundaryBlockWeighted_GEqCap_IsSmem_GCapPlus1_IsGlobal) {
    const int cap    = G_THRESHOLD_BLOCK_WEIGHT;
    const int NODE_A = 5;
    const int NODE_B = 6;
    std::vector<int> last_nodes;
    for (int i = 0; i < 300; ++i) last_nodes.push_back(NODE_A);
    for (int i = 0; i < 300; ++i) last_nodes.push_back(NODE_B);

    std::vector<int> g;
    set_g(g, NODE_A, cap);
    set_g(g, NODE_B, cap + 1);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::ExponentialWeight,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_block_smem_host,   1);
    EXPECT_EQ(r.num_block_global_host, 1);
    EXPECT_EQ(r.block_smem_nodes[0],   NODE_A);
    EXPECT_EQ(r.block_global_nodes[0], NODE_B);
}

TEST_F(GpuGPartitionTest, PickerClassSelectsCap_WarpTier) {
    // G=300: index (cap 340) -> smem; weighted (cap 220) -> global.
    const int NODE = 5;
    std::vector<int> last_nodes(10, NODE);
    std::vector<int> g; set_g(g, NODE, 300);

    auto r_index = run_and_materialize(
        last_nodes, g, RandomPickerType::Linear, block_dim_, stream_);
    EXPECT_EQ(r_index.num_warp_smem_host,   1);
    EXPECT_EQ(r_index.num_warp_global_host, 0);

    auto r_weighted = run_and_materialize(
        last_nodes, g, RandomPickerType::ExponentialWeight,
        block_dim_, stream_);
    EXPECT_EQ(r_weighted.num_warp_smem_host,   0);
    EXPECT_EQ(r_weighted.num_warp_global_host, 1);
}

TEST_F(GpuGPartitionTest, PickerClassSelectsCap_BlockTier) {
    // G=2000: index (cap 2800) -> smem; weighted (cap 1800) -> global.
    const int NODE = 7;
    std::vector<int> last_nodes(300, NODE);
    std::vector<int> g; set_g(g, NODE, 2000);

    auto r_index = run_and_materialize(
        last_nodes, g, RandomPickerType::Linear, block_dim_, stream_);
    EXPECT_EQ(r_index.num_block_smem_host,   1);
    EXPECT_EQ(r_index.num_block_global_host, 0);

    auto r_weighted = run_and_materialize(
        last_nodes, g, RandomPickerType::ExponentialWeight,
        block_dim_, stream_);
    EXPECT_EQ(r_weighted.num_block_smem_host,   0);
    EXPECT_EQ(r_weighted.num_block_global_host, 1);
}

TEST_F(GpuGPartitionTest, SoloTier_UnaffectedByG) {
    std::vector<int> last_nodes;
    std::vector<int> g;
    for (int i = 0; i < 10; ++i) {
        const int node = 100 + i;
        last_nodes.push_back(node);
        set_g(g, node, 99999);
    }

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_solo_walks_host,   10);
    EXPECT_EQ(r.num_warp_smem_host,    0);
    EXPECT_EQ(r.num_warp_global_host,  0);
    EXPECT_EQ(r.num_block_smem_host,   0);
    EXPECT_EQ(r.num_block_global_host, 0);
}

TEST_F(GpuGPartitionTest, MixedScenario_AllFiveTiersPopulated_CountIdentity) {
    const int NODE_SOLO_A = 101;
    const int NODE_SOLO_B = 102;
    const int NODE_WARP_SMEM   = 11;
    const int NODE_WARP_GLOBAL = 12;
    const int NODE_BLOCK_SMEM  = 13;
    const int NODE_BLOCK_GLOB  = 14;

    std::vector<int> last_nodes;
    last_nodes.push_back(NODE_SOLO_A);
    last_nodes.push_back(NODE_SOLO_B);
    for (int i = 0; i < 10;  ++i) last_nodes.push_back(NODE_WARP_SMEM);
    for (int i = 0; i < 10;  ++i) last_nodes.push_back(NODE_WARP_GLOBAL);
    for (int i = 0; i < 300; ++i) last_nodes.push_back(NODE_BLOCK_SMEM);
    for (int i = 0; i < 300; ++i) last_nodes.push_back(NODE_BLOCK_GLOB);
    const int expected_active = 2 + 10 + 10 + 300 + 300;

    std::vector<int> g;
    set_g(g, NODE_SOLO_A,       50);
    set_g(g, NODE_SOLO_B,       9999);
    set_g(g, NODE_WARP_SMEM,    100);
    set_g(g, NODE_WARP_GLOBAL,  400);
    set_g(g, NODE_BLOCK_SMEM,   1000);
    set_g(g, NODE_BLOCK_GLOB,   3000);

    auto r = run_and_materialize(last_nodes, g,
                                 RandomPickerType::Linear,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host,       expected_active);
    EXPECT_EQ(r.num_solo_walks_host,   2);
    EXPECT_EQ(r.num_warp_smem_host,    1);
    EXPECT_EQ(r.num_warp_global_host,  1);
    EXPECT_EQ(r.num_block_smem_host,   1);
    EXPECT_EQ(r.num_block_global_host, 1);

    ASSERT_EQ(r.warp_smem_nodes.size(),   1u);
    EXPECT_EQ(r.warp_smem_nodes[0],       NODE_WARP_SMEM);
    ASSERT_EQ(r.warp_global_nodes.size(), 1u);
    EXPECT_EQ(r.warp_global_nodes[0],     NODE_WARP_GLOBAL);
    ASSERT_EQ(r.block_smem_nodes.size(),  1u);
    EXPECT_EQ(r.block_smem_nodes[0],      NODE_BLOCK_SMEM);
    ASSERT_EQ(r.block_global_nodes.size(), 1u);
    EXPECT_EQ(r.block_global_nodes[0],    NODE_BLOCK_GLOB);

    int total = r.num_solo_walks_host;
    total += std::accumulate(r.warp_smem_walk_counts.begin(),
                             r.warp_smem_walk_counts.end(), 0);
    total += std::accumulate(r.warp_global_walk_counts.begin(),
                             r.warp_global_walk_counts.end(), 0);
    total += std::accumulate(r.block_smem_walk_counts.begin(),
                             r.block_smem_walk_counts.end(), 0);
    total += std::accumulate(r.block_global_walk_counts.begin(),
                             r.block_global_walk_counts.end(), 0);
    EXPECT_EQ(total, expected_active);
}

#endif  // HAS_CUDA
