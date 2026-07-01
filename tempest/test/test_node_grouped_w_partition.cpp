#ifdef HAS_CUDA

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

#include <cuda_runtime.h>

#include "../src/common/const.cuh"
#include "../src/common/cuda_config.cuh"
#include "../src/common/cuda_config.cuh"
#include "../src/core/node_grouped/scheduler.cuh"
#include "../src/data/walk_set/walk_set_view.cuh"

namespace {

using tempest::NodeGroupedScheduler;

struct DeviceWalkSet {
    int*     nodes              = nullptr;
    int64_t* timestamps         = nullptr;
    std::size_t* walk_lens      = nullptr;
    int64_t* edge_ids           = nullptr;
    std::size_t num_walks       = 0;
    std::size_t max_len         = 0;
    int walk_padding_value      = EMPTY_NODE_VALUE;

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
    ws->num_walks = last_nodes_at_step.size();
    ws->max_len   = max_walk_len;
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

// G=0 setup forces all warp/block tasks into smem variants.
struct SchedulerResult {
    int num_active_host          = 0;
    int num_solo_walks_host      = 0;
    int num_warp_tasks_host      = 0;
    int num_block_tasks_host     = 0;

    std::vector<int> sorted_walk_idx;
    std::vector<int> solo_walks;
    std::vector<int> warp_nodes;
    std::vector<int> warp_walk_starts;
    std::vector<int> warp_walk_counts;
    std::vector<int> block_nodes;
    std::vector<int> block_walk_starts;
    std::vector<int> block_walk_counts;
};

std::vector<int> download_ints(const int* device_ptr, const std::size_t n) {
    std::vector<int> h(n);
    if (n > 0 && device_ptr != nullptr) {
        cudaMemcpy(h.data(), device_ptr,
                   n * sizeof(int), cudaMemcpyDeviceToHost);
    }
    return h;
}

struct DeviceG {
    std::size_t* ptr = nullptr;
    ~DeviceG() { if (ptr) cudaFree(ptr); }
};

std::unique_ptr<DeviceG> make_trivial_count_ts_group(int max_node_id) {
    auto g = std::make_unique<DeviceG>();
    // need indices [0..max_node_id+1] addressable (kernel reads [node] and [node+1]).
    const std::size_t n = static_cast<std::size_t>(max_node_id) + 2;
    cudaMalloc(&g->ptr, n * sizeof(std::size_t));
    cudaMemset(g->ptr, 0, n * sizeof(std::size_t));
    return g;
}

SchedulerResult run_and_materialize(
    const std::vector<int>& last_nodes_at_step,
    const int step_number,
    const std::size_t max_walk_len,
    const int padding_value,
    const dim3 block_dim,
    const cudaStream_t stream) {

    auto ws = make_walk_set(last_nodes_at_step, step_number,
                            max_walk_len, padding_value);
    const WalkSetView view = make_view(*ws);

    int max_node_id = 0;
    for (int n : last_nodes_at_step) {
        if (n != padding_value && n > max_node_id) max_node_id = n;
    }
    auto g = make_trivial_count_ts_group(max_node_id);

    NodeGroupedScheduler scheduler(ws->num_walks, block_dim, /*w_threshold_warp=*/1, stream);
    auto outs = scheduler.run_step(
        view, step_number, static_cast<int>(max_walk_len),
        g->ptr,
        RandomPickerType::Linear);

    cudaStreamSynchronize(stream);

    EXPECT_EQ(outs.warp_global.num_tasks_host, 0);
    EXPECT_EQ(outs.block_global.num_tasks_host, 0);

    SchedulerResult r;
    r.num_active_host      = outs.num_active_host;
    r.num_solo_walks_host  = outs.num_solo_walks_host;
    r.num_warp_tasks_host  = outs.warp_smem.num_tasks_host;
    r.num_block_tasks_host = outs.block_smem.num_tasks_host;

    r.sorted_walk_idx = download_ints(
        outs.sorted_walk_idx, static_cast<std::size_t>(outs.num_active_host));
    r.solo_walks = download_ints(
        outs.solo_walks, static_cast<std::size_t>(outs.num_solo_walks_host));
    r.warp_nodes = download_ints(
        outs.warp_smem.nodes, static_cast<std::size_t>(r.num_warp_tasks_host));
    r.warp_walk_starts = download_ints(
        outs.warp_smem.walk_starts, static_cast<std::size_t>(r.num_warp_tasks_host));
    r.warp_walk_counts = download_ints(
        outs.warp_smem.walk_counts, static_cast<std::size_t>(r.num_warp_tasks_host));
    r.block_nodes = download_ints(
        outs.block_smem.nodes, static_cast<std::size_t>(r.num_block_tasks_host));
    r.block_walk_starts = download_ints(
        outs.block_smem.walk_starts, static_cast<std::size_t>(r.num_block_tasks_host));
    r.block_walk_counts = download_ints(
        outs.block_smem.walk_counts, static_cast<std::size_t>(r.num_block_tasks_host));

    return r;
}

std::set<int> collect_covered_walks(const SchedulerResult& r,
                                    std::string* duplicate_detail) {
    std::set<int> covered;
    auto insert_unique = [&](const int w, const char* tier) {
        if (!covered.insert(w).second) {
            if (duplicate_detail)
                *duplicate_detail = std::string("walk ") + std::to_string(w)
                    + " appears in " + tier + " and another tier";
        }
    };
    for (int w : r.solo_walks) insert_unique(w, "solo");
    for (std::size_t t = 0; t < r.warp_walk_counts.size(); ++t) {
        const int start = r.warp_walk_starts[t];
        const int count = r.warp_walk_counts[t];
        for (int k = 0; k < count; ++k) {
            insert_unique(r.sorted_walk_idx[start + k], "warp");
        }
    }
    for (std::size_t t = 0; t < r.block_walk_counts.size(); ++t) {
        const int start = r.block_walk_starts[t];
        const int count = r.block_walk_counts[t];
        for (int k = 0; k < count; ++k) {
            insert_unique(r.sorted_walk_idx[start + k], "block");
        }
    }
    return covered;
}

constexpr int PAD = EMPTY_NODE_VALUE;

}  // namespace

class GpuSchedulingTest : public ::testing::Test {
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

    static constexpr int STEP_NUMBER   = 1;
    static constexpr int MAX_WALK_LEN  = 4;
};

TEST_F(GpuSchedulingTest, AllWalksTerminated_AllTierCountsZero) {
    std::vector<int> last_nodes(10, PAD);
    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, 0);
    EXPECT_EQ(r.num_solo_walks_host, 0);
    EXPECT_EQ(r.num_warp_tasks_host, 0);
    EXPECT_EQ(r.num_block_tasks_host, 0);
}

TEST_F(GpuSchedulingTest, SingleActiveWalk_IsSolo) {
    std::vector<int> last_nodes(5, PAD);
    last_nodes[2] = 42;
    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, 1);
    EXPECT_EQ(r.num_solo_walks_host, 1);
    EXPECT_EQ(r.num_warp_tasks_host, 0);
    EXPECT_EQ(r.num_block_tasks_host, 0);
    ASSERT_EQ(r.solo_walks.size(), 1u);
    EXPECT_EQ(r.solo_walks[0], 2);
}

TEST_F(GpuSchedulingTest, AllUniqueNodes_AllWalksGoToSolo) {
    std::vector<int> last_nodes;
    for (int i = 0; i < 10; ++i) last_nodes.push_back(100 + i);
    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, 10);
    EXPECT_EQ(r.num_solo_walks_host, 10);
    EXPECT_EQ(r.num_warp_tasks_host, 0);
    EXPECT_EQ(r.num_block_tasks_host, 0);

    std::set<int> solo_set(r.solo_walks.begin(), r.solo_walks.end());
    EXPECT_EQ(solo_set.size(), 10u);
    for (int w : solo_set) {
        EXPECT_GE(w, 0);
        EXPECT_LT(w, 10);
    }
}

TEST_F(GpuSchedulingTest, AllSameNodeWarpSize_SingleWarpTask) {
    constexpr int W = 10;
    std::vector<int> last_nodes(W, 42);
    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, W);
    EXPECT_EQ(r.num_solo_walks_host, 0);
    ASSERT_EQ(r.num_warp_tasks_host, 1);
    EXPECT_EQ(r.num_block_tasks_host, 0);

    EXPECT_EQ(r.warp_nodes[0], 42);
    EXPECT_EQ(r.warp_walk_starts[0], 0);
    EXPECT_EQ(r.warp_walk_counts[0], W);
}

TEST_F(GpuSchedulingTest, AllSameNodeBlockSize_SingleBlockTask) {
    const int W = static_cast<int>(BLOCK_DIM) + 1;
    std::vector<int> last_nodes(W, 7);
    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, W);
    EXPECT_EQ(r.num_solo_walks_host, 0);
    EXPECT_EQ(r.num_warp_tasks_host, 0);
    ASSERT_EQ(r.num_block_tasks_host, 1);

    EXPECT_EQ(r.block_nodes[0], 7);
    EXPECT_EQ(r.block_walk_starts[0], 0);
    EXPECT_EQ(r.block_walk_counts[0], W);
}

TEST_F(GpuSchedulingTest, BoundaryW1IsSolo_W2IsWarp) {
    std::vector<int> last_nodes;
    last_nodes.push_back(100);
    last_nodes.push_back(200);
    last_nodes.push_back(200);

    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, 3);
    EXPECT_EQ(r.num_solo_walks_host, 1);
    EXPECT_EQ(r.num_warp_tasks_host, 1);
    EXPECT_EQ(r.num_block_tasks_host, 0);

    EXPECT_EQ(r.solo_walks[0], 0);
    EXPECT_EQ(r.warp_nodes[0], 200);
    EXPECT_EQ(r.warp_walk_counts[0], 2);
}

TEST_F(GpuSchedulingTest, BoundaryW_BLOCK_DIM_IsWarp_AndW_BLOCK_DIMPlus1_IsBlock) {
    // upper bound is inclusive: W=BLOCK_DIM stays warp, +1 flips to block.
    const int W_warp  = static_cast<int>(BLOCK_DIM);
    const int W_block = static_cast<int>(BLOCK_DIM) + 1;

    std::vector<int> last_nodes;
    for (int i = 0; i < W_warp;  ++i) last_nodes.push_back(10);
    for (int i = 0; i < W_block; ++i) last_nodes.push_back(20);

    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, W_warp + W_block);
    EXPECT_EQ(r.num_solo_walks_host, 0);
    ASSERT_EQ(r.num_warp_tasks_host, 1);
    ASSERT_EQ(r.num_block_tasks_host, 1);

    EXPECT_EQ(r.warp_walk_counts[0], W_warp);
    EXPECT_EQ(r.block_walk_counts[0], W_block);
}

TEST_F(GpuSchedulingTest, MixedTiers_CountIdentityHolds) {
    std::vector<int> last_nodes;
    for (int i = 0; i < 5;   ++i) last_nodes.push_back(201 + i);
    for (int i = 0; i < 10;  ++i) last_nodes.push_back(42);
    for (int i = 0; i < 300; ++i) last_nodes.push_back(7);
    const int expected_active = 5 + 10 + 300;

    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, expected_active);
    EXPECT_EQ(r.num_solo_walks_host, 5);
    EXPECT_EQ(r.num_warp_tasks_host, 1);
    EXPECT_EQ(r.num_block_tasks_host, 1);

    int total = r.num_solo_walks_host;
    total += std::accumulate(r.warp_walk_counts.begin(),
                             r.warp_walk_counts.end(), 0);
    total += std::accumulate(r.block_walk_counts.begin(),
                             r.block_walk_counts.end(), 0);
    EXPECT_EQ(total, expected_active);
}

TEST_F(GpuSchedulingTest, MixedWithTermination_DisjointUnionEqualsActive) {
    std::vector<int> last_nodes(30);
    std::vector<int> alive_indices;
    for (int i = 0; i < 5;  ++i) { last_nodes[i] = 201 + i;   alive_indices.push_back(i); }
    for (int i = 5; i < 15; ++i) { last_nodes[i] = 42;        alive_indices.push_back(i); }
    for (int i = 15; i < 20; ++i) last_nodes[i] = PAD;
    for (int i = 20; i < 30; ++i) { last_nodes[i] = 99;       alive_indices.push_back(i); }

    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, 25);
    EXPECT_EQ(r.num_solo_walks_host, 5);
    EXPECT_EQ(r.num_warp_tasks_host, 2);
    EXPECT_EQ(r.num_block_tasks_host, 0);

    std::string dup_detail;
    auto covered = collect_covered_walks(r, &dup_detail);
    EXPECT_TRUE(dup_detail.empty()) << dup_detail;

    EXPECT_EQ(covered.size(), 25u);
    for (int i : alive_indices) {
        EXPECT_EQ(covered.count(i), 1u)
            << "alive walk " << i << " missing from partition output";
    }
    for (int i = 15; i < 20; ++i) {
        EXPECT_EQ(covered.count(i), 0u)
            << "terminated walk " << i << " leaked into partition output";
    }
}

TEST_F(GpuSchedulingTest, SoloTierPreservesOriginalWalkIdx) {
    std::vector<int> last_nodes(20, PAD);
    std::vector<int> expected_solo;
    const int alive_indices[] = {3, 5, 8, 11, 13, 17, 19};
    int node = 500;
    for (int i : alive_indices) {
        last_nodes[i] = node++;
        expected_solo.push_back(i);
    }

    auto r = run_and_materialize(last_nodes, STEP_NUMBER, MAX_WALK_LEN, PAD,
                                 block_dim_, stream_);

    EXPECT_EQ(r.num_active_host, static_cast<int>(expected_solo.size()));
    EXPECT_EQ(r.num_solo_walks_host, static_cast<int>(expected_solo.size()));

    std::set<int> got(r.solo_walks.begin(), r.solo_walks.end());
    std::set<int> expected(expected_solo.begin(), expected_solo.end());
    EXPECT_EQ(got, expected);
}

#endif  // HAS_CUDA
