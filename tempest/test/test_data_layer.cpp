#include <gtest/gtest.h>
#include <vector>
#include <utility>
#include <cstdlib>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

#include "../src/data/buffer.cuh"
#include "../src/data/device_arena.cuh"
#include "../src/data/temporal_graph_data.cuh"
#include "../src/data/temporal_graph_view.cuh"
#include "../src/data/walk_set/walk_set_device.cuh"
#include "../src/data/walk_set/walk_set_host.cuh"
#include "../src/data/walk_set/walk_set_view.cuh"

#ifdef HAS_CUDA
using BACKEND_TYPES = ::testing::Types<
    std::integral_constant<bool, false>,
    std::integral_constant<bool, true>
>;
#else
using BACKEND_TYPES = ::testing::Types<
    std::integral_constant<bool, false>
>;
#endif

template <typename T>
class BufferTest : public ::testing::Test {};
TYPED_TEST_SUITE(BufferTest, BACKEND_TYPES);

TYPED_TEST(BufferTest, DefaultConstructedIsEmpty) {
    Buffer<int> b(TypeParam::value);
    EXPECT_EQ(b.size(), 0u);
    EXPECT_EQ(b.capacity(), 0u);
    EXPECT_EQ(b.data(), nullptr);
    EXPECT_TRUE(b.empty());
    EXPECT_EQ(b.is_gpu(), TypeParam::value);
}

TYPED_TEST(BufferTest, ResizeAndFill) {
    Buffer<int> b(TypeParam::value);
    b.resize(10);
    b.fill(42);
    EXPECT_EQ(b.size(), 10u);
    auto v = b.to_host_vector();
    ASSERT_EQ(v.size(), 10u);
    for (int x : v) EXPECT_EQ(x, 42);
}

TYPED_TEST(BufferTest, AppendFromHost) {
    Buffer<int> b(TypeParam::value);
    const std::vector<int> src{1, 2, 3, 4, 5};
    b.append_from_host(src.data(), src.size());
    EXPECT_EQ(b.size(), 5u);
    auto v = b.to_host_vector();
    EXPECT_EQ(v, src);
}

TYPED_TEST(BufferTest, DropFront) {
    Buffer<int> b(TypeParam::value);
    const std::vector<int> src{1, 2, 3, 4, 5};
    b.append_from_host(src.data(), src.size());
    b.drop_front(2);
    EXPECT_EQ(b.size(), 3u);
    auto v = b.to_host_vector();
    const std::vector<int> expected{3, 4, 5};
    EXPECT_EQ(v, expected);
}

TYPED_TEST(BufferTest, MoveOnlySemantics) {
    Buffer<int> a(TypeParam::value);
    a.resize(5);
    a.fill(7);
    Buffer<int> b = std::move(a);
    EXPECT_EQ(a.size(), 0u);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(b.size(), 5u);
    auto v = b.to_host_vector();
    for (int x : v) EXPECT_EQ(x, 7);
}

template <typename T>
class DeviceArenaTest : public ::testing::Test {};
TYPED_TEST_SUITE(DeviceArenaTest, BACKEND_TYPES);

TYPED_TEST(DeviceArenaTest, AcquireAndReset) {
    DeviceArena arena(TypeParam::value, 4096);
    int* a = arena.acquire<int>(100);
    int64_t* b = arena.acquire<int64_t>(50);
    EXPECT_NE(a, nullptr);
    EXPECT_NE(b, nullptr);
    EXPECT_NE(static_cast<void*>(a), static_cast<void*>(b));
    EXPECT_GT(arena.used_bytes(), 0u);
    arena.reset();
    EXPECT_EQ(arena.used_bytes(), 0u);
}

template <typename T>
class TemporalGraphDataTest : public ::testing::Test {};
TYPED_TEST_SUITE(TemporalGraphDataTest, BACKEND_TYPES);

TYPED_TEST(TemporalGraphDataTest, DefaultConstructsEmpty) {
    TemporalGraphData data(TypeParam::value);
    EXPECT_EQ(data.use_gpu, TypeParam::value);
    EXPECT_EQ(data.sources.size(), 0u);
    EXPECT_EQ(data.targets.size(), 0u);
    EXPECT_EQ(data.timestamps.size(), 0u);
    EXPECT_EQ(data.max_node_id, -1);
    EXPECT_FALSE(data.is_directed);
    EXPECT_EQ(data.feature_dim, 0u);
}

TYPED_TEST(TemporalGraphDataTest, MoveOnly) {
    TemporalGraphData a(TypeParam::value);
    a.latest_timestamp = 42;
    a.sources.resize(3);
    a.sources.fill(7);

    TemporalGraphData b = std::move(a);
    EXPECT_EQ(b.latest_timestamp, 42);
    EXPECT_EQ(b.sources.size(), 3u);
    auto v = b.sources.to_host_vector();
    for (int x : v) EXPECT_EQ(x, 7);
    EXPECT_EQ(a.sources.size(), 0u);
}

TYPED_TEST(TemporalGraphDataTest, ViewAliasesDataFields) {
    TemporalGraphData data(TypeParam::value);
    data.latest_timestamp = 123;
    data.is_directed = true;
    data.inv_p = 0.5;
    data.inv_q = 2.0;
    data.sources.resize(4);
    data.sources.fill(9);
    data.timestamps.resize(4);

    TemporalGraphView view = make_temporal_graph_view(data);
    EXPECT_EQ(view.latest_timestamp, 123);
    EXPECT_TRUE(view.is_directed);
    EXPECT_EQ(view.inv_p, 0.5);
    EXPECT_EQ(view.inv_q, 2.0);
    EXPECT_EQ(view.num_edges, 4u);
    EXPECT_EQ(view.sources, data.sources.data());
    EXPECT_EQ(view.targets, data.targets.data());
    EXPECT_EQ(view.timestamps, data.timestamps.data());
}

template <typename T>
class WalkSetRoundTripTest : public ::testing::Test {};
TYPED_TEST_SUITE(WalkSetRoundTripTest, BACKEND_TYPES);

TYPED_TEST(WalkSetRoundTripTest, PreservesInitialPadding) {
    constexpr bool use_gpu = TypeParam::value;
    const size_t num_walks = 3;
    const size_t max_len   = 5;
    const int padding      = -1;

    WalkSetHost h;
    if (use_gpu) {
#ifdef HAS_CUDA
        WalkSetDevice d(num_walks, max_len, padding);
        h = std::move(d).download_to_host();
#else
        GTEST_SKIP() << "HAS_CUDA not defined; use_gpu=true is unreachable.";
#endif
    } else {
        h = WalkSetHost(num_walks, max_len, padding);
    }

    EXPECT_EQ(h.num_walks(),     num_walks);
    EXPECT_EQ(h.max_len(),       max_len);
    EXPECT_EQ(h.padding_value(), padding);

    ASSERT_NE(h.nodes_ptr(), nullptr);
    for (size_t i = 0; i < num_walks * max_len; ++i) {
        EXPECT_EQ(h.nodes_ptr()[i], padding);
    }
    for (size_t i = 0; i < num_walks; ++i) {
        EXPECT_EQ(h.walk_lens_ptr()[i], 0u);
    }
    EXPECT_EQ(h.non_empty_count(), 0u);
}

template <typename T>
class WalkSetReleaseTest : public ::testing::Test {};
TYPED_TEST_SUITE(WalkSetReleaseTest, BACKEND_TYPES);

TYPED_TEST(WalkSetReleaseTest, ReleaseTransfersOwnership) {
    constexpr bool use_gpu = TypeParam::value;
    const size_t num_walks = 2;
    const size_t max_len   = 3;
    const int padding      = -1;

    WalkSetHost h;
    if (use_gpu) {
#ifdef HAS_CUDA
        WalkSetDevice d(num_walks, max_len, padding);
        h = std::move(d).download_to_host();
#else
        GTEST_SKIP() << "HAS_CUDA not defined; use_gpu=true is unreachable.";
#endif
    } else {
        h = WalkSetHost(num_walks, max_len, padding);
    }

    HostRelease rn = h.release_nodes();
    int* n_raw = static_cast<int*>(rn.ptr);
    ASSERT_NE(n_raw, nullptr);
    for (size_t i = 0; i < num_walks * max_len; ++i) EXPECT_EQ(n_raw[i], padding);
    EXPECT_EQ(h.nodes_ptr(), nullptr);

    std::free(n_raw);
}
