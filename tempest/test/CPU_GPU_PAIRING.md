# CPU/GPU Test Pairing — Required

Every test in this directory **must** exercise both the CPU and GPU code
path. Tests that only run on one backend are forbidden.

## Why

The codebase has two backends for almost every operation: a `_std` /
host-side variant and a `_cuda` / device-side variant, dispatched at
runtime on `data.use_gpu`. A test that only runs on one backend leaves
the other path completely unchecked — bugs caught by CPU tests slip
through to GPU and vice versa. We've already hit this: the HOST-safe
dispatch inside `node_edge_index::get_timestamp_group_range` was missing
until a GPU-only run segfaulted, because the CPU test was green.

## How

Every test suite in this directory is written as a `TYPED_TEST`
parameterized on a `std::integral_constant<bool, use_gpu>` type. The
conventional alias is `GPU_USAGE_TYPES`:

```cpp
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

TYPED_TEST_SUITE(MyTest, GPU_USAGE_TYPES);

TYPED_TEST(MyTest, Something) {
    core::Tempest trw(/*is_directed=*/true, /*use_gpu=*/TypeParam::value);
    // ... test body uses trw.data() ...
}
```

The fixture constructs a `core::Tempest` (or underlying
`TemporalGraphData`) with the parameterized `use_gpu`. Runtime dispatch
inside the code-under-test (e.g. `edge_data::update_timestamp_groups_*`,
`temporal_graph::add_multiple_edges_*`) then exercises both backends.

When a gtest run emits two result lines per logical test — `.../0` for
CPU and `.../1` for GPU — the pairing is correct. One line only means
the test is unpaired.

## What this rule covers

- Anything holding / mutating a `TemporalGraphData`.
- Anything going through `temporal_graph::` / `edge_data::` /
  `node_edge_index::` free functions.
- Walk sampling through `core::Tempest` or the
  `Tempest` proxy.
- Random pickers (already `TYPED_TEST` on GPU_USAGE_TYPES).

## Exceptions

There are none. Even features with host-only storage (e.g.
`TemporalGraphData::node_features`, which is always a host `Buffer<float>`
regardless of `use_gpu`) still parameterize the fixture — the GPU variant
exercises the host-only feature path through a GPU-backed graph, which
matches how real users will touch it.

## Review rule

If a PR adds a `TEST(...)` in this directory instead of a `TYPED_TEST`,
reject it. If a PR adds a TYPED_TEST suite whose type list excludes one
backend (e.g. only `bool, false`), reject it. If a PR forces a specific
case to CPU-only to sidestep a GPU crash, reject it and fix the GPU
path instead — the crash is the point.
