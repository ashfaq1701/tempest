find_package(GTest CONFIG REQUIRED)

set(TEST_FILES
        test_edge_data.cpp
        test_edge_data_weights.cpp
        test_node_edge_index.cpp
        test_node_edge_index_weights.cpp
        test_node_features.cpp
        test_random_picker.cpp
        test_temporal_graph.cpp
        test_temporal_graph_get_node_edge_at.cpp
        test_temporal_graph_weights.cpp
        test_tempest.cpp
        test_tempest_edge_features.cpp
        test_walk_padding_value.cpp
        test_weight_based_random_picker.cpp
        test_temporal_node2vec.cpp
        test_temporal_node2vec_rejection.cpp
        test_data_layer.cpp
        test_node_grouped_parity.cpp
        test_node_grouped_w_partition.cpp
        test_node_grouped_g_partition.cpp
        test_node_grouped_block_task_expansion.cpp
        test_node_grouped_walk_tier_routing.cpp
        test_walk_cutoff_times.cpp
)

set(MODULE_SRC_FILES ${SRC_FILES} ${TEST_FILES})

if(HAS_CUDA AND CMAKE_CUDA_COMPILER)
    set_source_files_properties(
            ${MODULE_SRC_FILES}
            PROPERTIES
            LANGUAGE CUDA
    )

    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    set(CUDA_LIBS
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::cublas
            CUDA::nvrtc
            CUDA::curand
            cuda
            cudart
    )

    add_executable(test_tempest ${TEST_FILES} ${SRC_FILES})
    # RDC is deliberately OFF. CUDA_SEPARABLE_COMPILATION ON passes -rdc=true,
    # and combined with a multi-arch (or low-arch PTX-JIT) fatbin + CUDA 12.x
    # thrust/CUB it makes the runtime dispatch of CUB primitives (e.g.
    # thrust::copy_if via cub::DeviceSelect) launch a policy kernel not emitted
    # for the running arch, failing with cudaLaunchKernel: cudaErrorInvalidValue
    # ("copy_if failed on 2nd step: invalid argument"). Nothing here needs RDC
    # (no cross-TU __device__ calls, no dynamic parallelism — device helpers are
    # __forceinline__ in .cuh headers). Mirrors the py_interface fix (1d8404c);
    # CUDA_RESOLVE_DEVICE_SYMBOLS is only meaningful with RDC on, so it is dropped.
    set_target_properties(test_tempest PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
    )

    target_include_directories(test_tempest PRIVATE
            ${CMAKE_SOURCE_DIR}/tempest/src
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
else()
    set_source_files_properties(
            ${MODULE_SRC_FILES}
            PROPERTIES
            LANGUAGE CXX
    )

    set(CUDA_OPTIONS "")
    set(CUDA_LIBS "")

    add_executable(test_tempest ${TEST_FILES} ${SRC_FILES})

    target_include_directories(test_tempest PRIVATE
            ${CMAKE_SOURCE_DIR}/tempest/src
    )
endif()

target_compile_features(test_tempest PRIVATE cxx_std_17)
target_compile_options(test_tempest PRIVATE ${CUDA_OPTIONS})

target_link_libraries(test_tempest PRIVATE
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        GTest::gmock_main
        OpenMP::OpenMP_CXX
        TBB::tbb
        ${CUDA_LIBS}
)
