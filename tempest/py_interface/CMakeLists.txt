pybind11_add_module(_tempest
        _tempest.cu
        ${SRC_FILES}
)

set(MODULE_SRC_FILES ${SRC_FILES} _tempest.cu)

target_compile_features(_tempest PRIVATE cxx_std_17)
set_target_properties(_tempest PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(_tempest PRIVATE OpenMP::OpenMP_CXX TBB::tbb)

if(HAS_CUDA AND CMAKE_CUDA_COMPILER)
    set_source_files_properties(
            ${MODULE_SRC_FILES}
            PROPERTIES
            LANGUAGE CUDA
    )

    set_property(TARGET _tempest PROPERTY
            CUDA_HOST_COMPILER_OPTIONS "${OpenMP_CXX_FLAGS}")

    target_link_libraries(_tempest PRIVATE
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::curand
    )

    set_target_properties(_tempest PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            CUDA_VISIBILITY_PRESET "hidden"
            CXX_VISIBILITY_PRESET "hidden"
            VISIBILITY_INLINES_HIDDEN ON
    )

    target_include_directories(_tempest PRIVATE
            ${CMAKE_SOURCE_DIR}/tempest/src
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
else ()
    set_source_files_properties(
            ${MODULE_SRC_FILES}
            PROPERTIES
            LANGUAGE CXX
    )

    target_include_directories(_tempest PRIVATE
            ${CMAKE_SOURCE_DIR}/tempest/src
    )
endif()
