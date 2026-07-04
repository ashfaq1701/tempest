
function(add_test_executable target_name source_file)
    add_executable(${target_name} ${source_file} ${SRC_FILES})
    set(MOD_SRC_FILES ${SRC_FILES} ${source_file})

    target_compile_features(${target_name} PRIVATE cxx_std_17)

    target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX TBB::tbb)

    if(HAS_CUDA AND CMAKE_CUDA_COMPILER)
        set_source_files_properties(
                ${MOD_SRC_FILES}
                PROPERTIES
                LANGUAGE CUDA
        )

        # RDC OFF: -rdc=true + multi-arch/PTX-JIT fatbin + CUDA 12.x thrust/CUB
        # makes CUB dispatch launch a wrong-arch policy kernel and fail with
        # cudaErrorInvalidValue ("copy_if failed on 2nd step"). Nothing here
        # needs RDC. Mirrors the py_interface fix (1d8404c).
        set_target_properties(${target_name} PROPERTIES
                CUDA_SEPARABLE_COMPILATION OFF
        )

        set_property(TARGET ${target_name} PROPERTY
                CUDA_HOST_COMPILER_OPTIONS "${OpenMP_CXX_FLAGS}")

        target_link_libraries(${target_name} PRIVATE
                OpenMP::OpenMP_CXX
                curand
                CUDA::cudart
                CUDA::cuda_driver
                cuda
                cudart
                cudadevrt
        )

        target_include_directories(${target_name} PRIVATE
                ${CMAKE_SOURCE_DIR}/tempest/src
                ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )
    else()
        set_source_files_properties(
                ${MOD_SRC_FILES}
                PROPERTIES
                LANGUAGE CXX
        )

        target_include_directories(${target_name} PRIVATE
                ${CMAKE_SOURCE_DIR}/tempest/src
        )
    endif()

    set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endfunction()

add_test_executable(test_run_tempest test_run_tempest.cpp)
add_test_executable(walk_sampling_speed_test walk_sampling_speed_test.cpp)
add_test_executable(ablation_streaming ablation_streaming.cpp)
add_test_executable(test_alibaba_streaming test_alibaba_streaming.cpp)
add_test_executable(test_run_exponential_random_picker test_run_exponential_picker.cpp)

# ---------------------------------------------------------------------
# test_alibaba_streaming: optional parquet reader.
# The parquet impl lives in test_utils_parquet.cpp, compiled as plain CXX
# (Arrow headers are not nvcc-friendly, and TRW's thrust-templated .cuh
# headers aren't g++-friendly, so the file is a pure Arrow TU).
# Arrow is located through the pyarrow Python package; no system-level
# libarrow-dev install needed. Without pyarrow, the TU compiles as a
# stub that throws at runtime on parquet inputs — CSV still works.
# ---------------------------------------------------------------------
target_sources(test_alibaba_streaming PRIVATE test_utils_parquet.cpp)
set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/test_utils_parquet.cpp"
        PROPERTIES LANGUAGE CXX)

set(PYARROW_HOME "" CACHE PATH "Path to a pyarrow install dir. Leave empty to auto-detect via `import pyarrow`.")
if(NOT PYARROW_HOME)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
            "import pyarrow, os; print(os.path.dirname(pyarrow.__file__))"
        OUTPUT_VARIABLE _pyarrow_auto
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pyarrow_rc
        ERROR_QUIET)
    if(_pyarrow_rc EQUAL 0 AND _pyarrow_auto)
        set(PYARROW_HOME "${_pyarrow_auto}")
    endif()
endif()

if(PYARROW_HOME
        AND EXISTS "${PYARROW_HOME}/include/parquet/arrow/reader.h"
        AND EXISTS "${PYARROW_HOME}/include/arrow/api.h")
    message(STATUS "test_alibaba_streaming: Arrow/Parquet from ${PYARROW_HOME}")

    # Arrow 24 headers use C++20 (std::span etc.); the CUDA-side TUs of
    # this target still compile under CUDA_STANDARD=17.
    target_compile_features(test_alibaba_streaming PRIVATE cxx_std_20)

    # The parquet TU: Arrow include path, new libstdc++ ABI (pyarrow was
    # built with _GLIBCXX_USE_CXX11_ABI=1; the rest of the repo forces =0).
    # Interface types in test_utils.h (const char*, vector<tuple<int...>>)
    # are ABI-stable, so the two-ABI TU mix is safe at the link boundary.
    set_source_files_properties(
            "${CMAKE_CURRENT_SOURCE_DIR}/test_utils_parquet.cpp"
            PROPERTIES
                LANGUAGE CXX
                INCLUDE_DIRECTORIES "${PYARROW_HOME}/include"
                COMPILE_DEFINITIONS "TRW_HAS_PARQUET"
                COMPILE_OPTIONS "-U_GLIBCXX_USE_CXX11_ABI;-D_GLIBCXX_USE_CXX11_ABI=1")

    # pyarrow ships libarrow.so.NNNN with no unversioned symlink; link by full path.
    file(GLOB _trw_arrow_lib   "${PYARROW_HOME}/libarrow.so.*")
    file(GLOB _trw_parquet_lib "${PYARROW_HOME}/libparquet.so.*")
    list(GET _trw_arrow_lib   0 _trw_arrow_lib_first)
    list(GET _trw_parquet_lib 0 _trw_parquet_lib_first)
    if(NOT _trw_arrow_lib_first OR NOT _trw_parquet_lib_first)
        message(FATAL_ERROR
            "test_alibaba_streaming: libarrow.so.* or libparquet.so.* "
            "missing from ${PYARROW_HOME}")
    endif()
    target_link_libraries(test_alibaba_streaming PRIVATE
            "${_trw_arrow_lib_first}"
            "${_trw_parquet_lib_first}")

    set_target_properties(test_alibaba_streaming PROPERTIES
            BUILD_RPATH   "${PYARROW_HOME}"
            INSTALL_RPATH "${PYARROW_HOME}")
else()
    message(STATUS "test_alibaba_streaming: pyarrow not found; "
                   "parquet inputs will throw at runtime (CSV still works).")
endif()
