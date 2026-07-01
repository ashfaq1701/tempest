cmake_minimum_required(VERSION 3.22)
project(tempest)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CXX_STANDARD 17)

# Use directly provided Python paths if available
if(DEFINED PYTHON_EXECUTABLE AND DEFINED PYTHON_INCLUDE_DIR AND DEFINED PYTHON_LIBRARY)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
    set(Python3_INCLUDE_DIRS ${PYTHON_INCLUDE_DIR})
    set(Python3_LIBRARIES ${PYTHON_LIBRARY})
    message(STATUS "Using provided Python paths:")
    message(STATUS "  Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
    message(STATUS "  Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
    message(STATUS "  Python3_LIBRARIES: ${Python3_LIBRARIES}")
else()
    # Fall back to automatic detection
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
endif()

# Get pybind11 include path using the correct Python executable
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(f'{pybind11.get_include()}/../share/cmake/pybind11')"
        OUTPUT_VARIABLE PYBIND11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${PYBIND11_DIR}")

# Get Python version dynamically
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        OUTPUT_VARIABLE DETECTED_PYTHON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set pybind11 Python version to match detected version
set(PYBIND11_PYTHON_VERSION ${DETECTED_PYTHON_VERSION} CACHE STRING "Python version used for pybind11")
message(STATUS "Setting pybind11 Python version to: ${PYBIND11_PYTHON_VERSION}")

find_package(pybind11 REQUIRED)

find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)

# Ubuntu 24.04's TBBTargets.cmake puts /usr/include in INTERFACE_INCLUDE_
# DIRECTORIES, which CMake passes to nvcc as `-isystem /usr/include`.
# That breaks gcc-13's `#include_next <math.h>` in <cmath> and aborts the
# CUDA build. Strip the offending entry (build-enabler, zero perf effect).
if(TARGET TBB::tbb)
    get_target_property(_tbb_inc TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
    if(_tbb_inc)
        list(REMOVE_ITEM _tbb_inc "/usr/include")
        set_target_properties(TBB::tbb PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_tbb_inc}")
    endif()
endif()

option(HAS_CUDA "Enable CUDA support" ON)

if(HAS_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
        endif()
        add_definitions(-DHAS_CUDA)

        # NVTX3 is header-only; ships with the CUDA Toolkit. When found,
        # define HAS_NVTX so RAII range markers compile in. Otherwise the
        # NvtxRange class becomes a pure no-op.
        find_path(NVTX3_INCLUDE_DIR
            NAMES nvtx3/nvToolsExt.h
            PATHS ${CUDAToolkit_INCLUDE_DIRS} /usr/local/cuda/include
            NO_DEFAULT_PATH)
        if(NVTX3_INCLUDE_DIR)
            message(STATUS "Found NVTX3 at ${NVTX3_INCLUDE_DIR}")
            add_compile_definitions(HAS_NVTX=1)
            include_directories(${NVTX3_INCLUDE_DIR})
        else()
            message(STATUS "NVTX3 not found; profiling annotations disabled.")
        endif()
    else()
        set(HAS_CUDA OFF)
    endif()
endif()

if(APPLE)
    add_definitions(-DIS_MACOS)
endif()

set(SRC_FILES
        ${CMAKE_SOURCE_DIR}/tempest/src/data/buffer.cu
        ${CMAKE_SOURCE_DIR}/tempest/src/data/walk_set/walk_set_device.cu
        ${CMAKE_SOURCE_DIR}/tempest/src/data/walk_set/walk_set_host.cu

        ${CMAKE_SOURCE_DIR}/tempest/src/graph/edge_data.cu
        ${CMAKE_SOURCE_DIR}/tempest/src/graph/node_edge_index.cu
        ${CMAKE_SOURCE_DIR}/tempest/src/graph/node_features.cu
        ${CMAKE_SOURCE_DIR}/tempest/src/graph/temporal_graph.cu
        ${CMAKE_SOURCE_DIR}/tempest/src/core/tempest.cu
        ${CMAKE_SOURCE_DIR}/tempest/src/core/node_grouped/scheduler.cu

        ${CMAKE_SOURCE_DIR}/tempest/src/proxies/RandomPicker.cu
        ${CMAKE_SOURCE_DIR}/tempest/src/proxies/Tempest.cu
)

if(HAS_CUDA AND CMAKE_CUDA_COMPILER)
    add_compile_options(
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    )

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
            --ptxas-options=-v         # Verbose PTX assembly
            -Xptxas -dlcm=ca          # Cache all operations
            -Xcompiler=-fopenmp
    )

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")

    # For debug builds
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
else()
    # If CUDA is not enabled, ensure OpenMP is still activated for C++
    add_compile_options(-fopenmp)
endif()

option(BUILD_TESTS "Build C++ tests" ON)

add_subdirectory(tempest/py_interface)

if(BUILD_TESTS)
    add_subdirectory(tempest/test)
    add_subdirectory(tempest/test_run)
endif()
