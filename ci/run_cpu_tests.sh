#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_DIR="${BUILD_DIR:-build-ci-cpu}"
TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE:-/opt/vcpkg/scripts/buildsystems/vcpkg.cmake}"

echo "Using Python interpreter: ${PYTHON_BIN}"
echo "Using build directory: ${BUILD_DIR}"
echo "Using CMake toolchain file: ${TOOLCHAIN_FILE}"

# Configure
cmake -S . -B "${BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DHAS_CUDA=OFF \
    -DPython3_EXECUTABLE="$(command -v python)"

# Build
cmake --build "${BUILD_DIR}" --target test_tempest --parallel

# Run C++ tests
"${BUILD_DIR}/tempest/test/test_tempest"

# Install Python package (project-specific, not cached)
CMAKE_ARGS="-DHAS_CUDA=OFF" python -m pip install .

# Run Python tests
python -m pytest py_tests -v --maxfail=1
