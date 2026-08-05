#!/usr/bin/env bash
# Build a self-contained macOS wheel: no Homebrew needed on the user's machine.
#
# Homebrew LLVM (OpenMP) and TBB are BUILD-time requirements only. delocate
# copies libomp.dylib and libtbb into the wheel and rewrites the install names
# to @loader_path, so the shipped extension links nothing outside /usr/lib.
#
#   Usage: build_scripts/build_wheel_macos.sh
#
# The resulting platform tag is bounded by the OLDEST macOS that every vendored
# dylib supports. Homebrew bottles are built for the host OS, so a wheel built
# on macOS 15 only installs on macOS 15+. To support older macOS, build on an
# older machine/runner (bottles then match), or build libomp + TBB from source
# with MACOSX_DEPLOYMENT_TARGET set. See MACOS_PUBLISHING.md.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "ERROR: this script is macOS only. Use build_scripts/build_wheels.sh on Linux."
    exit 1
fi

# Build-time dependencies.
for dep in llvm tbb; do
    if ! brew --prefix "$dep" >/dev/null 2>&1; then
        echo "ERROR: missing build dependency. Run: brew install llvm tbb"
        exit 1
    fi
done

python -c "import delocate" 2>/dev/null || {
    echo "Installing delocate..."
    python -m pip install --quiet delocate
}

# The wheel's platform tag comes from the building Python's deployment target.
# Export it so delocate verifies the vendored dylibs actually support it, and
# fails loudly rather than shipping a wheel that dyld rejects at import.
: "${MACOSX_DEPLOYMENT_TARGET:=$(python -c \
    "import sysconfig; print(sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET'))")}"
export MACOSX_DEPLOYMENT_TARGET
echo "Targeting macOS ${MACOSX_DEPLOYMENT_TARGET}"

ARCH="$(uname -m)"
rm -rf build dist wheelhouse ./*.egg-info

# setup.py picks Homebrew LLVM automatically; CC/CXX still override.
python setup.py bdist_wheel

RAW_WHEEL="$(ls dist/*.whl)"
echo "Built: ${RAW_WHEEL}"

# Vendor the third-party dylibs and relink to @loader_path.
delocate-wheel -v --require-archs "${ARCH}" -w wheelhouse "${RAW_WHEEL}"

WHEEL="$(ls wheelhouse/*.whl)"

# A publishable wheel must reference nothing outside /usr/lib and its own
# bundled .dylibs directory. Anything else is a machine-specific path.
LEAKED="$(delocate-listdeps --all "${WHEEL}" | grep -v '^/usr/lib/' | grep -v '\.dylibs/' || true)"
if [[ -n "${LEAKED}" ]]; then
    echo "ERROR: wheel still references paths outside /usr/lib:"
    echo "${LEAKED}"
    exit 1
fi

echo
echo "Self-contained wheel ready: ${WHEEL}"
delocate-listdeps --all "${WHEEL}"
echo
echo "Upload with: twine upload ${WHEEL}"
