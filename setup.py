from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys
import pybind11


def find_cmake():
    try:
        subprocess.check_output(['cmake', '--version'])
    except subprocess.CalledProcessError:
        raise RuntimeError("CMake must be installed to build the following extensions: _tempest")


DEFAULT_COMPILERS = ('/usr/bin/gcc', '/usr/bin/g++')

# ---------------------------------------------------------------------------
# macOS build support
#
# Two host quirks need patching up on macOS, and nowhere else:
#
#   1. Apple's /usr/bin/{gcc,g++} are clang shims with no OpenMP support, so
#      CMake's `find_package(OpenMP REQUIRED)` fails outright. Homebrew LLVM
#      ships a working one.
#   2. The pip-installed cmake does not search Homebrew prefixes, so
#      `find_package(TBB REQUIRED)` fails even after `brew install tbb`.
#
# Homebrew is a build-time requirement only - the published wheel vendors the
# resulting dylibs. See build_scripts/build_wheel_macos.sh and
# docs/MACOS_PUBLISHING.md.
# ---------------------------------------------------------------------------

IS_MACOS = sys.platform == 'darwin'

# Compiler pairs to probe, most preferred first. Homebrew LLVM covers both the
# Apple Silicon (/opt/homebrew) and Intel (/usr/local) layouts; Homebrew GCC is
# a last resort.
_MACOS_COMPILER_CANDIDATES = (
    ('/opt/homebrew/opt/llvm/bin/clang', '/opt/homebrew/opt/llvm/bin/clang++'),
    ('/usr/local/opt/llvm/bin/clang', '/usr/local/opt/llvm/bin/clang++'),
    ('/opt/homebrew/bin/gcc-14', '/opt/homebrew/bin/g++-14'),
)

# Homebrew prefixes to probe, Apple Silicon first. TBB is the marker, being the
# dependency CMake cannot locate unaided.
_MACOS_BREW_PREFIXES = ('/opt/homebrew', '/usr/local')


def _macos_openmp_compilers():
    """First installed pair from _MACOS_COMPILER_CANDIDATES, i.e. one with OpenMP."""
    for cc, cxx in _MACOS_COMPILER_CANDIDATES:
        if os.path.exists(cc) and os.path.exists(cxx):
            return cc, cxx

    raise RuntimeError(
        "No OpenMP-capable compiler found. Apple's /usr/bin/g++ cannot build this "
        "project. Install one with `brew install llvm tbb`, or set CC/CXX explicitly."
    )


def _macos_homebrew_prefix():
    """Homebrew prefix that has TBB installed, or None if none of them do."""
    for prefix in _MACOS_BREW_PREFIXES:
        if os.path.isdir(os.path.join(prefix, 'opt', 'tbb')):
            return prefix
    return None


# ---------------------------------------------------------------------------
# Platform-neutral entry points, used by CMakeBuild below
# ---------------------------------------------------------------------------


def resolve_compilers():
    """The (C, C++) compiler pair to build with.

    CC/CXX from the environment always win - the Linux wheel build and the macOS
    ASAN script both set them explicitly. A platform default is only probed for
    when one of the two is missing.
    """
    cc = os.environ.get('CC')
    cxx = os.environ.get('CXX')
    if cc and cxx:
        return cc, cxx

    default_cc, default_cxx = (
        _macos_openmp_compilers() if IS_MACOS else DEFAULT_COMPILERS
    )
    return cc or default_cc, cxx or default_cxx


def platform_cmake_args():
    """Extra CMake args the host platform needs. Empty everywhere but macOS."""
    if not IS_MACOS:
        return []

    # An explicit CMAKE_PREFIX_PATH always wins - never clobber the caller's
    # prefix list, which on Linux points CMake at vcpkg and CUDA.
    prefix = _macos_homebrew_prefix()
    if prefix and 'CMAKE_PREFIX_PATH' not in os.environ:
        return [f'-DCMAKE_PREFIX_PATH={prefix}']
    return []


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        find_cmake()
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Check for debug mode via environment variable
        is_debug = os.environ.get('DEBUG_BUILD', '0').lower() in ('1', 'true', 'yes')
        build_type = 'Debug' if is_debug else 'Release'

        print(f"Building in {build_type} mode")

        # Initialize cmake args with the standard options
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DCMAKE_BUILD_TYPE={build_type}',
            '-DBUILD_TESTS=OFF',
        ]

        # Find compilers - environment variables win over the platform default
        cc, cxx = resolve_compilers()

        # Add compiler paths to cmake args
        cmake_args.append(f'-DCMAKE_C_COMPILER={cc}')
        cmake_args.append(f'-DCMAKE_CXX_COMPILER={cxx}')

        # Anything else this host needs (Homebrew prefix on macOS, nothing on Linux)
        cmake_args += platform_cmake_args()

        # Use environment-defined Python paths if available
        python_executable = os.environ.get('PYTHON_EXECUTABLE', sys.executable)
        python_include_dir = os.environ.get('PYTHON_INCLUDE_DIR', '')
        python_library = os.environ.get('PYTHON_LIBRARY', '')

        # Add Python paths to cmake args
        cmake_args.append(f'-DPYTHON_EXECUTABLE={python_executable}')

        if python_include_dir:
            cmake_args.append(f'-DPYTHON_INCLUDE_DIR={python_include_dir}')

        if python_library:
            cmake_args.append(f'-DPYTHON_LIBRARY={python_library}')

        # Add pybind11 include directory
        cmake_args.append(f'-DPYBIND11_INCLUDE_DIR={pybind11.get_include()}')

        # Debug output
        print(f"Building with compilers: CC={cc}, CXX={cxx}")
        print(f"Building with Python: {python_executable}")
        if python_include_dir:
            print(f"Python include dir: {python_include_dir}")
        if python_library:
            print(f"Python library: {python_library}")
        print(f"CMake arguments: {cmake_args}")

        # Inject extra CMake args from environment
        extra_cmake_args = os.environ.get("CMAKE_ARGS")
        if extra_cmake_args:
            print(f"Using extra CMake args from env: {extra_cmake_args}")
            cmake_args += extra_cmake_args.split()

        # Set up compilation flags based on build type
        if is_debug:
            # Debug flags for C++
            base_cxxflags = "-D_GLIBCXX_USE_CXX11_ABI=0 -g -O0 -fno-omit-frame-pointer"
            # CUDA debug flags
            base_cudaflags = "-g -G -O0"
            print("Using debug compilation flags")
        else:
            # Release flags
            base_cxxflags = "-D_GLIBCXX_USE_CXX11_ABI=0"
            base_cudaflags = ""
            print("Using release compilation flags")

        # Merge with existing flags if any
        if "CXXFLAGS" in os.environ:
            os.environ["CXXFLAGS"] += " " + base_cxxflags
        else:
            os.environ["CXXFLAGS"] = base_cxxflags

        if "CUDAFLAGS" in os.environ:
            os.environ["CUDAFLAGS"] += " " + base_cudaflags
        else:
            os.environ["CUDAFLAGS"] = base_cudaflags

        # NVCC flags need to be passed through CMake
        if is_debug:
            cmake_args.append('-DCMAKE_CUDA_FLAGS=-g -G -O0')

        build_args = ['--config', build_type, '--target', ext.name]
        os.makedirs(self.build_temp, exist_ok=True)

        try:
            # Pass environment variables to CMAKE
            env = os.environ.copy()
            print(f"Environment variables during build:")
            print(f"  CXXFLAGS: {env.get('CXXFLAGS', 'not set')}")
            print(f"  CUDAFLAGS: {env.get('CUDAFLAGS', 'not set')}")

            subprocess.check_call(['cmake', os.path.abspath('.')] + cmake_args, cwd=self.build_temp, env=env)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error during CMake configuration or build: {e}")
            sys.exit(1)

def read_version_number():
    with open('version_number.txt', 'r') as file:
        version_number = file.readline()
    return version_number.strip()


setup(
    name="tempest-rw",
    version=read_version_number(),
    author="Ashfaq Salehin",
    author_email="ashfaq.salehin1701@gmail.com",
    description="A library to sample temporal random walks from in-memory temporal graphs",
    long_description=open('README.md').read(),
    packages=find_packages(),
    package_data={
        'tempest': ['*.so'],
    },
    include_package_data=True,
    long_description_content_type="text/markdown",
    url="https://github.com/ashfaq1701/tempest",
    ext_modules=[CMakeExtension('_tempest')],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=[
        "pybind11>=2.6.0",
        "numpy",
        "networkx",
        # CUDA runtime libs the Linux PyPI wheel needs at runtime.
        "nvidia-cuda-runtime-cu12>=12.6; platform_system == 'Linux'",
        "nvidia-curand-cu12>=10.3;        platform_system == 'Linux'",
    ],
)
