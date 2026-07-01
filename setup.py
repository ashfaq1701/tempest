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

        # Find compilers - Use environment variables or defaults
        cc = os.environ.get('CC', '/usr/bin/gcc')
        cxx = os.environ.get('CXX', '/usr/bin/g++')

        # Add compiler paths to cmake args
        cmake_args.append(f'-DCMAKE_C_COMPILER={cc}')
        cmake_args.append(f'-DCMAKE_CXX_COMPILER={cxx}')

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
