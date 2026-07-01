#!/bin/bash

# Ensure a valid repository URL is provided
if [ -z "$1" ]; then
    echo "ERROR: No repository URL provided."
    echo "Usage: $0 <repository-url>"
    exit 1
fi

REPO_URL=$1

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

# Update package list and install dependencies
echo "Updating package list and installing dependencies..."
$SUDO apt-get update
$SUDO apt-get install -y zip pkg-config libtbb-dev git python3 python3-pip python3-venv

# Check if vcpkg already exists
if [ ! -d "/opt/vcpkg" ]; then
    echo "Cloning vcpkg repository..."
    git clone https://github.com/microsoft/vcpkg.git /opt/vcpkg
else
    echo "vcpkg directory already exists. Skipping cloning."
fi

# Bootstrap vcpkg
echo "Bootstrapping vcpkg..."
/opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics

# Install gtest via vcpkg
echo "Installing gtest via vcpkg..."
/opt/vcpkg/vcpkg install gtest

# Set the necessary environment variables and append them to .bashrc
echo "Setting up environment variables..."

# Check if .bashrc exists and append the env variables
if ! grep -q "VCPKG_ROOT" "$HOME/.bashrc"; then
    echo 'export VCPKG_ROOT="/opt/vcpkg"' >> "$HOME/.bashrc"
    echo 'export CMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"' >> "$HOME/.bashrc"
    echo 'export PATH="$VCPKG_ROOT:/root/.local/bin:$PATH"' >> "$HOME/.bashrc"
else
    echo "Environment variables already set in .bashrc."
fi

# Automatically source .bashrc
echo "Sourcing .bashrc to apply changes..."
source ~/.bashrc

# Clone the user-provided repository into the home directory
echo "Cloning repository: $REPO_URL into the home directory"
git clone "$REPO_URL" ~/tempest

# Navigate into the cloned directory
cd ~/tempest || { echo "Failed to change directory"; exit 1; }

# Create a Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists. Skipping creation."
fi

# Activate the virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Build the project
echo "Building the project..."
mkdir cmake_build_debug
cd cmake_build_debug/
cmake .. -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake
make

echo "Setup complete. The wheel has been built successfully."
