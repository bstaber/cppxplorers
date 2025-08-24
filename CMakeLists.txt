cmake_minimum_required(VERSION 3.15)

# Project configuration
project(cppxplorers LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependencies from Spack
# Spack's "view: true" ensures these are in CMAKE_PREFIX_PATH automatically
find_package(Eigen3 REQUIRED NO_MODULE)

# Enable tests
include(CTest)      # sets BUILD_TESTING option
enable_testing()    # actually turns testing on

# Add crates subdirectory
add_subdirectory(crates)
