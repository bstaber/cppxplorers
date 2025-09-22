cmake_minimum_required(VERSION 3.25)
project(cuda_hello LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

add_executable(cuda_hello_world hello.cu)
add_executable(cuda_saxpy saxpy.cu)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(cuda_hello_world PRIVATE CUDA::cudart)
target_link_libraries(cuda_saxpy PRIVATE CUDA::cudart)

set_property(TARGET cuda_hello_world PROPERTY CUDA_ARCHITECTURES native)
set_property(TARGET cuda_saxpy PROPERTY CUDA_ARCHITECTURES native)