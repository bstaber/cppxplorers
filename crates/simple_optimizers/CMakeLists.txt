# Header-only target
add_library(simple_optimizers INTERFACE)

# Public includes (exports the include/ dir to consumers)
target_include_directories(simple_optimizers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link Eigen and set language level
target_link_libraries(simple_optimizers INTERFACE Eigen3::Eigen)
target_compile_features(simple_optimizers INTERFACE cxx_std_20)

# Demo executable
add_executable(opt_demo src/main.cpp)
target_link_libraries(opt_demo PRIVATE simple_optimizers)

# Tests
enable_testing()
add_subdirectory(tests)
