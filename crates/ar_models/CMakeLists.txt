# Header-only target
add_library(ar_models INTERFACE)

# Public includes (exports the include/ dir to consumers)
target_include_directories(ar_models INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link Eigen and set language level
target_link_libraries(ar_models INTERFACE Eigen3::Eigen)
target_compile_features(ar_models INTERFACE cxx_std_20)

# Demo executable
add_executable(ar_demo src/main.cpp)
target_link_libraries(ar_demo PRIVATE ar_models)

# Tests
enable_testing()
add_subdirectory(tests)
