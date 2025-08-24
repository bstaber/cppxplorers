# Header-only target
add_library(kf_linear INTERFACE)

# Public includes (exports the include/ dir to consumers)
target_include_directories(kf_linear INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link Eigen and set language level
target_link_libraries(kf_linear INTERFACE Eigen3::Eigen)
target_compile_features(kf_linear INTERFACE cxx_std_17)

# Demo executable
add_executable(kf_demo src/main.cpp)
target_link_libraries(kf_demo PRIVATE kf_linear)

# Tests
enable_testing()
add_subdirectory(tests)
