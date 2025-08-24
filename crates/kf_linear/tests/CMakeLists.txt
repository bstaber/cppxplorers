# crates/kf_linear/tests/CMakeLists.txt

find_package(Catch2 3 REQUIRED CONFIG)

add_executable(kf_tests test_kf_linear.cpp)

target_link_libraries(kf_tests PRIVATE kf_linear Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(kf_tests)