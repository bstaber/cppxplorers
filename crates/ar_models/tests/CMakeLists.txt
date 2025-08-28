# crates/simple_optimizers/tests/CMakeLists.txt

add_executable(ar_models_test test_ar_models.cpp)

target_link_libraries(ar_models_test PRIVATE ar_models)

add_test(NAME ar_models_test COMMAND ar_models_test)