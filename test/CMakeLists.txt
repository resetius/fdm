project(fdm_test)
include_directories(../src)
add_executable(test_barvortex test_barvortex.cpp)
target_link_libraries(test_barvortex fdm)
