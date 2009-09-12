project(fdm_test)
include_directories(../src)
add_executable(fdm_barvortex test_barvortex.cpp)
target_link_libraries(fdm_barvortex fdm)

if (OPENMP_FOUND)
	set_target_properties(fdm_barvortex PROPERTIES LINK_FLAGS ${OpenMP_C_FLAGS})
endif (OPENMP_FOUND)
