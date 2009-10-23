project(fdm_test)
include_directories(../src)
add_executable(fdm_barvortex test_barvortex.cpp)
target_link_libraries(fdm_barvortex fdm)

add_executable(fdm_baroclin test_baroclin.cpp)
target_link_libraries(fdm_baroclin fdm)

if (OPENMP_FOUND)
	set_target_properties(fdm_barvortex PROPERTIES LINK_FLAGS ${OpenMP_C_FLAGS})
	set_target_properties(fdm_baroclin PROPERTIES LINK_FLAGS ${OpenMP_C_FLAGS})
endif (OPENMP_FOUND)
