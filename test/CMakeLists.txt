project(fdm_test)
include_directories(../src)

add_executable(fdm_barvortex test_barvortex.cpp)
target_link_libraries(fdm_barvortex fdm)

add_executable(fdm_barvortex_forcing fdm_barvortex_forcing.cpp config.cpp)
target_link_libraries(fdm_barvortex_forcing fdm)

add_executable(fdm_baroclin test_baroclin.cpp)
target_link_libraries(fdm_baroclin fdm)

add_executable(fdm_slaplace test_slaplace.cpp)
target_link_libraries(fdm_slaplace fdm)

add_executable(fdm_ns_rect test_ns_rect.cpp config.cpp)
target_link_libraries(fdm_ns_rect fdm)

if (OPENMP_FOUND)
	set_target_properties(fdm_barvortex PROPERTIES LINK_FLAGS ${OpenMP_C_FLAGS})
	set_target_properties(fdm_barvortex_forcing PROPERTIES LINK_FLAGS ${OpenMP_C_FLAGS})
	set_target_properties(fdm_baroclin PROPERTIES LINK_FLAGS ${OpenMP_C_FLAGS})
	set_target_properties(fdm_slaplace PROPERTIES LINK_FLAGS ${OpenMP_C_FLAGS})
	set_target_properties(fdm_ns_rect PROPERTIES LINK_FLAGS ${OpenMP_C_FLAGS})
endif (OPENMP_FOUND)
