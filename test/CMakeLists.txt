project(fdm_test)
include_directories(../src)

add_executable(fdm_barvortex test_barvortex.cpp)
target_link_libraries(fdm_barvortex fdm)

add_executable(fdm_barvortex_forcing fdm_barvortex_forcing.cpp)
target_link_libraries(fdm_barvortex_forcing fdm)

add_executable(fdm_baroclin test_baroclin.cpp)
target_link_libraries(fdm_baroclin fdm)

add_executable(fdm_slaplace test_slaplace.cpp)
target_link_libraries(fdm_slaplace fdm)

add_executable(fdm_ns_rect test_ns_rect.cpp)
target_link_libraries(fdm_ns_rect fdm)

add_executable(fdm_ns_cyl test_ns_cyl.cpp)
target_link_libraries(fdm_ns_cyl fdm)

add_executable(fdm_ns_cyl_spectral test_ns_cyl_spectral.cpp)
target_link_directories(fdm_ns_cyl_spectral PRIVATE ${NETCDF_LIBRARY_DIRS})
target_include_directories(fdm_ns_cyl_spectral PRIVATE ${NETCDF_INCLUDE_DIRS})
target_link_libraries(fdm_ns_cyl_spectral fdm ${NETCDF_LIBRARIES})

add_executable(fdm_ns_cube test_ns_cube.cpp)
target_link_libraries(fdm_ns_cube fdm)

add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor fdm)

add_executable(test_solver_bench test_solver_bench.cpp)
target_link_libraries(test_solver_bench fdm)

add_executable(test_arpack test_arpack.cpp)
target_link_libraries(test_arpack fdm)
