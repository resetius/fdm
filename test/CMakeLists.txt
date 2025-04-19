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
target_link_libraries(fdm_ns_cyl_spectral fdm)

add_executable(fdm_ns_cube test_ns_cube.cpp)
target_link_libraries(fdm_ns_cube fdm)

add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor fdm)

add_executable(test_solver_bench test_solver_bench.cpp)
target_link_libraries(test_solver_bench fdm)

add_executable(test_arpack test_arpack.cpp)
target_link_libraries(test_arpack fdm)

add_executable(test_nbody nbody.cpp)
target_link_libraries(test_nbody fdm)

add_executable(test_nbody_part nbody_part.cpp)
target_link_libraries(test_nbody_part fdm)

add_executable(test_nbody2d nbody2d.cpp)
target_link_libraries(test_nbody2d fdm)


add_executable(test_fft_bench test_fft_bench.cpp)
target_link_libraries(test_fft_bench fdm)

add_executable(test_fft_3d_bench test_fft_3d_bench.cpp)
target_link_libraries(test_fft_3d_bench fdm)

add_executable(test_tdiag_bench test_tdiag_bench.cpp)
target_link_libraries(test_tdiag_bench fdm)
