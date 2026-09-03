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

add_executable(fdm_ns_cyl_spectral_probe test_ns_cyl_spectral_probe.cpp)
target_link_libraries(fdm_ns_cyl_spectral_probe fdm)

add_executable(fdm_ns_cyl_spectral_filter_experiment
    test_ns_cyl_spectral_filter_experiment.cpp)
target_link_libraries(fdm_ns_cyl_spectral_filter_experiment fdm)

add_executable(fdm_ns_cyl_spectral_filter_developed
    test_ns_cyl_spectral_filter_developed.cpp)
target_link_libraries(fdm_ns_cyl_spectral_filter_developed fdm)

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

if (SYCL_FOUND AND APPLE)
find_path(METAL_CPP_INCLUDE_DIR NAMES Metal/Metal.hpp
    HINTS $ENV{HOME}/Projects/AdaptiveCpp/metal-cpp)
find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
    pkg_check_modules(SDL2 QUIET sdl2)
endif()
if (METAL_CPP_INCLUDE_DIR AND SDL2_FOUND)
    add_executable(ns_cube_sycl_demo
        ns_cube_sycl_demo.cpp
        ns_cube_sycl_demo_metal_impl.cpp)
    target_link_libraries(ns_cube_sycl_demo fdm)
    target_include_directories(ns_cube_sycl_demo PRIVATE
        ${METAL_CPP_INCLUDE_DIR}
        ${SDL2_INCLUDE_DIRS})
    target_link_directories(ns_cube_sycl_demo PRIVATE
        $ENV{HOME}/opt/adaptivecpp/lib
        ${SDL2_LIBRARY_DIRS}
        $ENV{HOME}/opt/adaptivecpp/lib/hipSYCL)
    target_link_libraries(ns_cube_sycl_demo
        ${SDL2_LIBRARIES}
        rt-backend-metal
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore")
    set_target_properties(ns_cube_sycl_demo PROPERTIES
        BUILD_RPATH "$ENV{HOME}/opt/adaptivecpp/lib/hipSYCL")

    add_executable(ns_cyl_sycl_demo
        ns_cyl_sycl_demo.cpp
        ns_cyl_sycl_demo_metal_impl.cpp)
    target_link_libraries(ns_cyl_sycl_demo fdm)
    target_include_directories(ns_cyl_sycl_demo PRIVATE
        ${METAL_CPP_INCLUDE_DIR}
        ${SDL2_INCLUDE_DIRS})
    target_link_directories(ns_cyl_sycl_demo PRIVATE
        $ENV{HOME}/opt/adaptivecpp/lib
        ${SDL2_LIBRARY_DIRS}
        $ENV{HOME}/opt/adaptivecpp/lib/hipSYCL)
    target_link_libraries(ns_cyl_sycl_demo
        ${SDL2_LIBRARIES}
        rt-backend-metal
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore")
    set_target_properties(ns_cyl_sycl_demo PROPERTIES
        BUILD_RPATH "$ENV{HOME}/opt/adaptivecpp/lib/hipSYCL")
else()
    message(STATUS "metal-cpp or SDL2 not found — sycl demos skipped")
endif()
endif()

if (SYCL_FOUND)
add_executable(test_tile_bench test_tile_bench.cpp)
target_link_libraries(test_tile_bench fdm)

find_path(BOOST_INCLUDE_DIR NAMES boost/version.hpp
    HINTS /opt/homebrew/include)
if (BOOST_INCLUDE_DIR)
add_executable(test_mandelbrot test_mandelbrot.cpp)
target_link_libraries(test_mandelbrot fdm)
target_include_directories(test_mandelbrot PRIVATE ${BOOST_INCLUDE_DIR})
else()
message(STATUS "Boost not found — test_mandelbrot skipped")
endif()
endif ()
