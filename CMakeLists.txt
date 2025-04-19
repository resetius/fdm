cmake_minimum_required(VERSION 3.20)
project(fdm)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(ICPX_EXECUTABLE
    NAMES icpx
    HINTS
      ENV SYCL_ROOT
      /opt/intel/oneapi/compiler/latest/bin
      $ENV{ONEAPI_ROOT}/compiler/latest/bin
)

find_path(ONEMATH_INCLUDE_DIR
    NAMES oneapi/math.hpp
    HINTS
      /usr/local/include
)

if (ICPX_EXECUTABLE AND ONEMATH_INCLUDE_DIR)
    message(STATUS "Found icpx: ${ICPX_EXECUTABLE}")
    message(STATUS "Found oneapi/math.hpp in: ${ONEMATH_INCLUDE_DIR}")
    set(SYCL_FOUND TRUE CACHE BOOL "Found SYCL/icpx + oneapi/math.hpp")
else()
    message(WARNING "SYCL compiler (icpx) or oneapi/math.hpp not found")
    set(SYCL_FOUND FALSE CACHE BOOL "Found SYCL/icpx + oneapi/math.hpp")
endif()

if (SYCL_FOUND)
set(CMAKE_CXX_COMPILER ${ICPX_EXECUTABLE} CACHE FILEPATH "C++ compiler for SYCL target" FORCE)
set(CMAKE_CXX_FLAGS
   "${CMAKE_CXX_FLAGS} -qopenmp -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64 -DHAVE_ONEMATH"
)
endif ()

find_package(PkgConfig REQUIRED)
find_package(OpenMP)
enable_testing()

if (EXISTS ${CMAKE_BINARY_DIR}/local.cmake)
include(${CMAKE_BINARY_DIR}/local.cmake)
endif ()

pkg_check_modules(NETCDF REQUIRED netcdf)
pkg_check_modules(PLPLOT REQUIRED plplot)
# pkg_check_modules(VULKAN vulkan) # remporary disable
pkg_check_modules(FFTW3 fftw3)
pkg_check_modules(FFTW3F fftw3f)

if (NOT BLAS_FOUND)
pkg_check_modules(BLAS openblas)
endif()
if (NOT BLAS_FOUND)
pkg_check_modules(BLAS blas)
endif ()
if (NOT BLAS_FOUND)
pkg_check_modules(BLAS REQUIRED blas-atlas)
endif()

if (NOT LAPACK_FOUND)
include(CheckLibraryExists REQUIRED)
set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LINK_LIBRARIES})
list(GET BLAS_LINK_LIBRARIES 0 HEAD)
check_library_exists(${HEAD} clacpy_ ""  HAS_LAPACK)
unset(CMAKE_REQUIRED_LIBRARIES)
if (HAS_LAPACK)
set (LAPACK_LIBRARIES ${BLAS_LIBRARIES})
set (LAPACK_LIBRARY_DIRS ${BLAS_LIBRARY_DIRS})
else ()
pkg_check_modules(LAPACK lapack)
endif ()
endif ()

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (EXISTS ${CMAKE_SOURCE_DIR}/deps/umfpack/UFconfig/UFconfig.h)
set(UMFPACK_SHARED TRUE)
add_subdirectory(deps/umfpack)
endif()
if (EXISTS ${CMAKE_SOURCE_DIR}/deps/superlu/SRC/superluConfig.cmake.in)
add_subdirectory(deps/superlu_build)
endif ()
if (EXISTS ${CMAKE_SOURCE_DIR}/deps/arpack/CMakeLists.txt)
add_subdirectory(deps/arpack)
endif ()

set(SRC
    src/cyclic_reduction.cpp
    src/sds_bar.cpp
    src/asp_sphere_lapl.cpp
    src/asp_gauss.cpp
    src/asp_sphere_jac.cpp
    src/asp_misc.cpp
    src/sds_bar2.cpp
    src/asp_lapl.cpp
    src/matrix_plot.cpp
    src/asp_fft.cpp
    src/config.cpp
    src/jacobi_solver.cpp
    src/gmres_solver.cpp
    src/arpack_solver.cpp
    src/sparse.cpp
    src/fft.cpp
    src/fft_fftw3.cpp
    src/fft_old.cpp
    src/fft_debug_omp.cpp
    src/lapl_rect.cpp
    src/lapl_cyl.cpp
    src/lapl_cube.cpp
    src/ns_cube.cpp
    src/ns_cyl.cpp
    src/velocity_plot.cpp
    src/eigenvectors_storage.cpp
    src/unixbench_score.cpp
    )
add_library(fdm ${SRC})

if (VULKAN_FOUND)
    find_package(glslang CONFIG REQUIRED)
    find_package(Threads REQUIRED)

    set(COMPUTE_SRC
        compute/vulkan/compute.cpp
        compute/vulkan/command_pool.cpp
        compute/vulkan/loader.c
        compute/vulkan/instance.cpp
        compute/vulkan/device.cpp
        compute/vulkan/shader.cpp
    )
    add_library(compute SHARED ${COMPUTE_SRC})
    target_link_libraries(compute PUBLIC glslang::glslang glslang::SPIRV glslang::glslang-default-resource-limits)
    target_include_directories(compute PUBLIC glslang::glslang glslang::SPIRV glslang::glslang-default-resource-limits)
    target_include_directories(compute PUBLIC ${CMAKE_SOURCE_DIR})

endif ()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    if (COMPILER_SUPPORTS_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
        add_definitions(-D__AVX2__)
    endif()
endif()

if (FFTW3_FOUND)
    set(FFTW3_FLAGS "-DHAVE_FFTW3")
else ()
    set(FFTW3_FLAGS "")
endif ()

if (MSVC)
    target_compile_options(fdm PUBLIC ${PLPLOT_CFLAGS_OTHER} ${FFTW3_FLAGS} PRIVATE "/DFDM_LIBRARY_EXPORT" "/W4")
else ()
    target_compile_options(fdm PUBLIC ${PLPLOT_CFLAGS_OTHER} ${FFTW3_FLAGS} "-Wall")
endif()

target_include_directories(fdm PUBLIC ${PLPLOT_INCLUDE_DIRS} ${BLAS_INCLUDE_DIRS} ${FFTW3_INCLUDE_DIRS} ${NETCDF_INCLUDE_DIRS})
target_link_directories(fdm PUBLIC ${PLPLOT_LIBRARY_DIRS} ${BLAS_LIBRARY_DIRS} ${LAPACK_LIBRARY_DIRS} ${FFTW3_LIBRARY_DIRS} ${NETCDF_LIBRARY_DIRS})
target_link_libraries(fdm PUBLIC ${PLPLOT_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} arpack superlu umfpack ${FFTW3_LIBRARIES} ${FFTW3F_LIBRARIES} ${NETCDF_LIBRARIES})
if (OpenMP_CXX_FOUND)
    target_link_libraries(fdm PUBLIC OpenMP::OpenMP_CXX)
endif()
if (SYCL_FOUND)
    target_link_libraries(fdm PUBLIC onemath_dft_mklgpu onemath)
endif ()

add_subdirectory(test)
add_subdirectory(utils)
add_subdirectory(ut)
