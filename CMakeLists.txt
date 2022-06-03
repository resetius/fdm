project(fdm)
cmake_minimum_required(VERSION 3.20)
find_package(PkgConfig REQUIRED)
find_package(OpenMP)

pkg_check_modules(PLPLOT REQUIRED plplot)
if (NOT BLAS_FOUND)
pkg_check_modules(BLAS openblas)
endif()
if (NOT BLAS_FOUND)
pkg_check_modules(BLAS blas)
endif ()
if (NOT BLAS_FOUND)
pkg_check_modules(BLAS REQUIRED blas-atlas)
endif()

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

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_CXX_STANDARD 17)
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
    )
add_library(fdm SHARED ${SRC})

if (MSVC)
    target_compile_options(fdm PUBLIC ${PLPLOT_CFLAGS_OTHER} PRIVATE "/DFDM_LIBRARY_EXPORT" "/W4")
else ()
    target_compile_options(fdm PUBLIC ${PLPLOT_CFLAGS_OTHER} "-Wall")
endif()
target_include_directories(fdm PUBLIC ${PLPLOT_INCLUDE_DIRS} ${BLAS_INCLUDE_DIRS})
target_link_directories(fdm PUBLIC ${PLPLOT_STATIC_LIBRARY_DIRS} ${BLAS_LIBRARY_DIRS})
target_link_libraries(fdm PUBLIC ${PLPLOT_STATIC_LIBRARIES} ${BLAS_LIBRARIES} arpack superlu umfpack)
if (OpenMP_CXX_FOUND)
    target_link_libraries(fdm PUBLIC OpenMP::OpenMP_CXX)
endif()

add_subdirectory(test)
add_subdirectory(utils)
