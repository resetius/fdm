project(fdm)
cmake_minimum_required(VERSION 3.20)
find_package(PkgConfig REQUIRED)
find_package(OpenMP)
pkg_check_modules(PLPLOT REQUIRED plplot)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    )
add_library(fdm SHARED ${SRC})

if (MSVC)
    target_compile_options(fdm PUBLIC ${PLPLOT_CFLAGS_OTHER} PRIVATE "/DFDM_LIBRARY_EXPORT" "/W4")
else ()
    target_compile_options(fdm PUBLIC ${PLPLOT_CFLAGS_OTHER} "-Wall")
endif()
target_include_directories(fdm PUBLIC ${PLPLOT_INCLUDE_DIRS})
target_link_directories(fdm PUBLIC ${PLPLOT_STATIC_LIBRARY_DIRS})
target_link_libraries(fdm PUBLIC ${PLPLOT_STATIC_LIBRARIES})
if (OpenMP_CXX_FOUND)
    target_link_libraries(fdm PUBLIC OpenMP::OpenMP_CXX)
endif()

set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

if (EXISTS ${CMAKE_SOURCE_DIR}/deps/umfpack/UFconfig/UFconfig.h)
set(UMFPACK_SHARED TRUE)
add_subdirectory(deps/umfpack)
endif()

add_subdirectory(test)
add_subdirectory(utils)
