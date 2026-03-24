project(utils)

add_definitions(-DTEST)
add_executable(fdm_topo30_mesh fdm_srtm_rel.cpp)

if (OpenMP_CXX_FOUND)
    target_link_libraries(fdm_topo30_mesh PRIVATE OpenMP::OpenMP_CXX)
elseif (SYCL_OMP_LIBRARIES)
    target_link_directories(fdm_topo30_mesh PRIVATE ${SYCL_OMP_LIBRARIES})
endif()
