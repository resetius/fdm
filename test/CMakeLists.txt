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
target_link_libraries(fdm_ns_rect fdm umfpack)
