project(ut)

pkg_check_modules(CMOCKA cmocka)

if (CMOCKA_FOUND)

include_directories(../src)
add_executable(ut_tensor ut_tensor.cpp)
target_include_directories(ut_tensor PRIVATE ${CMOCKA_INCLUDE_DIRS})
target_link_directories(ut_tensor PRIVATE ${CMOCKA_LIBRARY_DIRS})
target_link_libraries(ut_tensor fdm ${CMOCKA_LIBRARIES})

add_test(NAME tensor COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ut_tensor)

endif()


