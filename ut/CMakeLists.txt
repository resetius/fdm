project(ut)

pkg_check_modules(CMOCKA cmocka)

if (CMOCKA_FOUND)

include_directories(../src)

add_executable(ut_tensor ut_tensor.cpp)
target_include_directories(ut_tensor PRIVATE ${CMOCKA_INCLUDE_DIRS})
target_link_directories(ut_tensor PRIVATE ${CMOCKA_LIBRARY_DIRS})
target_link_libraries(ut_tensor fdm ${CMOCKA_LIBRARIES})

add_test(NAME tensor COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ut_tensor)
set_tests_properties(tensor PROPERTIES ENVIRONMENT "CMOCKA_MESSAGE_OUTPUT=xml;CMOCKA_XML_FILE=tensor.xml")

add_executable(ut_arpack_solver ut_arpack_solver.cpp)
target_include_directories(ut_arpack_solver PRIVATE ${CMOCKA_INCLUDE_DIRS})
target_link_directories(ut_arpack_solver PRIVATE ${CMOCKA_LIBRARY_DIRS})
target_link_libraries(ut_arpack_solver fdm ${CMOCKA_LIBRARIES})

add_test(NAME arpack_solver COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ut_arpack_solver)
set_tests_properties(arpack_solver PROPERTIES ENVIRONMENT "CMOCKA_MESSAGE_OUTPUT=xml;CMOCKA_XML_FILE=tensor.xml")

endif()
