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
set_tests_properties(arpack_solver PROPERTIES ENVIRONMENT "CMOCKA_MESSAGE_OUTPUT=xml;CMOCKA_XML_FILE=arpack_solver.xml")

add_executable(ut_fft ut_fft.cpp)
target_include_directories(ut_fft PRIVATE ${CMOCKA_INCLUDE_DIRS})
target_link_directories(ut_fft PRIVATE ${CMOCKA_LIBRARY_DIRS})
target_link_libraries(ut_fft fdm ${CMOCKA_LIBRARIES})

add_test(NAME fft COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ut_fft)
set_tests_properties(fft PROPERTIES ENVIRONMENT "CMOCKA_MESSAGE_OUTPUT=xml;CMOCKA_XML_FILE=fft.xml")

add_executable(ut_lapl_cyl ut_lapl_cyl.cpp)
target_include_directories(ut_lapl_cyl PRIVATE ${CMOCKA_INCLUDE_DIRS})
target_link_directories(ut_lapl_cyl PRIVATE ${CMOCKA_LIBRARY_DIRS})
target_link_libraries(ut_lapl_cyl fdm ${CMOCKA_LIBRARIES})

add_test(NAME lapl_cyl COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ut_lapl_cyl)
set_tests_properties(lapl_cyl PROPERTIES ENVIRONMENT "CMOCKA_MESSAGE_OUTPUT=xml;CMOCKA_XML_FILE=lapl_cyl.xml")

add_executable(ut_lapl_rect ut_lapl_rect.cpp)
target_include_directories(ut_lapl_rect PRIVATE ${CMOCKA_INCLUDE_DIRS})
target_link_directories(ut_lapl_rect PRIVATE ${CMOCKA_LIBRARY_DIRS})
target_link_libraries(ut_lapl_rect fdm ${CMOCKA_LIBRARIES})

add_test(NAME lapl_rect COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ut_lapl_rect)
set_tests_properties(lapl_rect PROPERTIES ENVIRONMENT "CMOCKA_MESSAGE_OUTPUT=xml;CMOCKA_XML_FILE=lapl_rect.xml")

endif()
