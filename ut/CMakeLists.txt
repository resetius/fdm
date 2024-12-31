project(ut)

pkg_check_modules(CMOCKA cmocka)

if (CMOCKA_FOUND)

include_directories(../src)

macro(ut name source)
  add_executable(ut_${name} ${source})
  target_include_directories(ut_${name} PRIVATE ${CMOCKA_INCLUDE_DIRS})
  target_link_directories(ut_${name} PRIVATE ${CMOCKA_LIBRARY_DIRS})
  target_link_libraries(ut_${name} fdm ${CMOCKA_LIBRARIES})
  if (VULKAN_FOUND)
    target_link_libraries(ut_${name} compute)
  endif ()

  add_test(NAME ${name} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ut_${name} ${CMAKE_SOURCE_DIR})
  set_tests_properties(${name} PROPERTIES ENVIRONMENT "CMOCKA_MESSAGE_OUTPUT=xml;CMOCKA_XML_FILE=${name}.xml")
endmacro()

ut(tensor ut_tensor.cpp)
ut(arpack_solver ut_arpack_solver.cpp)
ut(fft ut_fft.cpp)
ut(lapl_cyl ut_lapl_cyl.cpp)
ut(lapl_cyl_slice ut_lapl_cyl_slice.cpp)
ut(lapl_rect ut_lapl_rect.cpp)
ut(lapl_cube ut_lapl_cube.cpp)
ut(eig_storage ut_eigenvectors_storage.cpp)
ut(mgsch ut_mgsch.cpp)
ut(interpolate ut_interpolate.cpp)
ut(nbody ut_nbody.cpp)
ut(compute ut_compute.cpp)
ut(tdiag ut_tdiag.cpp)

endif()
