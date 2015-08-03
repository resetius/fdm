project(fdm)
cmake_minimum_required(VERSION 2.8)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")

file(GLOB SRC src/*.c src/*.cpp src/*.h)
add_library(fdm SHARED ${SRC})
if (MSVC)
	set_source_files_properties(${SRC} PROPERTIES
		COMPILE_FLAGS "/DFDM_LIBRARY_EXPORT")
endif()

set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

add_subdirectory(test)
add_subdirectory(utils)

