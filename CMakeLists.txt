project(fdm)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")

file(GLOB SRC src/*.c src/*.cpp src/*.h)
add_library(fdm ${SRC})

add_subdirectory(test)
add_subdirectory(utils)

