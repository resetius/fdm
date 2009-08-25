project(fdm)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
file(GLOB SRC src/*.c src/*.cpp src/*.h)
add_library(fdm ${SRC})

add_subdirectory(test)

