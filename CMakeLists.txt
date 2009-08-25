project(fdm)
file(GLOB SRC src/*.c src/*.cpp src/*.h)
add_library(fdm ${SRC})
