cmake_minimum_required(VERSION 3.10)

project(serilal_test)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

#add_definitions(-DUSE_NVIDIA)
#add_definitions(-DTWO_CAMERAS)

include_directories(./)
link_directories(./)

add_executable(serilal_test serial_test.cpp my_serial.h my_serial.cpp )

