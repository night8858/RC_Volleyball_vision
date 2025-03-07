cmake_minimum_required(VERSION 3.20)
project(volleyball_detect)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(realsense2 REQUIRED)
find_package(CUDA REQUIRED)0-
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "-w -Ofast -DNDEBUG -flto -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")

include_directories(/usr/src/tensorrt/samples/common)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/include/aarch64-linux-gnu)
include_directories(/usr/include/librealsense2)
include_directories(/usr/include/opencv4)
include_directories(/home/nvidia/RC_Volleyball_track_2025/camera)
include_directories(/home/nvidia/RC_Volleyball_track_2025/detector)
include_directories(/home/nvidia/RC_Volleyball_track_2025/struct)
include_directories(/home/nvidia/RC_Volleyball_track_2025/detector)
include_directories(/home/nvidia/RC_Volleyball_track_2025/communicator)


cuda_add_executable(volleyball_detect main.cpp 
/home/nvidia/RC_Volleyball_track_2025/camera/intel_realsence.cpp
#/home/nvidia/RC_Volleyball_track_2025/detector/nv_detect.cpp
/home/nvidia/RC_Volleyball_track_2025/detector/detect.cpp
/home/nvidia/RC_Volleyball_track_2025/detector/parse_config.cpp
/usr/src/tensorrt/samples/common/logger.cpp
/home/nvidia/RC_Volleyball_track_2025/detector/detect.cu
/home/nvidia/RC_Volleyball_track_2025/communicator/my_serial.h
)

target_link_libraries(volleyball_detect ${realsense2_LIBRARY} ${OpenCV_LIBS})
#target_link_libraries(volleyball_detect /usr/lib/aarch64-linux-gnu)
target_link_libraries(volleyball_detect ${CUDA_LIBRARIES})
target_link_libraries(volleyball_detect ${OpenCV_LIBS})
target_link_libraries(volleyball_detect nvinfer)
target_link_libraries(volleyball_detect nvparsers)
target_link_libraries(volleyball_detect nvonnxparser jsoncpp)