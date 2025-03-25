cmake_minimum_required(VERSION 3.20)
project(volleyball_detect)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(realsense2 REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "-w -Ofast -DNDEBUG -flto -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")

include_directories(/usr/src/tensorrt/samples/common)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/include/aarch64-linux-gnu)
include_directories(/usr/include/librealsense2)
include_directories(/usr/include/opencv4)
include_directories(/home/nvidia/RC_Volleyball_vision/camera)
include_directories(/home/nvidia/RC_Volleyball_vision/detector)
include_directories(/home/nvidia/RC_Volleyball_vision/struct)
include_directories(/home/nvidia/RC_Volleyball_vision/communicator)
include_directories(/opt/MVS/include)

link_directories(/opt/MVS/lib/aarch64)
cuda_add_executable(volleyball_detect main.cpp 
/home/nvidia/RC_Volleyball_vision/camera/intel_realsence.cpp
/home/nvidia/RC_Volleyball_vision/camera/hikvision.cpp
/home/nvidia/RC_Volleyball_vision/camera/usbcam.cpp
#/home/nvidia/RC_Volleyball_vision/detector/nv_detect.cpp
/home/nvidia/RC_Volleyball_vision/detector/detect.cpp
/home/nvidia/RC_Volleyball_vision/detector/postprocess.cpp
/usr/src/tensorrt/samples/common/logger.cpp
/home/nvidia/RC_Volleyball_vision/detector/process.cu
/home/nvidia/RC_Volleyball_vision/communicator/my_serial.cpp
)

target_link_libraries(volleyball_detect ${realsense2_LIBRARY} ${OpenCV_LIBS})
#target_link_libraries(volleyball_detect /usr/lib/aarch64-linux-gnu)
target_link_libraries(volleyball_detect ${CUDA_LIBRARIES})
target_link_libraries(volleyball_detect ${OpenCV_LIBS})
target_link_libraries(volleyball_detect nvinfer)
target_link_libraries(volleyball_detect nvparsers)
target_link_libraries(volleyball_detect nvonnxparser jsoncpp)
target_link_libraries(volleyball_detect MvCameraControl ${OpenCV_LIBS})