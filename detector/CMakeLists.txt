cmake_minimum_required(VERSION 3.10)
project(gpu_detect_test)


find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(/usr/src/tensorrt/samples/common)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/include/aarch64-linux-gnu)
include_directories(/usr/include/librealsense2)
include_directories(/usr/include/opencv4)
include_directories(/home/nvidia/RC_Volleyball_track_2025/struct)
include_directories(/home/nvidia/RC_Volleyball_track_2025/detector)
include_directories(/home/nvidia/RC_Volleyball_track_2025/detector/data)

cuda_add_executable(gpu_detect_test main.cpp 
/home/nvidia/RC_Volleyball_track_2025/detector/process.cu
/home/nvidia/RC_Volleyball_track_2025/detector/detect.cpp
/usr/src/tensorrt/samples/common/logger.cpp
)

#target_link_libraries(volleyball_detect /usr/lib/aarch64-linux-gnu)
target_link_libraries(gpu_detect_test ${CUDA_LIBRARIES})
target_link_libraries(gpu_detect_test ${OpenCV_LIBS})
target_link_libraries(gpu_detect_test nvinfer)
target_link_libraries(gpu_detect_test nvparsers)
target_link_libraries(gpu_detect_test nvonnxparser jsoncpp )
