cmake_minimum_required(VERSION 3.10)
project(UltraFace)

# Manually specify OpenCV
set(OpenCV_INCLUDE_DIRS /usr/include/opencv4)
set(OpenCV_LIBS
    /usr/lib/libopencv_core.so
    /usr/lib/libopencv_imgproc.so
    /usr/lib/libopencv_highgui.so
    /usr/lib/libopencv_imgcodecs.so
    /usr/lib/libopencv_videoio.so
)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find MNN
set(MNN_LIB_DIR /home/MNN/ti/lib/)
set(MNN_INCLUDE_DIR /home/MNN/ti/MNN_os/MNN/include/)
find_library(MNN_LIB MNN HINTS ${MNN_LIB_DIR})
include_directories(${MNN_INCLUDE_DIR})

# Include directories for source files
include_directories(${CMAKE_SOURCE_DIR}/src)

# Source files
set(SOURCES
    src/webcam_mnn.cpp
)

# Create executable
add_executable(Ultra-face-mnn ${SOURCES})

# Link libraries
target_link_libraries(Ultra-face-mnn ${OpenCV_LIBS} ${MNN_LIB})
