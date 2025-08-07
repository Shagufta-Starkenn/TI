cmake_minimum_required(VERSION 3.10)

project(OpenSeeFaceMNN)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure OpenCV
set(OpenCV_INCLUDE_DIRS /usr/include/opencv4)
set(OpenCV_LIBS
    /usr/lib/libopencv_core.so
    /usr/lib/libopencv_imgproc.so
    /usr/lib/libopencv_highgui.so
    /usr/lib/libopencv_imgcodecs.so
    /usr/lib/libopencv_videoio.so
)

# Configure MNN with explicit paths
# The path for MNN headers must point to the parent directory of the MNN folder.
# For example, if Interpreter.hpp is at /home/MNN/ti/MNN_os/MNN/include/Interpreter.hpp
# then the include path should be /home/MNN/ti/MNN_os/MNN/include
set(MNN_INCLUDE_DIR /home/MNN/ti/MNN_os/MNN/include)
set(MNN_LIBRARY /home/MNN/ti/lib/libMNN.so) # Assuming your MNN library is named libMNN.so

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${MNN_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}
)

# Add executable
add_executable(openseeface_mnn /home/MNN/ti/src/FINAL_WORKING_CODE.cpp)

# Link libraries
target_link_libraries(openseeface_mnn
    ${OpenCV_LIBS}
    ${MNN_LIBRARY}
)

# Optional: Enable optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
