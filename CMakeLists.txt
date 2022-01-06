cmake_minimum_required(VERSION 3.5)

project(face_dec)

find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories()
set(SRC 
    main.cpp
    )
add_executable(test ${SRC})
target_link_libraries(test ${OpenCV_LIBS} realsense2 dlib::dlib)