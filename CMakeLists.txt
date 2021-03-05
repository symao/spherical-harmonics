cmake_minimum_required(VERSION 2.8.3)
project(spherical_harmonics)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-w -std=c++11")

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    ./
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(${PROJECT_NAME} sh/spherical_harmonics.cc sh/sh_api.cpp main.cpp)
target_link_libraries(${PROJECT_NAME} vs_common ${OpenCV_LIBS} pthread)
