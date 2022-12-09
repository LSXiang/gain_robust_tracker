cmake_minimum_required(VERSION 3.2)
project(gain_robust_tracker)

set(CMAKE_CXX_STANDARD 11)

include_directories(./include)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(gainRobustTracker src/gain_robust_tracker.cc)
target_link_libraries(gainRobustTracker ${OpenCV_LIBS})

add_executable(gain_robust_tracker example/main.cpp)
target_link_libraries(gain_robust_tracker gainRobustTracker ${OpenCV_LIBS})
