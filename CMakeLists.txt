cmake_minimum_required(VERSION 2.8.5)
project(ssmid CXX)

add_definitions(-DPROJECT_NAME=${PROJECT_NAME})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Compilation options
option(DEBUG "Compile with debugging information")

# Include modules in the CMake directory.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

if(DEBUG)
    message("Building with debug information")
    add_definitions(-DDEBUG)
    set(CMAKE_BUILD_TYPE Debug)
else()
    message("Building release")
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Armadillo REQUIRED)
find_package(Mlpack REQUIRED)

# go to subdirectories
add_subdirectory(include)
include_directories(${PROJECT_INCLUDE_DIR})

message(STATUS "Project include directory: ${PROJECT_INCLUDE_DIR}")
message(STATUS "Project header files: ${PROJECT_HEADER_FILES}")
add_subdirectory(src)
add_subdirectory(test)

