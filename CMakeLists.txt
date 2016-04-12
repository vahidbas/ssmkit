cmake_minimum_required(VERSION 2.8.5)
project(ssmpack CXX)

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
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# go to subdirectories
  add_subdirectory(src)
include_directories(${PROJECT_INCLUDE_DIR})

  message(STATUS "Project include directory: ${PROJECT_INCLUDE_DIR}")
  message(STATUS "Project header files: ${PROJECT_HEADER_FILES}")
# add_subdirectory(src)
add_subdirectory(test)

# Generate the documentation.
  find_package(Doxygen)
if (DOXYGEN_FOUND)
# Preprocess the Doxyfile.  This is done before 'make doc'.
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/Doxyfile
      PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -D DESTDIR="${CMAKE_BINARY_DIR}" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateDoxyfile.cmake"
      WORKING_DIRECTORY
      "${CMAKE_CURRENT_SOURCE_DIR}"
      DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile"
      COMMENT "Creating Doxyfile to generate Doxygen documentation"
      )

# Generate documentation.
  add_custom_target(doc
      COMMAND "${DOXYGEN_EXECUTABLE}" "${CMAKE_BINARY_DIR}/Doxyfile"
      DEPENDS "${CMAKE_BINARY_DIR}/Doxyfile"
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      COMMENT "Generating API documentation with Doxygen"
      )

  install(DIRECTORY ${CMAKE_BINARY_DIR}/doc/html
      DESTINATION share/doc/mlpack
      COMPONENT doc
      OPTIONAL
      )
endif ()
