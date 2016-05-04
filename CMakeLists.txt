cmake_minimum_required(VERSION 3.2.0)
project(ssmpack CXX)

add_definitions(-DPROJECT_NAME=${PROJECT_NAME})
add_definitions(-DSOURCE_DIR=${CMAKE_SOURCE_DIR})


# Include modules in the CMake directory.
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Compilation options
  option(DEBUG "Compile with debugging information")
if(DEBUG)
  message("Building with debug information")
  add_definitions(-DDEBUG)
  set(CMAKE_BUILD_TYPE Debug)
else()
  message("Building release")
  set(CMAKE_BUILD_TYPE Release)
endif()

#clang
  option(CLANG "build application with clang")
if(CLANG)
  SET (CMAKE_CXX_COMPILER "/usr/bin/clang++")
  set(CMAKE_CXX_FLAGS "-Wall -stdlib=libstdc++ -std=c++14")
  message(STATUS "Clang compiler will be used")
endif(CLANG)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0")

  find_package(Armadillo REQUIRED)
  find_package(Mlpack REQUIRED)

# target directory
#set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include_directories(${CMAKE_SOURCE_DIR}/src)
# go to subdirectories
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(example)

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

# other staffs related to documantation (e.g. diagrams)
  add_subdirectory(doc)

# Generate documentation.
  add_custom_target(doc
      COMMAND "${DOXYGEN_EXECUTABLE}" "${CMAKE_BINARY_DIR}/Doxyfile"
      DEPENDS "${CMAKE_BINARY_DIR}/Doxyfile"
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      COMMENT "Generating API documentation with Doxygen"
      )

  add_dependencies(doc tikz_images)

  install(DIRECTORY ${CMAKE_BINARY_DIR}/doc/html
      DESTINATION share/doc/ssmpack
      COMPONENT doc
      OPTIONAL
      )
endif ()
