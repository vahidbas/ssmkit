# include directory

set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

# finding all *.hpp files in sub_directories
file(GLOB_RECURSE HPP_FILES ${CMAKE_SOURCE_DIR} *.hpp)
# PROJECT_HEADER_FILES is available from parent directory
set(PROJECT_HEADER_FILES ${HPP_FILES} PARENT_SCOPE)

#message(STATUS "Project include directory: ${PROJECT_INCLUDE_DIR}")
#message(STATUS "Project header files: ${PROJECT_HEADER_FILES}")
