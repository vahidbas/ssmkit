# finding all *.hpp files in sub_directories
file(GLOB_RECURSE HPP_FILES ${CMAKE_CURRENT_SOURCE_DIR} *.hpp)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ssmpack DESTINATION include
  COMPONENT ssmpack)
