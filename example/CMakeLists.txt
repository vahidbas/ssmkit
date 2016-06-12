file(GLOB EXAMPLE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# each example file should be main()
foreach( EXAMPLE_FILE ${EXAMPLE_FILES} )
  string( REPLACE ".cpp" "" EXAMPLE_NAME ${EXAMPLE_FILE} )
  add_executable( ${EXAMPLE_NAME} EXCLUDE_FROM_ALL ${EXAMPLE_FILE})
  target_link_libraries(${EXAMPLE_NAME} ${ARMADILLO_LIBRARIES})
  list(APPEND EXAMPLES ${EXAMPLE_NAME})
endforeach(EXAMPLE_FILE ${EXAMPLE_FILES})

add_custom_target(examples)
add_dependencies(examples ${EXAMPLES})
