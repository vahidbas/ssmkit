# build tests
find_package(Boost COMPONENTS unit_test_framework)
if (Boost_UNIT_TEST_FRAMEWORK_FOUND)
  enable_testing()
  
  message(STATUS ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
  add_definitions(-DTEST_MODULE_NAME=ssmpack_test)
  
  # find all .cpp test files
  file(GLOB_RECURSE TEST_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
  # 
  
  add_executable(tests EXCLUDE_FROM_ALL ${TEST_FILES})
  # indicates the include paths
  target_include_directories(tests PRIVATE ${Boost_INCLUDE_DIRS})
  add_definitions(-DBOOST_TEST_DYN_LINK)
  # indicates the shared library variant
  #target_compile_definitions(tests PRIVATE "BOOST_TEST_DYN_LINK=1")
  target_link_libraries(tests ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ARMADILLO_LIBRARIES})
  
  # declare tests
  add_test(NAME alltests COMMAND tests)
endif()
