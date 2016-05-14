# build tests

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

add_definitions(-DTEST_MODULE_NAME=ssmpack_test)

# find all .cpp test files
file(GLOB_RECURSE TEST_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# 
add_definitions(-DBOOST_TEST_DYN_LINK)

add_executable( ${PROJECT_NAME}_test ${TEST_FILES})
target_link_libraries( ssmpack_test  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ARMADILLO_LIBRARIES})

