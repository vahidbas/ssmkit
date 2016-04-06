# build tests

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

add_definitions(-DTEST_MODULE_NAME=${PROJECT_NAME}_test)

# find all .cpp files here
file(GLOB TEST_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
MESSAGE( STATUS "TEST_SRC: " ${TEST_SRC} )
add_definitions(-DBOOST_TEST_DYN_LINK)
add_executable( ${PROJECT_NAME}_test ${TEST_SRC})
target_link_libraries( ${PROJECT_NAME}_test  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${Mlpack_LIBRARIES} ${ARMADILLO_LIBRARIES})

