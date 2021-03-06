cmake_minimum_required(VERSION 2.8)

#Project name
project(common-lib C CXX)

if(NOT DEFINED ROOT_DRL_PATH)
        set(ROOT_DRL_PATH "${CMAKE_SOURCE_DIR}/..")
endif()

include(${CMAKE_SOURCE_DIR}/cmake/Config.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/GTest.cmake) #GTest must be before Caffe (depedency)
include(${CMAKE_SOURCE_DIR}/cmake/BoostConfig.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/CaffeConfig.cmake)

include_directories(${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS})

file ( GLOB_RECURSE all_include include/*.hpp )
file ( GLOB all_sources src/*.cpp src/zoo/*.cpp)

#-------------------
# Library Build
#-------------------
set(LIBRARY_NAME "common-drl")
rename_buildtype(LIBRARY_NAME)
add_library(${LIBRARY_NAME} SHARED ${all_sources})
target_link_libraries(${LIBRARY_NAME} ${Boost_LIBRARIES} caffe)

#-------------------
# Personnals Test
#-------------------
add_executable(dummy-simulator src/test/DummySimulator.cpp ${all_include} ${all_sources})
target_link_libraries(dummy-simulator ${Boost_LIBRARIES} caffe)

add_executable(dpmt-dummy-simulator src/test/DpmtDummySimulator.cpp ${all_include} ${all_sources})
target_link_libraries(dpmt-dummy-simulator ${Boost_LIBRARIES} caffe)

file(COPY data/minimal_config.ini DESTINATION . ) 
file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/minimal_config.ini ${CMAKE_CURRENT_BINARY_DIR}/config.ini)

enable_utest(all_sources ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES} caffe)
