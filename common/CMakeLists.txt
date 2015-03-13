cmake_minimum_required(VERSION 2.8)

#Project name
project(common-lib CXX)

include(${CMAKE_SOURCE_DIR}/cmake/Config.cmake)

#Boost library
find_package(Boost COMPONENTS serialization program_options thread system REQUIRED)
find_package(GTest REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS})

file ( GLOB_RECURSE all_include include/*.hpp )
file ( GLOB all_sources src/*.cpp )

#-------------------
# Library Build
#-------------------
set(LIBRARY_NAME "common-drl")
rename_buildtype(LIBRARY_NAME)
add_library(${LIBRARY_NAME} SHARED ${all_sources})
target_link_libraries(${LIBRARY_NAME} ${Boost_LIBRARIES})

#-------------------
# Personnals Test
#-------------------
add_executable(dummy-simulator src/test/DummySimulator.cpp ${all_include} ${all_sources})
target_link_libraries(dummy-simulator ${Boost_LIBRARIES})
file(COPY data/minimal_config.ini DESTINATION . ) 
file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/minimal_config.ini ${CMAKE_CURRENT_BINARY_DIR}/config.ini)

enable_utest(all_sources ${Boost_LIBRARIES})
