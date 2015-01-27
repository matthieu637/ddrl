cmake_minimum_required(VERSION 2.8)

#Project name
project(common-lib CXX C)

IF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE DEBUG)
ENDIF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

#Boost library
find_package(Boost COMPONENTS serialization)

include_directories(${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS})

file ( GLOB_RECURSE all_include include/*.hpp )
file ( GLOB all_sources src/*.cpp )

set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11 -Wno-switch") #-fPIC 
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -02 -ffastmath")

set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/lib")
add_library(common-drl SHARED ${all_sources})
target_link_libraries(common-drl ${Boost_LIBRARIES})