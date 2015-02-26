cmake_minimum_required(VERSION 2.8)

#Project name
project(common-lib CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

#Boost library
find_package(Boost COMPONENTS serialization program_options thread system REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS})

file ( GLOB_RECURSE all_include include/*.hpp )
file ( GLOB all_sources src/*.cpp )

set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11 -Wno-switch") #-fPIC 
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -ffast-math -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O2 -g -DNDEBUG")

set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/lib")


set(LIBRARY_NAME "common-drl")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(LIBRARY_NAME "${LIBRARY_NAME}-d")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(LIBRARY_NAME "${LIBRARY_NAME}-rd")
endif()
  

add_library(${LIBRARY_NAME} SHARED ${all_sources})
target_link_libraries(${LIBRARY_NAME} ${Boost_LIBRARIES})

add_executable(dummy-simulator src/test/DummySimulator.cpp ${all_include} ${all_sources})
target_link_libraries(dummy-simulator ${Boost_LIBRARIES})
file(COPY data/minimal_config.ini DESTINATION . ) 
file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/minimal_config.ini ${CMAKE_CURRENT_BINARY_DIR}/config.ini)
