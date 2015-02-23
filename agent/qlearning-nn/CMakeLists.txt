cmake_minimum_required(VERSION 2.8)

#Project name
project(qlearning-nn CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

#Boost library
find_package(Boost COMPONENTS serialization program_options)
find_package(CommonDRL)
find_package(AdvancedAcrobot)

include_directories(${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} ${COMMON_DRL_INCLUDE_DIRS} ${ADVANCED_ACROBOT_INCLUDE_DIRS})


file ( GLOB_RECURSE all_include include/*.hpp )
file ( GLOB all_sources src/*.cpp )

set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11 -Wno-switch") #-fPIC 
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -ffast-math -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O2 -g -DNDEBUG")


add_executable(advanced-acrobot src/test/AdvancedAcrobot.cpp ${all_include} ${all_sources})
target_link_libraries(advanced-acrobot optimized ${Boost_LIBRARIES} ${COMMON_DRL_LIBRARY} ${ADVANCED_ACROBOT_LIBRARY})
#file(COPY data/minimal_config.ini DESTINATION . ) 
#file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/minimal_config.ini ${CMAKE_CURRENT_BINARY_DIR}/config.ini)

