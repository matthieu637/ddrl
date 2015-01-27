cmake_minimum_required(VERSION 2.8)

#Nom du projet
project(advanced-acrobot CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

IF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE DEBUG)
ENDIF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

find_package(Boost COMPONENTS serialization filesystem system REQUIRED)
find_package(ODE REQUIRED)
#find_package(Drawstuff) #to display the environnement
find_package(OpenGL)
find_package(TBB)
find_package(Threads)
find_package(CommonDRL)

include_directories(${CMAKE_SOURCE_DIR}/include ${ODE_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/extern/drawstuff/include/ ${TBB_INCLUDE_DIRS} ${COMMON_DRL_INCLUDE_DIRS})


file ( GLOB_RECURSE drawstuff_src extern/drawstuff/src/*.cpp)

file ( GLOB_RECURSE all_include include/*.hpp )
file ( GLOB all_sources src/*.cpp )

message(STATUS "ODE flags : " ${ODE_CFLAGS})

set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11 -Wno-switch") #-fPIC 
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -02 -ffastmath")

# if(DRAWSTUFF_FOUND)
  add_executable(internal-visu src/test/InternalView.cpp ${all_include} ${all_sources} ${drawstuff_src})
  target_link_libraries(internal-visu ${ODE_LIBRARY} ${Boost_LIBRARIES} ${DRAWSTUFF_LIBRARY} ${OPENGL_LIBRARIES} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${COMMON_DRL_LIBRARY})
# endif(DRAWSTUFF_FOUND)

add_executable(multi-instance src/test/MultiInstance.cpp ${all_include} ${all_sources} ${drawstuff_src} )
target_link_libraries(multi-instance ${ODE_LIBRARY} ${Boost_LIBRARIES} ${DRAWSTUFF_LIBRARY} ${OPENGL_LIBRARIES} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${COMMON_DRL_LIBRARY})


file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
