cmake_minimum_required(VERSION 2.8)

#Nom du projet
project(ode-env CXX C)

if(NOT DEFINED ROOT_DRL_PATH)
	set(ROOT_DRL_PATH "${CMAKE_SOURCE_DIR}/../..")
endif()

include(${ROOT_DRL_PATH}/common/cmake/Config.cmake)

find_package(Boost 1.54 COMPONENTS serialization filesystem system program_options thread REQUIRED)
find_package(OpenGL)
find_package(X11)
find_package(TBB)
find_package(Threads)
find_package(CommonDRL)
include(${ROOT_DRL_PATH}/common/cmake/ODEConfig.cmake)

include_directories(${CMAKE_SOURCE_DIR}/include ${ODE_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/extern/drawstuff/include/ ${TBB_INCLUDE_DIRS} ${COMMON_DRL_INCLUDE_DIRS})


file ( GLOB_RECURSE drawstuff_src extern/drawstuff/src/*.cpp)

file ( GLOB_RECURSE all_include include/*.hpp )
file ( GLOB all_sources src/*.cpp )

message(STATUS "ODE flags : " ${ODE_CFLAGS})

#-------------------
# Library Build
#-------------------
set(LIBRARY_NAME "ode-env")
rename_buildtype(LIBRARY_NAME)
add_library(${LIBRARY_NAME} SHARED ${all_include} ${all_sources} ${drawstuff_src})
message(${LIBRARY_NAME} ${ODE_LIBRARY} ${Boost_LIBRARIES} ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${COMMON_DRL_LIBRARY})
target_link_libraries(${LIBRARY_NAME} ${ODE_LIBRARY} ${Boost_LIBRARIES} ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${COMMON_DRL_LIBRARY_LIGHT})

