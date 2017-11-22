cmake_minimum_required(VERSION 2.8)

#Nom du projet
project(ode-env CXX C)

if(NOT DEFINED ROOT_DRL_PATH)
	set(ROOT_DRL_PATH "${CMAKE_SOURCE_DIR}/../..")
endif()

include(${ROOT_DRL_PATH}/common/cmake/Config.cmake)

find_package(OpenGL)
find_package(CommonDRL)
include(${ROOT_DRL_PATH}/common/cmake/ODEConfig.cmake)

include_directories(${CMAKE_SOURCE_DIR}/include ${ODE_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/extern/drawstuff/include/ ${COMMON_DRL_INCLUDE_DIRS})


set(drawstuff_src extern/drawstuff/src/drawstuff.cpp)
if(APPLE)
	find_package(GLUT)
	set(drawstuff_src ${drawstuff_src} extern/drawstuff/src/osx.cpp)
	set(draw_lib ${GLUT_LIBRARIES})
else()
	find_package(X11)
	set(drawstuff_src ${drawstuff_src} extern/drawstuff/src/x11.cpp)
	set(draw_lib ${X11_LIBRARIES} -lfreeimage )
endif()


file ( GLOB_RECURSE all_include include/*.hpp )
file ( GLOB all_sources src/*.cpp )

message(STATUS "ODE flags : " ${ODE_CFLAGS})

#-------------------
# Library Build
#-------------------
set(LIBRARY_NAME "ode-env")
rename_buildtype(LIBRARY_NAME)
add_library(${LIBRARY_NAME} SHARED ${all_include} ${all_sources} ${drawstuff_src})
target_link_libraries(${LIBRARY_NAME} ${ODE_LIBRARY} ${draw_lib} ${OPENGL_LIBRARIES} ${COMMON_DRL_LIBRARY_LIGHT})

