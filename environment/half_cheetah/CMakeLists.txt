cmake_minimum_required(VERSION 2.8)

#Nom du projet
project(half_cheetah CXX C)

if(NOT DEFINED ROOT_DRL_PATH)
	set(ROOT_DRL_PATH "${CMAKE_SOURCE_DIR}/../..")
endif()

include(${ROOT_DRL_PATH}/common/cmake/Config.cmake)

find_package(Boost 1.60 COMPONENTS serialization filesystem system program_options thread REQUIRED)
find_package(ODE)
find_package(OpenGL)
find_package(X11)
find_package(TBB)
find_package(Threads)
find_package(CommonDRL)
find_package(AdvancedAcrobot)

include_directories(${CMAKE_SOURCE_DIR}/include ${ODE_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/extern/drawstuff/include/ ${TBB_INCLUDE_DIRS} ${COMMON_DRL_INCLUDE_DIRS} ${ADVANCED_ACROBOT_INCLUDE_DIRS})


file ( GLOB_RECURSE all_include include/*.hpp )
file ( GLOB all_sources src/*.cpp )

message(STATUS "ODE flags : " ${ODE_CFLAGS})

#-------------------
# Library Build
#-------------------
set(LIBRARY_NAME "half_cheetah")
rename_buildtype(LIBRARY_NAME)
add_library(${LIBRARY_NAME} SHARED ${all_include} ${all_sources})
message(${LIBRARY_NAME} ${ODE_LIBRARY} ${Boost_LIBRARIES} ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${COMMON_DRL_LIBRARY_LIGHT})
target_link_libraries(${LIBRARY_NAME} ${ODE_LIBRARY} ${Boost_LIBRARIES} ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${COMMON_DRL_LIBRARY_LIGHT} ${ADVANCED_ACROBOT_LIBRARY})

#-------------------
# Personnals Test
#-------------------
add_executable(half-cheetah-env src/test/HalfCheetah.cpp ${all_include} ${all_sources} ${drawstuff_src} )
target_link_libraries(half-cheetah-env ${ODE_LIBRARY} ${Boost_LIBRARIES} ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${COMMON_DRL_LIBRARY_LIGHT} ${ADVANCED_ACROBOT_LIBRARY})

file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY ${ROOT_DRL_PATH}/environment/half_cheetah/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/half_cheetah.ini config.ini)

#-------------------
# Unit Test
#-------------------
# set(all_test_source ${all_sources})
# enable_utest(all_test_source ${ODE_LIBRARY} ${Boost_LIBRARIES} ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${COMMON_DRL_LIBRARY_LIGHT})
