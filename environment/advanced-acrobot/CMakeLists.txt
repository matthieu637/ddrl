cmake_minimum_required(VERSION 2.8)

#Nom du projet
project(advanced-acrobot CXX C)

if(NOT DEFINED ROOT_DRL_PATH)
	set(ROOT_DRL_PATH "${CMAKE_SOURCE_DIR}/../..")
endif()

include(${ROOT_DRL_PATH}/common/cmake/Config.cmake)

find_package(Boost COMPONENTS serialization filesystem system program_options thread REQUIRED)
find_package(ODE)
find_package(OpenGL)
find_package(X11)
find_package(TBB)
find_package(Threads)
find_package(CommonDRL)

include_directories(${CMAKE_SOURCE_DIR}/include ${ODE_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/extern/drawstuff/include/ ${TBB_INCLUDE_DIRS} ${COMMON_DRL_INCLUDE_DIRS})


file ( GLOB_RECURSE drawstuff_src extern/drawstuff/src/*.cpp)

file ( GLOB_RECURSE all_include include/*.hpp )
file ( GLOB all_sources src/*.cpp )

message(STATUS "ODE flags : " ${ODE_CFLAGS})

#-------------------
# Library Build
#-------------------
set(LIBRARY_NAME "advanced-acrobot")
rename_buildtype(LIBRARY_NAME)
add_library(${LIBRARY_NAME} SHARED ${all_include} ${all_sources} ${drawstuff_src})
message(${LIBRARY_NAME} ${ODE_LIBRARY} ${Boost_LIBRARIES} ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${COMMON_DRL_LIBRARY})
target_link_libraries(${LIBRARY_NAME} ${ODE_LIBRARY} ${Boost_LIBRARIES} ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${COMMON_DRL_LIBRARY_LIGHT})

#-------------------
# Personnals Test
#-------------------
add_executable(internal-visu src/test/InternalView.cpp ${all_include} ${all_sources} ${drawstuff_src})
target_link_libraries(internal-visu ${ODE_LIBRARY} ${Boost_LIBRARIES} ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${COMMON_DRL_LIBRARY})

add_executable(multi-instance src/test/MultiInstance.cpp ${all_include} ${all_sources} ${drawstuff_src} )
target_link_libraries(multi-instance ${ODE_LIBRARY} ${Boost_LIBRARIES} ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${COMMON_DRL_LIBRARY})

add_executable(acrobot-env src/test/AcrobotEnv.cpp ${all_include} ${all_sources} ${drawstuff_src} )
target_link_libraries(acrobot-env ${ODE_LIBRARY} ${Boost_LIBRARIES} ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${COMMON_DRL_LIBRARY})

file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/acrobot.ini config.ini)

#-------------------
# Unit Test
#-------------------
set(all_test_source ${all_sources} ${drawstuff_src})
enable_utest(all_test_source ${ODE_LIBRARY} ${Boost_LIBRARIES} ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${COMMON_DRL_LIBRARY})

# ALAIN => target etags/tags
IF (UNIX)
  ADD_CUSTOM_TARGET(tags etags --members --declarations  `find . -name 
*.cc -or -name *.hh -or -name *.cpp -or -name *.h -or -name *.c -or 
-name *.f`)
  ADD_CUSTOM_TARGET(etags DEPENDS tags)
ENDIF (UNIX)

