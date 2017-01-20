cmake_minimum_required(VERSION 2.8)

#Nom du projet
project(advanced-acrobot CXX C)

if(NOT DEFINED ROOT_DRL_PATH)
	set(ROOT_DRL_PATH "${CMAKE_SOURCE_DIR}/../..")
endif()

include(${ROOT_DRL_PATH}/common/cmake/Config.cmake)

find_package(CommonDRL)
find_package(ODEEnv)

include_directories(${CMAKE_SOURCE_DIR}/include ${COMMON_DRL_INCLUDE_DIRS} ${ODE_ENV_INCLUDE_DIRS})


file ( GLOB_RECURSE all_include include/*.hpp )
file ( GLOB all_sources src/*.cpp )

message(STATUS "ODE flags : " ${ODE_CFLAGS})

#-------------------
# Library Build
#-------------------
set(LIBRARY_NAME "advanced-acrobot")
rename_buildtype(LIBRARY_NAME)
add_library(${LIBRARY_NAME} SHARED ${all_include} ${all_sources})
target_link_libraries(${LIBRARY_NAME} ${COMMON_DRL_LIBRARY_LIGHT} ${ODE_ENV_LIBRARY})

#-------------------
# Personnals Test
#-------------------
add_executable(internal-visu src/test/InternalView.cpp ${all_include} ${all_sources})
target_link_libraries(internal-visu ${COMMON_DRL_LIBRARY} ${ODE_ENV_LIBRARY})

add_executable(multi-instance src/test/MultiInstance.cpp ${all_include} ${all_sources})
target_link_libraries(multi-instance ${COMMON_DRL_LIBRARY} ${ODE_ENV_LIBRARY})

add_executable(acrobot-env src/test/AcrobotEnv.cpp ${all_include} ${all_sources})
target_link_libraries(acrobot-env ${COMMON_DRL_LIBRARY} ${ODE_ENV_LIBRARY})

file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/acrobot.ini config.ini)

#-------------------
# Unit Test
#-------------------
set(all_test_source ${all_sources})
enable_utest(all_test_source ${COMMON_DRL_LIBRARY} ${ODE_ENV_LIBRARY})

# ALAIN => target etags/tags
IF (UNIX)
  ADD_CUSTOM_TARGET(tags etags --members --declarations  `find . -name 
*.cc -or -name *.hh -or -name *.cpp -or -name *.h -or -name *.c -or 
-name *.f`)
  ADD_CUSTOM_TARGET(etags DEPENDS tags)
ENDIF (UNIX)

