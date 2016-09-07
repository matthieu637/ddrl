cmake_minimum_required(VERSION 2.8)

#Project name
project(qlearning-nn CXX)

if(NOT DEFINED ROOT_DRL_PATH)
        set(ROOT_DRL_PATH "${CMAKE_SOURCE_DIR}/../..")
endif()

include(${ROOT_DRL_PATH}/common/cmake/Config.cmake)

#Boost library
find_package(Boost COMPONENTS serialization filesystem system program_options thread REQUIRED)
find_package(CommonDRL)
find_package(AdvancedAcrobot)
find_package(TBB)
find_package(FANN)
find_package(OPTPP)

include_directories(${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} ${COMMON_DRL_INCLUDE_DIRS} ${ADVANCED_ACROBOT_INCLUDE_DIRS} ${FANN_INCLUDE_DIRS})

set(BASE_LIBRARIES ${COMMON_DRL_LIBRARY} ${TBB_LIBRARIES} ${DOUBLEFANN_LIBRARY} ${Boost_LIBRARIES} )

file ( GLOB_RECURSE all_include include/*.hpp )
file ( GLOB all_sources src/*.cpp )
file ( GLOB all_sources_sml src/sml/*.cpp )

if(${OPTPP_LIBRARY_FOUND})
	include_directories(${OPTPP_INCLUDE_DIRS})
	set(BASE_LIBRARIES ${BASE_LIBRARIES} ${OPTPP_LIBRARIES})
else()
	list(REMOVE_ITEM all_include ${CMAKE_CURRENT_SOURCE_DIR}/include/UNLF2.hpp)
	list(REMOVE_ITEM all_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/UNLF2.cpp)
endif()

add_executable(adacrobot-ql src/test/AdacrobotQLNoreplay.cpp ${all_include} ${all_sources_sml})
target_link_libraries(adacrobot-ql ${ADVANCED_ACROBOT_LIBRARY} ${BASE_LIBRARIES})

add_executable(adacrobot-cqfitted src/test/AdacrobotCQFitted.cpp ${all_include} ${all_sources} ${all_sources_sml})
target_link_libraries(adacrobot-cqfitted ${ADVANCED_ACROBOT_LIBRARY} ${BASE_LIBRARIES} )

file(COPY ${ROOT_DRL_PATH}/environment/advanced-acrobot/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/acrobot.ini config.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash config.ini data/qlearning.ini config.ini)

#-------------------
# Unit Test
#-------------------
set(all_test_source ${all_sources} ${all_sources_sml})
enable_utest(all_test_source ${ADVANCED_ACROBOT_LIBRARY} ${BASE_LIBRARIES})
