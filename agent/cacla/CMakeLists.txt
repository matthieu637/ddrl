cmake_minimum_required(VERSION 2.8)

#Project name
project(cacla CXX)

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

include_directories(${CMAKE_SOURCE_DIR}/include ../qlearning-nn/include ${Boost_INCLUDE_DIRS} ${COMMON_DRL_INCLUDE_DIRS} ${ADVANCED_ACROBOT_INCLUDE_DIRS} ${FANN_INCLUDE_DIRS} ${OPTPP_INCLUDE_DIRS})


file ( GLOB_RECURSE all_include include/*.hpp ${ROOT_DRL_PATH}/agent/qlearning-nn/include/MLP.hpp ${ROOT_DRL_PATH}/agent/qlearning-nn/include/UNLF2.hpp )
file ( GLOB all_sources src/*.cpp ${ROOT_DRL_PATH}/agent/qlearning-nn/src/MLP.cpp ${ROOT_DRL_PATH}/agent/qlearning-nn/src/UNLF2.cpp )

add_executable(adacrobot-off src/test/AdacrobotOffline.cpp ${all_include} ${all_sources} )
target_link_libraries(adacrobot-off ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES} )

add_executable(adacrobot-on src/test/AdacrobotOnline.cpp ${all_include} ${all_sources} )
target_link_libraries(adacrobot-on ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES} )


file(COPY ${ROOT_DRL_PATH}/environment/advanced-acrobot/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/acrobot.ini config.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash config.ini data/cacla.ini config.ini)

#-------------------
# Unit Test
#-------------------
#set(all_test_source ${all_sources} ${all_sources_sml})
#enable_utest(all_test_source ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES})