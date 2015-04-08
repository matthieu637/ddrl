cmake_minimum_required(VERSION 2.8)

#Project name
project(qlearning-nn CXX)

include(${CMAKE_SOURCE_DIR}/../../common/cmake/Config.cmake)

#Boost library
find_package(Boost COMPONENTS serialization filesystem system program_options thread REQUIRED)
find_package(CommonDRL)
find_package(AdvancedAcrobot)
find_package(TBB)
find_package(FANN)
find_package(OPTPP)

include_directories(${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} ${COMMON_DRL_INCLUDE_DIRS} ${ADVANCED_ACROBOT_INCLUDE_DIRS} ${FANN_INCLUDE_DIRS} ${OPTPP_INCLUDE_DIRS})


file ( GLOB_RECURSE all_include include/*.hpp )
file ( GLOB all_sources src/*.cpp )
file ( GLOB all_sources_sml src/sml/*.cpp )


add_executable(advanced-acrobot src/test/AdvancedAcrobotDiscret.cpp ${all_include} ${all_sources_sml})
target_link_libraries(advanced-acrobot ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES})

add_executable(adacrobot-cn src/test/AdvancedAcrobotContinuous.cpp ${all_include} ${all_sources} ${all_sources_sml})
target_link_libraries(adacrobot-cn ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES} )

file(COPY ../../environment/advanced-acrobot/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

execute_process(COMMAND bash ../../../../scripts/merge_ini.bash ../../../../common/data/minimal_config.ini data/acrobot.ini config.ini)
execute_process(COMMAND bash ../../../../scripts/merge_ini.bash config.ini data/qlearning.ini config.ini)

#-------------------
# Unit Test
#-------------------
set(all_test_source ${all_sources} ${all_sources_sml})
enable_utest(all_test_source ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES})
