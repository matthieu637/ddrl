cmake_minimum_required(VERSION 2.8)

#Project name
project(deepqn CXX)

if(NOT DEFINED ROOT_DRL_PATH)
        set(ROOT_DRL_PATH "${CMAKE_SOURCE_DIR}/../..")
endif()

if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
        message("CUDA_TOOLKIT_ROOT_DIR not found or specified, set to /opt/cuda")
        set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda")
endif()

include(${ROOT_DRL_PATH}/common/cmake/Config.cmake)

#Boost library
find_package(Boost COMPONENTS serialization filesystem system program_options thread REQUIRED)
find_package(CommonDRL)
find_package(AdvancedAcrobot)
find_package(Cartpole)
find_package(CUDA)
find_package(GLOG)
find_package(Protobuf REQUIRED)
find_package(Caffe)

include_directories(${CMAKE_SOURCE_DIR}/include 
  ${Boost_INCLUDE_DIRS} ${FANN_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${GLOG_INCLUDE_DIRS}
  ${COMMON_DRL_INCLUDE_DIRS} ${ADVANCED_ACROBOT_INCLUDE_DIRS} ${CARTPOLE_INCLUDE_DIRS} )


file ( GLOB_RECURSE all_include include/*.hpp include/*.h)
file ( GLOB all_sources src/*.cpp src/*.c)

add_executable(adacrobot src/test/Adacrobot.cpp ${all_include} ${all_sources} )
target_link_libraries(adacrobot 
  ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY} 
  ${Boost_LIBRARIES} ${CUDA_LIBRARIES} ${PROTOBUF_LIBRARIES} ${GLOG_LIBRARIES} ${CAFFE_LIBRARY} )

file(COPY ${ROOT_DRL_PATH}/environment/advanced-acrobot/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
#file(COPY ${ROOT_DRL_PATH}/environment/cartpole/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/acrobot.ini data/base-acrobot.ini)
#execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/acrobot-3l.ini data/base-acrobot-3l.ini)
#execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/cartpole.ini data/base-cartpole.ini)

#execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/base-acrobot.ini data/cmaes-acrobot.ini acrobot.ini)
#execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/base-acrobot-3l.ini data/cmaes-acrobot-3l.ini acrobot-3l.ini)
#execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/base-cartpole.ini data/cmaes-cartpole.ini cartpole.ini)

#
# -------------------
# Unit Test
# -------------------
#set(all_test_source ${all_sources} ${all_sources_sml})
#enable_utest(all_test_source ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES})