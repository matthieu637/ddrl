cmake_minimum_required(VERSION 2.8)

#Project name
project(cmaes CXX C)

if(NOT DEFINED ROOT_DRL_PATH)
        set(ROOT_DRL_PATH "${CMAKE_SOURCE_DIR}/../..")
endif()

include(${ROOT_DRL_PATH}/common/cmake/Config.cmake)

#Boost library
find_package(CommonDRL)
find_package(AdvancedAcrobot)
find_package(Cartpole)
find_package(HalfCheetah)

include_directories(${CMAKE_SOURCE_DIR}/include ${COMMON_DRL_INCLUDE_DIRS} 
  ${ADVANCED_ACROBOT_INCLUDE_DIRS} ${CARTPOLE_INCLUDE_DIRS} ${HALF_CHEETAH_INCLUDE_DIRS})

set(BASE_LIBRARIES ${COMMON_DRL_LIBRARY} )
file ( GLOB_RECURSE all_include include/*.hpp include/*.h)
file ( GLOB all_sources src/*.cpp src/*.c )

add_executable(adacrobot src/test/Adacrobot.cpp ${all_include} ${all_sources} )
target_link_libraries(adacrobot ${BASE_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY})

add_executable(cartpole src/test/Cartpole.cpp ${all_include} ${all_sources} )
target_link_libraries(cartpole ${ADVANCED_ACROBOT_LIBRARY} ${CARTPOLE_LIBRARY} ${BASE_LIBRARIES} )

add_executable(half-cheetah src/test/HalfCheetah.cpp ${all_include} ${all_sources} )
target_link_libraries(half-cheetah ${ADVANCED_ACROBOT_LIBRARY} ${HALF_CHEETAH_LIBRARY} ${BASE_LIBRARIES} )

# add_executable(dpmt-adacrobot src/test/DpmtAdacrobot.cpp ${all_include} ${all_sources} )
# target_link_libraries(dpmt-adacrobot ${ADVANCED_ACROBOT_LIBRARY} ${BASE_LIBRARIES} )


file(COPY ${ROOT_DRL_PATH}/environment/advanced-acrobot/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY ${ROOT_DRL_PATH}/environment/cartpole/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/acrobot.ini data/base-acrobot.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/acrobot-3l.ini data/base-acrobot-3l.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/cartpole.ini data/base-cartpole.ini)

execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/base-acrobot.ini data/cmaes-acrobot.ini acrobot.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/base-acrobot-3l.ini data/cmaes-acrobot-3l.ini acrobot-3l.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/base-cartpole.ini data/cmaes-cartpole.ini cartpole.ini)

#
# -------------------
# Unit Test
# -------------------
#set(all_test_source ${all_sources} ${all_sources_sml})
#enable_utest(all_test_source ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES})
