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
find_package(Cartpole)
find_package(TBB)
find_package(FANN)
find_package(OPTPP)
find_package(Threads REQUIRED)
# find_package(FLANN)

include_directories(${CMAKE_SOURCE_DIR}/include ../qlearning-nn/include ${Boost_INCLUDE_DIRS} ${COMMON_DRL_INCLUDE_DIRS} ${ADVANCED_ACROBOT_INCLUDE_DIRS} ${CARTPOLE_INCLUDE_DIRS} ${FANN_INCLUDE_DIRS} ${OPTPP_INCLUDE_DIRS} )#  ${FLANN_INCLUDE_DIRS}


file ( GLOB_RECURSE all_include include/*.hpp ${ROOT_DRL_PATH}/agent/qlearning-nn/include/MLP.hpp ${ROOT_DRL_PATH}/agent/qlearning-nn/include/UNLF2.hpp )
file ( GLOB all_sources src/*.cpp ${ROOT_DRL_PATH}/agent/qlearning-nn/src/MLP.cpp ${ROOT_DRL_PATH}/agent/qlearning-nn/src/UNLF2.cpp )

add_executable(adacrobot-off src/test/AdacrobotOffline.cpp ${all_include} ${all_sources} )
target_link_libraries(adacrobot-off ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${CARTPOLE_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES} )

add_executable(adacrobot-on src/test/AdacrobotOnline.cpp ${all_include} ${all_sources} )
target_link_libraries(adacrobot-on ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES} )

add_executable(cp-cacla-on src/test/CartpoleCacOn.cpp ${all_include} ${all_sources} )
target_link_libraries(cp-cacla-on ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${CARTPOLE_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES} )

add_executable(cp-cacla-off src/test/CartpoleCacOff.cpp ${all_include} ${all_sources} )
target_link_libraries(cp-cacla-off ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${CARTPOLE_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES} )

# add_executable(ada-fitqac src/test/AdaQFittedAC.cpp ${all_include} ${all_sources} )
# target_link_libraries(ada-fitqac ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${CARTPOLE_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES} rt ${CMAKE_THREAD_LIBS_INIT})
# 
add_executable(ada-polset src/test/AdaPolset.cpp ${all_include} ${all_sources} )
target_link_libraries(ada-polset ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES} rt ${CMAKE_THREAD_LIBS_INIT})

add_executable(ada-polset-genvf src/test/AdaPolsetGenVF.cpp ${all_include} ${all_sources} )
target_link_libraries(ada-polset-genvf ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES} rt ${CMAKE_THREAD_LIBS_INIT})

add_executable(cp-polset src/test/CartpolePolset.cpp ${all_include} ${all_sources} )
target_link_libraries(cp-polset ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${CARTPOLE_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES} rt ${CMAKE_THREAD_LIBS_INIT})

add_executable(cp-polset-genvf src/test/CartpolePolsetGenVF.cpp ${all_include} ${all_sources} )
target_link_libraries(cp-polset-genvf ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${CARTPOLE_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES} rt ${CMAKE_THREAD_LIBS_INIT})

add_executable(ada-vset src/test/AdaVset.cpp ${all_include} ${all_sources} )
target_link_libraries(ada-vset ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES} rt ${CMAKE_THREAD_LIBS_INIT})

add_executable(cp-vset src/test/CartpoleVset.cpp ${all_include} ${all_sources} )
target_link_libraries(cp-vset ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${CARTPOLE_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES} rt ${CMAKE_THREAD_LIBS_INIT})

file(COPY ${ROOT_DRL_PATH}/environment/advanced-acrobot/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY ${ROOT_DRL_PATH}/environment/cartpole/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/acrobot.ini data/base-ada.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/base-ada.ini data/base-cacla-off.ini ada-cacla-off.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/base-ada.ini data/base-cacla-off-on.ini ada-cacla-off-on.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/base-ada.ini data/base-cacla-on.ini ada-cacla-on.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/base-ada.ini data/base-cacla-off-3d.ini ada-cacla-off-3d.ini)

execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/cp-cacla-off.ini cp-cacla-off.ini)


execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/acrobot.ini data/env-acrobot.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/cartpole.ini data/env-cartpole.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/env-acrobot.ini data/ada-polset.ini ada-polset.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/env-cartpole.ini data/cp-polset.ini cp-polset.ini)

execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/env-acrobot.ini data/ada-vset.ini ada-vset.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/env-cartpole.ini data/cp-vset.ini cp-vset.ini)

# -------------------
# Unit Test
# -------------------
set(all_test_source ${all_sources} ${all_sources_sml})
enable_utest(all_test_source ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES})
