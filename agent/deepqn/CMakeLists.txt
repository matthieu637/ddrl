cmake_minimum_required(VERSION 2.8)

#Project name
project(deepqn CXX C)

if(NOT DEFINED ROOT_DRL_PATH)
        set(ROOT_DRL_PATH "${CMAKE_SOURCE_DIR}/../..")
endif()

include(${ROOT_DRL_PATH}/common/cmake/Config.cmake)

find_package(CommonDRL)
if(NOT PYTHON_ONLY)
  find_package(AdvancedAcrobot)
  find_package(Cartpole)
  find_package(HalfCheetah)
  find_package(Humanoid)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include ${COMMON_DRL_INCLUDE_DIRS}
  ${ADVANCED_ACROBOT_INCLUDE_DIRS} ${CARTPOLE_INCLUDE_DIRS}
  ${HALF_CHEETAH_INCLUDE_DIRS} ${HUMANOID_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/../cacla/include/)

file ( GLOB_RECURSE all_include include/*.hpp include/*.h)
file ( GLOB all_sources src/*.cpp src/*.c)


if(NOT PYTHON_ONLY)
  add_executable(adacrobot src/test/Adacrobot.cpp ${all_include} ${all_sources} )
  target_link_libraries(adacrobot 
    ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY})
    
  # add_executable(ada-nfac src/test/NFACAda.cpp ${all_include} ${all_sources} )
  # target_link_libraries(ada-nfac 
  #   ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY})
  
  add_executable(cartpole src/test/Cartpole.cpp ${all_include} ${all_sources} )
  target_link_libraries(cartpole 
    ${CARTPOLE_LIBRARY} ${COMMON_DRL_LIBRARY} )
    
  add_executable(half-cheetah-dpmt src/test/DpmtHalfCheetah.cpp ${all_include} ${all_sources} )
  target_link_libraries(half-cheetah-dpmt 
    ${HALF_CHEETAH_LIBRARY} ${COMMON_DRL_LIBRARY} )
  
  add_executable(half-cheetah-dodev src/test/DodevHalfCheetah.cpp ${all_include} ${all_sources} )
  target_link_libraries(half-cheetah-dodev 
    ${HALF_CHEETAH_LIBRARY} ${COMMON_DRL_LIBRARY} )
    
  # add_executable(cp-nfac src/test/NFACCp.cpp ${all_include} ${all_sources} )
  # target_link_libraries(cp-nfac
  #   ${CARTPOLE_LIBRARY} ${COMMON_DRL_LIBRARY} )
    
  add_executable(half-cheetah src/test/HalfCheetah.cpp ${all_include} ${all_sources} )
  target_link_libraries(half-cheetah 
    ${HALF_CHEETAH_LIBRARY} ${COMMON_DRL_LIBRARY} )
    
  add_executable(half-cheetah-dqc src/test/HalfCheetahDQC.cpp ${all_include} ${all_sources} )
  target_link_libraries(half-cheetah-dqc 
    ${HALF_CHEETAH_LIBRARY} ${COMMON_DRL_LIBRARY} )
  
  # add_executable(half-cheetah-nfac src/test/NFACHalfCheetah.cpp ${all_include} ${all_sources} )
  # target_link_libraries(half-cheetah-nfac
  #   ${HALF_CHEETAH_LIBRARY} ${COMMON_DRL_LIBRARY} )
    
  # add_executable(half-cheetah-nfmac src/test/NFMACHalfCheetah.cpp ${all_include} ${all_sources} )
  # target_link_libraries(half-cheetah-nfmac
  #   ${HALF_CHEETAH_LIBRARY} ${COMMON_DRL_LIBRARY} )
    
  add_executable(humanoid src/test/Humanoid.cpp ${all_include} ${all_sources} )
  target_link_libraries(humanoid 
    ${HUMANOID_LIBRARY} ${COMMON_DRL_LIBRARY} )
    
  add_executable(humanoid-dodev src/test/DodevHumanoid.cpp ${all_include} ${all_sources} )
  target_link_libraries(humanoid-dodev 
    ${HUMANOID_LIBRARY} ${COMMON_DRL_LIBRARY} )
endif()

set(LIBRARY_NAME "ddrl-ddpg")
rename_buildtype(LIBRARY_NAME)
add_library(${LIBRARY_NAME} SHARED ${all_include} src/pybinding/ddpg.cpp)
target_link_libraries(${LIBRARY_NAME} ${COMMON_DRL_LIBRARY} )

set(LIBRARY_NAME2 "ddrl-foo")
rename_buildtype(LIBRARY_NAME2)
add_library(${LIBRARY_NAME2} SHARED ${all_include} src/pybinding/foo.cpp)
target_link_libraries(${LIBRARY_NAME2} ${COMMON_DRL_LIBRARY} )

set(LIBRARY_NAME3 "ddrl-td3")
rename_buildtype(LIBRARY_NAME3)
add_library(${LIBRARY_NAME3} SHARED ${all_include} src/pybinding/ddpg.cpp)
target_compile_definitions(${LIBRARY_NAME3} PUBLIC WANT_TD3)
target_link_libraries(${LIBRARY_NAME3} ${COMMON_DRL_LIBRARY} )

file(COPY ${ROOT_DRL_PATH}/environment/advanced-acrobot/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY ${ROOT_DRL_PATH}/environment/cartpole/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/acrobot.ini data/base-acrobot.ini)
#execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/acrobot-3l.ini data/base-acrobot-3l.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/cartpole.ini data/base-cartpole.ini)

execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/base-acrobot.ini data/ddpg-ada1d.ini acrobot-ddpg.ini)
#execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/base-acrobot-3l.ini data/cmaes-acrobot-3l.ini acrobot-3l.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/base-cartpole.ini data/ddpg-cp.ini cartpole-ddpg.ini)

# execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/base-acrobot.ini data/nfac-ada.ini acrobot-nfac.ini)
#
# -------------------
# Unit Test
# -------------------
#set(all_test_source ${all_sources} ${all_sources_sml})
#enable_utest(all_test_source ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES})
