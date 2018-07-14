cmake_minimum_required(VERSION 2.8)

#Project name
project(cacla CXX C)

if(NOT DEFINED ROOT_DRL_PATH)
        set(ROOT_DRL_PATH "${CMAKE_SOURCE_DIR}/../..")
endif()

include(${ROOT_DRL_PATH}/common/cmake/Config.cmake)

#Boost library
find_package(Boost COMPONENTS serialization filesystem system program_options thread REQUIRED)
find_package(CommonDRL)
find_package(AdvancedAcrobot)
find_package(Cartpole)
find_package(HalfCheetah)
find_package(Humanoid)
# find_package(TBB)
# find_package(OPTPP)
find_package(Threads REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} ${COMMON_DRL_INCLUDE_DIRS} 
  ${ADVANCED_ACROBOT_INCLUDE_DIRS} ${CARTPOLE_INCLUDE_DIRS} ${HALF_CHEETAH_INCLUDE_DIRS}
  ${HUMANOID_INCLUDE_DIRS})

set(BASE_LIBRARIES ${COMMON_DRL_LIBRARY} ${Boost_LIBRARIES} )
file ( GLOB_RECURSE all_include include/*.hpp)
#file ( GLOB all_sources src/*.cpp)
set(all_sources "")

if(${OPTPP_LIBRARY_FOUND})
	include_directories(${OPTPP_INCLUDE_DIRS})
	set(BASE_LIBRARIES ${BASE_LIBRARIES} ${OPTPP_LIBRARIES} ${TBB_LIBRARIES})
	set(all_include ${all_include} ${ROOT_DRL_PATH}/agent/qlearning-nn/include/UNLF2.hpp)
	set(all_sources ${all_sources} ${ROOT_DRL_PATH}/agent/qlearning-nn/src/UNLF2.cpp)
endif()

add_executable(adacrobot-off src/test/AdacrobotOffline.cpp ${all_include} ${all_sources} )
target_link_libraries(adacrobot-off ${ADVANCED_ACROBOT_LIBRARY} ${BASE_LIBRARIES} )

add_executable(adacrobot-on src/test/AdacrobotOnline.cpp ${all_include} ${all_sources} )
target_link_libraries(adacrobot-on ${ADVANCED_ACROBOT_LIBRARY} ${BASE_LIBRARIES} )

add_executable(cp-cacla-on src/test/CartpoleCacOn.cpp ${all_include} ${all_sources} )
target_link_libraries(cp-cacla-on ${CARTPOLE_LIBRARY} ${BASE_LIBRARIES} )

add_executable(cp-cacla-off src/test/CartpoleCacOff.cpp ${all_include} ${all_sources} )
target_link_libraries(cp-cacla-off ${CARTPOLE_LIBRARY} ${BASE_LIBRARIES} )

add_executable(hf-cacla-on src/test/HalfCheetahCacOn.cpp ${all_include} ${all_sources} )
target_link_libraries(hf-cacla-on ${HALF_CHEETAH_LIBRARY} ${BASE_LIBRARIES} )

add_executable(hf-caclatd-on src/test/HalfCheetahCacTDOn.cpp ${all_include} ${all_sources} )
target_link_libraries(hf-caclatd-on ${HALF_CHEETAH_LIBRARY} ${BASE_LIBRARIES} )

add_executable(hf-pac src/test/HalfCheetahOnPAC.cpp ${all_include} ${all_sources} )
target_link_libraries(hf-pac ${HALF_CHEETAH_LIBRARY} ${BASE_LIBRARIES} )

add_executable(hf-adpac src/test/HalfCheetahAdPAC.cpp ${all_include} ${all_sources} )
target_link_libraries(hf-adpac ${HALF_CHEETAH_LIBRARY} ${BASE_LIBRARIES} )

add_executable(hf-cacla-off src/test/HalfCheetahCacOff.cpp ${all_include} ${all_sources} )
target_link_libraries(hf-cacla-off ${HALF_CHEETAH_LIBRARY} ${BASE_LIBRARIES} )

add_executable(hf-nfac-off src/test/HalfCheetahOffNFAC.cpp ${all_include} ${all_sources} )
target_link_libraries(hf-nfac-off ${HALF_CHEETAH_LIBRARY} ${BASE_LIBRARIES} )

add_executable(hf-nfacvmem src/test/HalfCheetahNFACVMem.cpp ${all_include} ${all_sources} )
target_link_libraries(hf-nfacvmem ${HALF_CHEETAH_LIBRARY} ${BASE_LIBRARIES} )

add_executable(hf-nfacq-off src/test/HalfCheetahOffNFACQ.cpp ${all_include} ${all_sources} )
target_link_libraries(hf-nfacq-off ${HALF_CHEETAH_LIBRARY} ${BASE_LIBRARIES} )

add_executable(hf-cacla-off2ph src/test/HalfCheetahCacOff2Ph.cpp ${all_include} ${all_sources} )
target_link_libraries(hf-cacla-off2ph ${HALF_CHEETAH_LIBRARY} ${BASE_LIBRARIES} )

add_executable(hf-cacla-off-dodev src/test/HalfCheetahCacOffDodev.cpp ${all_include} ${all_sources} )
target_link_libraries(hf-cacla-off-dodev ${HALF_CHEETAH_LIBRARY} ${BASE_LIBRARIES} )

add_executable(hum-cacla-off src/test/HumanoidCacOff.cpp ${all_include} ${all_sources} )
target_link_libraries(hum-cacla-off ${HUMANOID_LIBRARY} ${BASE_LIBRARIES} )

add_executable(hum-cacla-off-dodev src/test/HumanoidCacOffDodev.cpp ${all_include} ${all_sources} )
target_link_libraries(hum-cacla-off-dodev ${HUMANOID_LIBRARY} ${BASE_LIBRARIES} )

add_executable(hum-nfacq-off src/test/HumanoidNFACQ.cpp ${all_include} ${all_sources} )
target_link_libraries(hum-nfacq-off ${HUMANOID_LIBRARY} ${BASE_LIBRARIES} )

set(LIBRARY_NAME "ddrl-cacla")
rename_buildtype(LIBRARY_NAME)
add_library(${LIBRARY_NAME} SHARED ${all_include} src/openai.cpp)
target_link_libraries(${LIBRARY_NAME} ${BASE_LIBRARIES} )

# add_executable(ada-nfac src/test/AdaNFAC.cpp ${all_include} ${all_sources} )
# target_link_libraries(ada-nfac ${ADVANCED_ACROBOT_LIBRARY} ${CARTPOLE_LIBRARY} ${BASE_LIBRARIES} rt ${CMAKE_THREAD_LIBS_INIT})

file(COPY ${ROOT_DRL_PATH}/environment/advanced-acrobot/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY ${ROOT_DRL_PATH}/environment/cartpole/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash ${ROOT_DRL_PATH}/common/data/minimal_config.ini data/acrobot.ini data/base-ada.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash data/base-ada.ini data/base-cacla-off.ini ada-cacla-off.ini)
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
enable_utest(all_test_source ${ADVANCED_ACROBOT_LIBRARY} ${BASE_LIBRARIES})
