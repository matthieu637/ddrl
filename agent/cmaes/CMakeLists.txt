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
find_package(Humanoid)

include_directories(${CMAKE_SOURCE_DIR}/include ${COMMON_DRL_INCLUDE_DIRS} 
  ${ADVANCED_ACROBOT_INCLUDE_DIRS} ${CARTPOLE_INCLUDE_DIRS} 
  ${HALF_CHEETAH_INCLUDE_DIRS} ${HUMANOID_INCLUDE_DIRS})

set(BASE_LIBRARIES ${COMMON_DRL_LIBRARY} )
file ( GLOB_RECURSE all_include include/*.hpp include/*.h)
file ( GLOB all_sources src/*.cpp src/*.c )

add_executable(adacrobot src/test/Adacrobot.cpp ${all_include} ${all_sources})
target_link_libraries(adacrobot ${ADVANCED_ACROBOT_LIBRARY} ${BASE_LIBRARIES})

add_executable(cartpole src/test/Cartpole.cpp ${all_include} ${all_sources})
target_link_libraries(cartpole ${CARTPOLE_LIBRARY} ${BASE_LIBRARIES})

add_executable(half-cheetah src/test/HalfCheetah.cpp ${all_include} ${all_sources} )
target_link_libraries(half-cheetah ${HALF_CHEETAH_LIBRARY} ${BASE_LIBRARIES} )

add_executable(random-half-cheetah src/test/RandomHC.cpp ${all_include} ${all_sources} )
target_link_libraries(random-half-cheetah ${HALF_CHEETAH_LIBRARY} ${BASE_LIBRARIES} )

add_executable(learn-pid-hf src/test/LearnPIDHF.cpp ${all_include} ${all_sources} )
target_link_libraries(learn-pid-hf ${HALF_CHEETAH_LIBRARY} ${BASE_LIBRARIES} )


add_executable(humanoid src/test/Humanoid.cpp ${all_include} ${all_sources} )
target_link_libraries(humanoid ${HUMANOID_LIBRARY} ${BASE_LIBRARIES} )

add_executable(dpmt-half-cheetah src/test/DpmtHalfCheetah.cpp ${all_include} ${all_sources} )
target_link_libraries(dpmt-half-cheetah ${HALF_CHEETAH_LIBRARY} ${BASE_LIBRARIES} )

add_executable(dodev-half-cheetah src/test/DODevHalfCheetah.cpp ${all_include} ${all_sources} )
target_link_libraries(dodev-half-cheetah ${HALF_CHEETAH_LIBRARY} ${BASE_LIBRARIES} )

add_executable(dodev-humanoid src/test/DODevHumanoid.cpp ${all_include} ${all_sources} )
target_link_libraries(dodev-humanoid ${HUMANOID_LIBRARY} ${BASE_LIBRARIES} )


file(COPY ${ROOT_DRL_PATH}/environment/ode-env/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY ${ROOT_DRL_PATH}/environment/advanced-acrobot/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY ${ROOT_DRL_PATH}/environment/cartpole/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY ${ROOT_DRL_PATH}/environment/half_cheetah/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY ${ROOT_DRL_PATH}/environment/humanoid/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash 
	${ROOT_DRL_PATH}/common/data/minimal_config.ini 
	${CMAKE_CURRENT_BINARY_DIR}/data/half_cheetah.ini 
	${CMAKE_CURRENT_BINARY_DIR}/data/base-half_cheetah.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash 
	${ROOT_DRL_PATH}/common/data/minimal_config.ini
	${CMAKE_CURRENT_BINARY_DIR}/data/humanoid.ini 
	${CMAKE_CURRENT_BINARY_DIR}/data/base-humanoid.ini)

execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash
	${CMAKE_CURRENT_BINARY_DIR}/data/base-half_cheetah.ini
	${CMAKE_CURRENT_BINARY_DIR}/data/cmaes-half_cheetah.ini
	${CMAKE_CURRENT_BINARY_DIR}/cmaes-half_cheetah.ini)
execute_process(COMMAND bash ${ROOT_DRL_PATH}/scripts/merge_ini.bash
	${CMAKE_CURRENT_BINARY_DIR}/data/base-humanoid.ini
	${CMAKE_CURRENT_BINARY_DIR}/data/cmaes-humanoid.ini
	${CMAKE_CURRENT_BINARY_DIR}/cmaes-humanoid.ini)

#
# -------------------
# Unit Test
# -------------------
#set(all_test_source ${all_sources} ${all_sources_sml})
#enable_utest(all_test_source ${OPTPP_LIBRARIES} ${ADVANCED_ACROBOT_LIBRARY} ${COMMON_DRL_LIBRARY} ${DOUBLEFANN_LIBRARY} ${TBB_LIBRARIES} ${Boost_LIBRARIES})
