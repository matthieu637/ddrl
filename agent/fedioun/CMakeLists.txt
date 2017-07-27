cmake_minimum_required(VERSION 2.8)

#Project name
project(fedioun CXX C)

if(NOT DEFINED ROOT_DRL_PATH)
        set(ROOT_DRL_PATH "${CMAKE_SOURCE_DIR}/../..")
endif()

include(${ROOT_DRL_PATH}/common/cmake/Config.cmake)

find_package(CommonDRL)
find_package(AdvancedAcrobot)
find_package(Cartpole)
find_package(HalfCheetah)
find_package(Humanoid)

include_directories(${CMAKE_SOURCE_DIR}/include ${COMMON_DRL_INCLUDE_DIRS} ${ADVANCED_ACROBOT_INCLUDE_DIRS} ${CARTPOLE_INCLUDE_DIRS} ${HALF_CHEETAH_INCLUDE_DIRS}
  ${HUMANOID_INCLUDE_DIRS})

file ( GLOB_RECURSE all_include include/*.hpp include/*.h)
file ( GLOB all_sources src/*.cpp src/*.c)

add_executable(denfac src/test/denfac.cpp ${all_include} ${all_sources} )
target_link_libraries(denfac ${COMMON_DRL_LIBRARY} ${HALF_CHEETAH_LIBRARY})

add_executable(adenfac-cheetah src/test/adenfac-cheetah.cpp ${all_include} ${all_sources} )
target_link_libraries(adenfac-cheetah ${COMMON_DRL_LIBRARY} ${HALF_CHEETAH_LIBRARY})


add_executable(adenfac-acrobot src/test/adenfac-acrobot.cpp ${all_include} ${all_sources} )
target_link_libraries(adenfac-acrobot ${COMMON_DRL_LIBRARY} ${ADVANCED_ACROBOT_LIBRARY})

add_executable(adenfac-humanoid src/test/adenfac-humanoid.cpp ${all_include} ${all_sources} )
target_link_libraries(adenfac-humanoid ${COMMON_DRL_LIBRARY} ${HUMANOID_LIBRARY})

add_executable(adenfac-cartpole src/test/adenfac-cartpole.cpp ${all_include} ${all_sources} )
target_link_libraries(adenfac-cartpole ${COMMON_DRL_LIBRARY} ${CARTPOLE_LIBRARY})

add_custom_command(TARGET denfac POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_SOURCE_DIR}/data ${CMAKE_CURRENT_BINARY_DIR})

add_custom_command(TARGET adenfac-cheetah POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_SOURCE_DIR}/data ${CMAKE_CURRENT_BINARY_DIR})


#message(${COMMON_DRL_LIBRARY})
#enable_utest(all_sources ${COMMON_DRL_LIBRARY})
#add_subdirectory(include)
