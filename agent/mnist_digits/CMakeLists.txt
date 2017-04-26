cmake_minimum_required(VERSION 2.8)

#Project name
project(mnist_digits CXX C)

if(NOT DEFINED ROOT_DRL_PATH)
        set(ROOT_DRL_PATH "${CMAKE_SOURCE_DIR}/../..")
endif()

include(${ROOT_DRL_PATH}/common/cmake/Config.cmake)

find_package(CommonDRL)

include_directories(${CMAKE_SOURCE_DIR}/include ${COMMON_DRL_INCLUDE_DIRS})

file ( GLOB_RECURSE all_include include/*.hpp include/*.h)
file ( GLOB all_sources src/*.cpp src/*.c)

add_executable(mnist_digits src/mnist_digits.cpp ${all_include} ${all_sources} )
target_link_libraries(mnist_digits ${COMMON_DRL_LIBRARY} )

add_custom_command(TARGET mnist_digits POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_SOURCE_DIR}/data ${CMAKE_CURRENT_BINARY_DIR})



#message(${COMMON_DRL_LIBRARY})
#enable_utest(all_sources ${COMMON_DRL_LIBRARY})
add_subdirectory(include)