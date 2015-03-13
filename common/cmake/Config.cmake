#-------------------
# Build Type
#-------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message("Build type set to ${CMAKE_BUILD_TYPE}")

#-------------------
# GCC config
#-------------------
set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11 -Wno-switch") #-fPIC 
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -ffast-math -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O2 -g -DNDEBUG")

#-------------------
# Global Config
#-------------------
set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/lib")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../common/cmake/")

if(EXISTS ${CMAKE_SOURCE_DIR}/cmake)
  include(${CMAKE_SOURCE_DIR}/cmake/Callable.cmake)
else()
  include(${CMAKE_SOURCE_DIR}/../../common/cmake/Callable.cmake)
endif()