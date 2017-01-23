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

if(APPLE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -DGTEST_USE_OWN_TR1_TUPLE=1 ")
	set(CMAKE_EXE_LINKER_FLAGS "-stdlib=libc++")
endif()

#-------------------
# Global Config
#-------------------
set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/lib")
list(APPEND CMAKE_MODULE_PATH "${ROOT_DRL_PATH}/common/cmake/")

include(${ROOT_DRL_PATH}/common/cmake/Callable.cmake)

#-------------------
# Number of Thread
#-------------------
if(NOT DEFINED PROCESSOR_COUNT)
  # Unknown:
  set(PROCESSOR_COUNT 0)

  # Linux:
  set(cpuinfo_file "/proc/cpuinfo")
  if(EXISTS "${cpuinfo_file}")
    file(STRINGS "${cpuinfo_file}" procs REGEX "^processor.: [0-9]+$")
    list(LENGTH procs PROCESSOR_COUNT)
  endif()

  # Mac:
  if(APPLE)
    find_program(cmd_sys_pro "sysctl")
    if(cmd_sys_pro)
      execute_process(COMMAND ${cmd_sys_pro} machdep.cpu.thread_count OUTPUT_VARIABLE info)
      string(REGEX MATCH "[0-9]+" PROCESSOR_COUNT "${info}")
    endif()
  endif()

  # Windows:
  if(WIN32)
    set(PROCESSOR_COUNT "$ENV{NUMBER_OF_PROCESSORS}")
  endif()
endif()

##############
# Build dir
##############

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(build_dir_type "debug" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(build_dir_type "relwithdeb" CACHE STRING "")
else()
  set(build_dir_type "release" CACHE STRING "")
endif()
