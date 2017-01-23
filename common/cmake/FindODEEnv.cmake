find_package(ODE)

message(STATUS "ODE flags : " ${ODE_CFLAGS})

set(ODE_ENV_INCLUDE_DIRS ${ROOT_DRL_PATH}/environment/ode-env/include ${ROOT_DRL_PATH}/environment/ode-env/extern/drawstuff/include)

set(ODE_ENV_NAME "ode-env")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(ODE_ENV_NAME "${ODE_ENV_NAME}-d")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(ODE_ENV_NAME "${ODE_ENV_NAME}-rd")
endif()

find_library( ODE_ENV_LIBRARY
  NAMES ${ODE_ENV_NAME}
  PATHS
  "${ROOT_DRL_PATH}/environment/ode-env/lib"
)

set(ODE_ENV_INCLUDE_DIRS ${ODE_ENV_INCLUDE_DIRS} ${ODE_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ODEEnv DEFAULT_MSG ODE_ENV_LIBRARY ODE_ENV_INCLUDE_DIRS)
mark_as_advanced(ODE_ENV_INCLUDE_DIRS ODE_ENV_LIBRARY )

set(ODE_ENV_LIBRARY ${ODE_ENV_LIBRARY} ${ODE_LIBRARY})
