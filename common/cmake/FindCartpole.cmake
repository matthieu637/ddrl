
set(CARTPOLE_INCLUDE_DIRS ${ROOT_DRL_PATH}/environment/cartpole/include)

set(CARTPOLE_NAME "cartpole")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CARTPOLE_NAME "${CARTPOLE_NAME}-d")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(CARTPOLE_NAME "${CARTPOLE_NAME}-rd")
endif()

find_library( CARTPOLE_LIBRARY
  NAMES ${CARTPOLE_NAME}
  PATHS
  "${ROOT_DRL_PATH}/environment/cartpole/lib"
)

find_package(ODEEnv)
set(CARTPOLE_INCLUDE_DIRS ${CARTPOLE_INCLUDE_DIRS} ${ODE_ENV_INCLUDE_DIRS})
set(CARTPOLE_LIBRARY ${CARTPOLE_LIBRARY} ${ODE_ENV_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AdvancedAcrobot DEFAULT_MSG CARTPOLE_LIBRARY CARTPOLE_INCLUDE_DIRS)
mark_as_advanced(CARTPOLE_INCLUDE_DIRS CARTPOLE_LIBRARY )
