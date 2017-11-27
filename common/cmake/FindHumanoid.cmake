set(HUMANOID_INCLUDE_DIRS ${ROOT_DRL_PATH}/environment/humanoid/include)

set(HUMANOID_NAME "humanoid")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(HUMANOID_NAME "${HUMANOID_NAME}-d")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(HUMANOID_NAME "${HUMANOID_NAME}-rd")
endif()

find_library( HUMANOID_LIBRARY
  NAMES ${HUMANOID_NAME}
  PATHS
  "${ROOT_DRL_PATH}/environment/humanoid/lib"
  "${ROOT_DRL_PATH}/environment/humanoid/lib/Debug"
  "${ROOT_DRL_PATH}/environment/humanoid/lib/Release"
)

find_package(ODEEnv)
set(HUMANOID_INCLUDE_DIRS ${HUMANOID_INCLUDE_DIRS} ${ODE_ENV_INCLUDE_DIRS})
set(HUMANOID_LIBRARY ${HUMANOID_LIBRARY} ${ODE_ENV_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AdvancedAcrobot DEFAULT_MSG HUMANOID_LIBRARY HUMANOID_INCLUDE_DIRS)
mark_as_advanced(HUMANOID_INCLUDE_DIRS HUMANOID_LIBRARY )
