
set(HALF_CHEETAH_INCLUDE_DIRS ${ROOT_DRL_PATH}/environment/half_cheetah/include)

set(HALF_CHEETAH_NAME "half_cheetah")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(HALF_CHEETAH_NAME "${HALF_CHEETAH_NAME}-d")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(HALF_CHEETAH_NAME "${HALF_CHEETAH_NAME}-rd")
endif()

find_library( HALF_CHEETAH_LIBRARY
  NAMES ${HALF_CHEETAH_NAME}
  PATHS
  "${ROOT_DRL_PATH}/environment/half_cheetah/lib"
  "${ROOT_DRL_PATH}/environment/half_cheetah/lib/Release"
  "${ROOT_DRL_PATH}/environment/half_cheetah/lib/Debug"
)

find_package(ODEEnv)
set(HALF_CHEETAH_INCLUDE_DIRS ${HALF_CHEETAH_INCLUDE_DIRS} ${ODE_ENV_INCLUDE_DIRS})
set(HALF_CHEETAH_LIBRARY ${HALF_CHEETAH_LIBRARY} ${ODE_ENV_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AdvancedAcrobot DEFAULT_MSG HALF_CHEETAH_LIBRARY HALF_CHEETAH_INCLUDE_DIRS)
mark_as_advanced(HALF_CHEETAH_INCLUDE_DIRS HALF_CHEETAH_LIBRARY )
