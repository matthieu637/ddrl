
set(COMMON_DRL_INCLUDE_DIRS ../../common/include)

set(COMMON_DRL_NAME "common-drl")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(COMMON_DRL_NAME "${COMMON_DRL_NAME}-d")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(COMMON_DRL_NAME "${COMMON_DRL_NAME}-rd")
endif()

find_library( COMMON_DRL_LIBRARY
  NAMES ${COMMON_DRL_NAME}
  PATHS
  "../../common/lib"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CommonDRL DEFAULT_MSG COMMON_DRL_LIBRARY COMMON_DRL_INCLUDE_DIRS)
mark_as_advanced(COMMON_DRL_INCLUDE_DIRS COMMON_DRL_LIBRARY )