include(${CMAKE_SOURCE_DIR}/../../common/cmake/Callable.cmake)
set(COMMON_DRL_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/../../common/include)

set(COMMON_DRL_NAME "common-drl")
rename_buildtype(COMMON_DRL_NAME)

find_library( COMMON_DRL_LIBRARY
  NAMES ${COMMON_DRL_NAME}
  PATHS
  "../../common/lib"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CommonDRL DEFAULT_MSG COMMON_DRL_LIBRARY COMMON_DRL_INCLUDE_DIRS)
mark_as_advanced(COMMON_DRL_INCLUDE_DIRS COMMON_DRL_LIBRARY )