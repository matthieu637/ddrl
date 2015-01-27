
set(COMMON_DRL_INCLUDE_DIRS ../../common/include)

find_library( COMMON_DRL_LIBRARY
  NAMES common-drl
  PATHS
  "../../common/lib"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CommonDRL DEFAULT_MSG COMMON_DRL_LIBRARY COMMON_DRL_INCLUDE_DIRS)
mark_as_advanced(COMMON_DRL_INCLUDE_DIRS COMMON_DRL_LIBRARY )