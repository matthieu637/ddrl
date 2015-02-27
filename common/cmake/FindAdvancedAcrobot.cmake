
set(ADVANCED_ACROBOT_INCLUDE_DIRS ../../environment/advanced-acrobot/include ../../environment/advanced-acrobot/extern/drawstuff/include)

set(ADVANCED_ACROBOT_NAME "advanced-acrobot")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(ADVANCED_ACROBOT_NAME "${ADVANCED_ACROBOT_NAME}-d")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(ADVANCED_ACROBOT_NAME "${ADVANCED_ACROBOT_NAME}-rd")
endif()

find_library( ADVANCED_ACROBOT_LIBRARY
  NAMES ${ADVANCED_ACROBOT_NAME}
  PATHS
  "../../environment/advanced-acrobot/lib"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AdvancedAcrobot DEFAULT_MSG ADVANCED_ACROBOT_LIBRARY ADVANCED_ACROBOT_INCLUDE_DIRS)
mark_as_advanced(ADVANCED_ACROBOT_INCLUDE_DIRS ADVANCED_ACROBOT_LIBRARY )
