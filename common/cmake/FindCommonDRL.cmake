include(${ROOT_DRL_PATH}/common/cmake/Callable.cmake)
include(${ROOT_DRL_PATH}/common/cmake/BoostConfig.cmake)
include(${ROOT_DRL_PATH}/common/cmake/CaffeConfig.cmake)

set(COMMON_DRL_INCLUDE_DIRS ${ROOT_DRL_PATH}/common/include)

set(COMMON_DRL_NAME "common-drl")
rename_buildtype(COMMON_DRL_NAME)

find_library( COMMON_DRL_LIBRARY
  NAMES ${COMMON_DRL_NAME}
  PATHS
  "${ROOT_DRL_PATH}/common/lib"
  "${ROOT_DRL_PATH}/common/lib/Debug"
  "${ROOT_DRL_PATH}/common/lib/Release"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CommonDRL DEFAULT_MSG COMMON_DRL_LIBRARY COMMON_DRL_INCLUDE_DIRS)
mark_as_advanced(COMMON_DRL_INCLUDE_DIRS COMMON_DRL_LIBRARY )

set(COMMON_DRL_INCLUDE_DIRS 
  ${COMMON_DRL_INCLUDE_DIRS} ${TBB_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
set(COMMON_DRL_LIBRARY_LIGHT 
  ${COMMON_DRL_LIBRARY} ${Boost_LIBRARIES} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
set(COMMON_DRL_LIBRARY 
  ${COMMON_DRL_LIBRARY} caffe ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES})

