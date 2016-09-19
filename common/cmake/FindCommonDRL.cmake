include(${ROOT_DRL_PATH}/common/cmake/Callable.cmake)
#Boost library
find_package(Boost COMPONENTS serialization filesystem system program_options thread REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Caffe REQUIRED)
find_package(GLOG)
find_package(CUDA)

if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
  if(EXISTS "/opt/cuda")
          set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda")
  else()
          set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-7.0/")
  endif()
  message("CUDA_TOOLKIT_ROOT_DIR not found or specified, set to ${CUDA_TOOLKIT_ROOT_DIR}")
endif()

if(${Caffe_CPU_ONLY})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCAFFE_CPU_ONLY ")
endif()

set(COMMON_DRL_INCLUDE_DIRS ${ROOT_DRL_PATH}/common/include)

set(COMMON_DRL_NAME "common-drl")
rename_buildtype(COMMON_DRL_NAME)

find_library( COMMON_DRL_LIBRARY
  NAMES ${COMMON_DRL_NAME}
  PATHS
  "${ROOT_DRL_PATH}/common/lib"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CommonDRL DEFAULT_MSG COMMON_DRL_LIBRARY COMMON_DRL_INCLUDE_DIRS)
mark_as_advanced(COMMON_DRL_INCLUDE_DIRS COMMON_DRL_LIBRARY )

set(COMMON_DRL_INCLUDE_DIRS 
  ${COMMON_DRL_INCLUDE_DIRS} ${Caffe_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${GLOG_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
set(COMMON_DRL_LIBRARY 
  ${COMMON_DRL_LIBRARY} ${CUDA_LIBRARIES} ${PROTOBUF_LIBRARIES} ${GLOG_LIBRARIES} ${Caffe_LIBRARIES} ${Boost_LIBRARIES})
