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

set(CAFFE_ALL_INCLUDE 
  ${Caffe_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${GLOG_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
set(CAFFE_ALL_LIBRARIES 
  ${Caffe_LIBRARIES} ${CUDA_LIBRARIES} ${PROTOBUF_LIBRARIES} ${GLOG_LIBRARIES} ${Boost_LIBRARIES})
