find_package(Boost 1.54 COMPONENTS serialization filesystem system program_options thread REQUIRED)
find_package(Protobuf REQUIRED)
find_package(GLOG)
find_package(CUDA)

if(${Boost_MAJOR_VERSION} VERSION_EQUAL 1 AND ${Boost_MINOR_VERSION} VERSION_LESS 60)
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_SCOPED_ENUMS")
endif()

if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
  if(EXISTS "/opt/cuda")
          set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda")
  else()
          set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-7.0/")
  endif()
  message("CUDA_TOOLKIT_ROOT_DIR not found or specified, set to ${CUDA_TOOLKIT_ROOT_DIR}")
endif()

find_package(Caffe QUIET)
if(NOT DEFINED Caffe_LIBRARIES)
	if(NOT EXISTS ${ROOT_DRL_PATH}/common/build/${build_dir_type}/caffe-install/)
	       configure_file(${ROOT_DRL_PATH}/common/cmake/Caffe.in caffe-download/CMakeLists.txt)
	       execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} .
	         RESULT_VARIABLE result
	         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/caffe-download )
	
	       if(result)
	         message(FATAL_ERROR "CMake step for caffe failed: ${result}")
	       endif()
	       execute_process(COMMAND ${CMAKE_COMMAND} --build .
	         RESULT_VARIABLE result
	         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/caffe-download )
	       if(result)
	         message(FATAL_ERROR "Build step for caffe failed: ${result}")
	       endif()
	       #remove wrong config files for debug building
	   	file(REMOVE ${CMAKE_BINARY_DIR}/caffe-build/CaffeTargets.cmake)
	    	file(REMOVE ${CMAKE_BINARY_DIR}/caffe-build/CaffeConfig.cmake)
	    	file(REMOVE ${CMAKE_BINARY_DIR}/caffe-build/cmake/CaffeConfig.cmake)

	endif()
	include(${ROOT_DRL_PATH}/common/build/${build_dir_type}/caffe-install/share/Caffe/CaffeConfig.cmake)
	#correct caffe include dir when not installed
	set_target_properties(caffe PROPERTIES 
	 	INTERFACE_INCLUDE_DIRECTORIES ${ROOT_DRL_PATH}/common/build/${build_dir_type}/caffe-install/include )
endif()

if(${Caffe_CPU_ONLY})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCAFFE_CPU_ONLY ")
endif()

