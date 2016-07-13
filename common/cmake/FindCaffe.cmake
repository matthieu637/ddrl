#  CAFFE_FOUND - system has caffe
#  CAFFE_INCLUDE_DIRS - the caffe include directory
#  CAFFE_LIBRARY - Link these to use caffe
#  CAFFE_DEFINITIONS - Compiler switches required for using caffe
#

if(CAFFE_LIBRARY AND CAFFE_INCLUDE_DIRS)
  set(CAFFE_FOUND TRUE)
else()
  find_path(CAFFE_INCLUDE_DIR
    NAMES
      caffe.hpp
    PATHS
      ${CAFFE_DIR}/include
      /usr/include
      /usr/include/caffe
      /usr/local/include
      /usr/local/include/caffe
      /opt/local/include
      /sw/include
  )

  find_library(CAFFE_LIBRARY NAMES caffe PATHS
        ${CAFFE_DIR}/lib
        /usr/lib
        /usr/local/lib
        /opt/local/lib
        /sw/lib
      )


  set(CAFFE_INCLUDE_DIRS
    ${CAFFE_INCLUDE_DIR}
  )

  if( UNIX )
    set( CAFFE_LIBRARIES ${CAFFE_LIBRARIES} m )
  endif()

  if(CAFFE_INCLUDE_DIRS AND CAFFE_LIBRARIES)
     set(CAFFE_FOUND TRUE)
  endif()

  if(CAFFE_FOUND)
    if(NOT CAFFE_FIND_QUIETLY)
      message(STATUS "Found CAFFE:")
      message(STATUS "CAFFE_INCLUDE_DIRS: ${CAFFE_INCLUDE_DIRS}")
      message(STATUS "CAFFE_LIBRARIES: ${CAFFE_LIBRARY}")
    endif()
  else()
    if(CAFFE_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find CAFFE")
    endif()
  endif()

  mark_as_advanced(CAFFE_INCLUDE_DIRS CAFFE_LIBRARY)
endif()
