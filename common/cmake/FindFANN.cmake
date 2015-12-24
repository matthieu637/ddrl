#  FANN_FOUND - system has fann
#  FANN_INCLUDE_DIRS - the fann include directory
#  FANN_LIBRARIES - Link these to use fann
#  FANN_DEFINITIONS - Compiler switches required for using fann
#

if(FANN_LIBRARIES AND FANN_INCLUDE_DIRS)
  set(FANN_FOUND TRUE)
else()
  find_path(FANN_INCLUDE_DIR
    NAMES
      fann.h
    PATHS
      ${FANN_DIR}/include
      /usr/include
      /usr/include/fann
      /usr/local/include/fann
      /usr/local/include
      /opt/local/include
      /sw/include
      ~/local_build/include/fann/
  )

  set( _libraries fann doublefann fixedfann floatfann )

  foreach( _lib ${_libraries} )
    string( TOUPPER ${_lib} _name )

    find_library(${_name}_LIBRARY
      NAMES
        ${_lib}
      PATHS
        ${FANN_DIR}/lib
        /usr/lib
        /usr/local/lib
        /opt/local/lib
        /sw/lib
	~/local_build/lib/
      )

  endforeach()


  set(FANN_INCLUDE_DIRS
    ${FANN_INCLUDE_DIR}
  )

  set(FANN_LIBRARIES
    ${FANN_LIBRARIES}
    ${FANN_LIBRARY}
    ${DOUBLEFANN_LIBRARY}
    ${FIXEDFANN_LIBRARY}
    ${FLOATFANN_LIBRARY}
  )

  if( UNIX )
    set( FANN_LIBRARIES ${FANN_LIBRARIES} m )
  endif()

  if(FANN_INCLUDE_DIRS AND FANN_LIBRARIES)
     set(FANN_FOUND TRUE)
  endif()

  if(FANN_FOUND)
    if(NOT FANN_FIND_QUIETLY)
      message(STATUS "Found FANN:")
      message(STATUS "FANN_INCLUDE_DIRS: ${FANN_INCLUDE_DIRS}")
      message(STATUS "FANN_LIBRARIES: ${FANN_LIBRARIES}")
    endif()
  else()
    if(FANN_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find FANN")
    endif()
  endif()

  mark_as_advanced(FANN_INCLUDE_DIRS FANN_LIBRARIES)
endif()
