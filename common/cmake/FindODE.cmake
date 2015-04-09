# find ODE (Open Dynamics Engine) includes and library
#
# ODE_INCLUDE_DIR - where the directory containing the ODE headers can be
# found
# ODE_LIBRARY - full path to the ODE library
# ODE_CFLAGS - additional compiler flags for ODE
# ODE_FOUND - TRUE if ODE was found


# To build ODE in realease mode :
# makepkg PKGBUILD


IF (NOT ODE_FOUND)
  IF (ODE_EXTRA_CFLAGS)
    SET(ODE_CFLAGS ${ODE_EXTRA_CFLAGS} CACHE STRING "Additional ODE flags")
    MESSAGE(STATUS "Found additional flags for ODE: ${ODE_CFLAGS}")
  ELSE (ODE_EXTRA_CFLAGS)
    SET(ODE_CFLAGS CACHE STRING "Additional ODE flags")
  ENDIF (ODE_EXTRA_CFLAGS)

  #Try to get header from Release version
  IF(CMAKE_BUILD_TYPE STREQUAL "Release")
    FIND_PATH(ODE_INCLUDE_DIR ode/ode.h /usr/local/include)
  ENDIF()
  
  IF(ODE_INCLUDE_DIR)
    MESSAGE(STATUS "ODE release include capted.")
  ELSE(ODE_INCLUDE_DIR)
    MESSAGE(STATUS "ODE release include not capted.")
    FIND_PATH(ODE_INCLUDE_DIR ode/ode.h
      /usr/include
      /usr/local/include
      $ENV{OGRE_HOME}/include # OGRE SDK on WIN32
      $ENV{INCLUDE}
      C:/library/ode/include
      "C:/Program Files/ode/include"
      C:/ode/include
    )
  ENDIF(ODE_INCLUDE_DIR)

  IF(CMAKE_BUILD_TYPE STREQUAL "Release")
    FIND_LIBRARY(ODE_LIBRARY NAMES ode_r PATHS /usr/local/lib)
  ENDIF()
  
  IF(ODE_LIBRARY)
    MESSAGE(STATUS "ODE release library capted.")
  ELSE(ODE_LIBRARY)
    MESSAGE(STATUS "ODE release library not capted.")
    FIND_LIBRARY(ODE_LIBRARY
      NAMES ode ode_double ode_single
      PATHS
      /usr/lib
      /usr/lib64
      /usr/local/lib
      $ENV{OGRE_HOME}/lib # OGRE SDK on WIN32
      C:/library/ode/lib/
      "C:/Program Files/ode/lib/"
      C:/ode/lib/
  PATH_SUFFIXES
  releaselib
  ReleaseDoubleDLL ReleaseDoubleLib
  ReleaseSingleDLL ReleaseSingleLib
    )
  ENDIF(ODE_LIBRARY)
  

  IF(ODE_INCLUDE_DIR)
    MESSAGE(STATUS "Found ODE include dir: ${ODE_INCLUDE_DIR}")
  ELSE(ODE_INCLUDE_DIR)
    MESSAGE(STATUS "Could NOT find ODE headers.")
  ENDIF(ODE_INCLUDE_DIR)

  IF(ODE_LIBRARY)
    MESSAGE(STATUS "Found ODE library: ${ODE_LIBRARY}")
  ELSE(ODE_LIBRARY)
    MESSAGE(STATUS "Could NOT find ODE library.")
  ENDIF(ODE_LIBRARY)

  IF(ODE_INCLUDE_DIR AND ODE_LIBRARY)
     SET(ODE_FOUND TRUE CACHE STRING "Whether ODE was found or not")
     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DdDOUBLE")
   ELSE(ODE_INCLUDE_DIR AND ODE_LIBRARY)
     SET(ODE_FOUND FALSE)
     IF(ODE_FIND_REQUIRED)
       MESSAGE(FATAL_ERROR "Could not find ODE. Please install ODE (http://www.ode.org)")
     ENDIF(ODE_FIND_REQUIRED)
   ENDIF(ODE_INCLUDE_DIR AND ODE_LIBRARY)
ENDIF (NOT ODE_FOUND)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ODE DEFAULT_MSG ODE_LIBRARY ODE_INCLUDE_DIR)
mark_as_advanced(ODE_INCLUDE_DIR ODE_LIBRARY )
