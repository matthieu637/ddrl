# find OPT Optimizer
FIND_PATH(OPTPP_INCLUDE_DIRS opt++/Opt.h
  /usr/include
  /usr/local/include
)

FIND_LIBRARY(OPTPP_LIBRARY
  NAMES opt opt++
  PATHS
  /usr/lib
  /usr/lib64
  /usr/local/lib
)

FIND_LIBRARY(NEWMAT_LIBRARY
  NAMES newmat
  PATHS
  /usr/lib
  /usr/lib64
  /usr/local/lib
)

FIND_LIBRARY(BLAS_LIBRARY
  NAMES blas
  PATHS
  /usr/lib
  /usr/lib64
  /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OPTPP DEFAULT_MSG OPTPP_LIBRARIES OPTPP_INCLUDE_DIRS)
mark_as_advanced(OPTPP_INCLUDE_DIRS OPTPP_LIBRARIES )

if(${OPTPP_LIBRARY_FOUND})
	set(OPTPP_LIBRARIES ${OPTPP_LIBRARY} ${NEWMAT_LIBRARY} ${BLAS_LIBRARY})
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_NAMESPACES -DHAVE_STD -DWANT_MATH")

	message(STATUS "OPTPP_INCLUDE_DIRS: ${OPTPP_INCLUDE_DIRS}")
	message(STATUS "OPTPP_LIBRARIES: ${OPTPP_LIBRARIES}")
else()
	message("OPTPP not found (optional)")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_OPTPP")
endif()

