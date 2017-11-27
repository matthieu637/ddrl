# find OPT Optimizer
FIND_PATH(FREEIMAGE_INCLUDE_DIRS FreeImage.h
  /usr/include
  /usr/local/include
)

FIND_LIBRARY(FREEIMAGE_LIBRARY
  NAMES freeimage 
  PATHS
  /usr/lib
  /usr/lib64
  /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FreeImage DEFAULT_MSG FREEIMAGE_LIBRARY FREEIMAGE_INCLUDE_DIRS)
mark_as_advanced(FREEIMAGE_LIBRARY FREEIMAGE_INCLUDE_DIRS)

