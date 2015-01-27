# - Try to find drawstuff (ODE internal library)
# Once done this will define
#  DRAWSTUFF_INCLUDE_DIRS
#  DRAWSTUFF_LIBRARY

set(DRAWSTUFF_INCLUDE_DIRS /usr/local/include/)

#Trouver lib
find_library( DRAWSTUFF_LIBRARY 	  NAMES drawstuff
				  PATHS 
					"/usr/lib"
					"/usr/local/lib/"
				)


include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Drawstuff  DEFAULT_MSG
                                  DRAWSTUFF_LIBRARY DRAWSTUFF_INCLUDE_DIRS)

mark_as_advanced(DRAWSTUFF_INCLUDE_DIRS DRAWSTUFF_LIBRARY )

if(DRAWSTUFF_INCLUDE_DIRS AND DRAWSTUFF_LIBRARY)
	set(DRAWSTUFF_FOUND TRUE)
else(DRAWSTUFF_INCLUDE_DIRS AND DRAWSTUFF_LIBRARY)
	set(DRAWSTUFF_FOUND FALSE)
endif(DRAWSTUFF_INCLUDE_DIRS AND DRAWSTUFF_LIBRARY)

