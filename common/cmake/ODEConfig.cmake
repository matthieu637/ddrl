find_package(ODE QUIET)

if( NOT ODE_FOUND )
	if(NOT EXISTS ${ROOT_DRL_PATH}/environment/ode-env/build/${build_dir_type}/ode-src/ode/src/.libs/)
	        configure_file(${ROOT_DRL_PATH}/common/cmake/ODE.in ode-download/CMakeLists.txt)
	        execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
	          RESULT_VARIABLE result
	          WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/ode-download )
	
	        if(result)
	          message(FATAL_ERROR "CMake step for ode failed: ${result}")
	        endif()
	        execute_process(COMMAND ${CMAKE_COMMAND} --build .
	          RESULT_VARIABLE result
	          WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/ode-download )
	        if(result)
	          message(FATAL_ERROR "Build step for ode failed: ${result}")
	        endif()
	endif()

        set(ODE_INCLUDE_DIR "${ROOT_DRL_PATH}/environment/ode-env/build/${build_dir_type}/ode-src/include/")
        set(ODE_INCLUDE_DIR "${CMAKE_BINARY_DIR}/ode-src/include/")
        find_library(ODE_LIBRARY NAMES ode PATHS 
		${CMAKE_BINARY_DIR}/ode-src/ode/src/.libs/
		${ROOT_DRL_PATH}/environment/ode-env/build/${build_dir_type}/ode-src/ode/src/.libs/
	)

        include(FindPackageHandleStandardArgs)
        find_package_handle_standard_args(ODE DEFAULT_MSG ODE_LIBRARY ODE_INCLUDE_DIR)
        mark_as_advanced(ODE_LIBRARY ODE_INCLUDE_DIR)
endif()
