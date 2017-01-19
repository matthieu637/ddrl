find_package(GTest)

if(NOT ${GTEST_FOUND})
        include(${ROOT_DRL_PATH}/common/cmake/Callable.cmake)
        if(NOT EXISTS ${ROOT_DRL_PATH}/common/build/${build_dir}/googletest-build/)
                # Download and unpack googletest at configure time
                configure_file(${CMAKE_SOURCE_DIR}/cmake/GTest.in googletest-download/CMakeLists.txt)
                execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                  RESULT_VARIABLE result
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
                if(result)
                  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
                endif()
                execute_process(COMMAND ${CMAKE_COMMAND} --build .
                  RESULT_VARIABLE result
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
                if(result)
                  message(FATAL_ERROR "Build step for googletest failed: ${result}")
                endif()
        endif()
        # Prevent overriding the parent project's compiler/linker
        # settings on Windows
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

        if (CMAKE_VERSION VERSION_LESS 2.8.11)
                include_directories("${CMAKE_BINARY_DIR}/googletest-src/googletest/include/")
        endif()

        set(GTEST_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/googletest-src/googletest/include/")
        find_library(GTEST_MAIN_LIBRARY NAMES gtest_main 
                PATHS 
                        ${CMAKE_BINARY_DIR}/googletest-build/googlemock/gtest/ 
                        ${ROOT_DRL_PATH}/common/build/${build_dir}/googletest-build/googlemock/gtest/
                )
        find_library(GTEST_LIBRARY NAMES gtest
                PATHS 
                        ${CMAKE_BINARY_DIR}/googletest-build/googlemock/gtest/
                        ${ROOT_DRL_PATH}/common/build/${build_dir}/googletest-build/googlemock/gtest/
                )
        set(GTEST_BOTH_LIBRARIES ${GTEST_MAIN_LIBRARY} ${GTEST_LIBRARY})

        include(FindPackageHandleStandardArgs)
        find_package_handle_standard_args(GTest DEFAULT_MSG GTEST_BOTH_LIBRARIES GTEST_INCLUDE_DIRS)
        mark_as_advanced(GTEST_INCLUDE_DIRS GTEST_BOTH_LIBRARIES)

endif()

