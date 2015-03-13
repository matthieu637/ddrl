
#-------------------
# Rename a string depending on the build type
#-------------------
function(rename_buildtype NAME_TO_CHANGE)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(${NAME_TO_CHANGE} "${${NAME_TO_CHANGE}}-d" PARENT_SCOPE)
  elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(${NAME_TO_CHANGE} "${${NAME_TO_CHANGE}}-rd" PARENT_SCOPE)
  endif()
endfunction()

#-------------------
# Unit Test macro
#-------------------
function(enable_utest)
  enable_testing()

  file ( GLOB all_test_sources src/test/*Utest.cpp )
  add_executable(unit-test ${all_test_sources} ${all_sources})
  target_link_libraries(unit-test ${GTEST_BOTH_LIBRARIES} ${ARGN})
  add_test(AllTests unit-test)
endfunction()