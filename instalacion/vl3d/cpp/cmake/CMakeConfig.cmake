# ---   CMAKE CONFIG   --- #
# ------------------------ #

# RELEASE as default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

# C++ 17 as the standard
set(CMAKE_CXX_STANDARD 17)

# Common compilation flags
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
if(WIN32 OR MSVC)  # Windows flags
    set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Ob2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "")
    set(CMAKE_CXX_FLAGS "-D_OS_WINDOWS_ /DWIN32_LEAN_AND_MEAN /EHsc")
else()  # Linux flags
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -Wall")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -DNDEBUG -Wall -g")
    set(CMAKE_CXX_FLAGS "-pthread -Wno-deprecated")
endif()

# Enable ASan (Address Sanitizer), if requested
if(${ASAN})
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
    message("Enabled Address Sanitizer (ASan)")
else()
    message("Disabled Address Sanitizer (ASan)")
endif()

# Add debug build definition
if("${BUILD_TYPE}" STREQUAL "DEBUG")
    add_definitions(-DDEBUG_BUILD)
endif()

# Initial messages
if(WIN32 OR MSVC)
    message("CMake compiling for WINDOWS")
else()
    message("CMake compiling for LINUX")
endif()
message("CMake compiling from : '${CMAKE_CURRENT_SOURCE_DIR}'")
message("CMAKE BUILD TYPE : '${CMAKE_BUILD_TYPE}'")
message("BUILD TYPE : '${BUILD_TYPE}'")
message("--------------------\n")
