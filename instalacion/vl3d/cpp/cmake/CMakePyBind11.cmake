# ---   CMAKE PYBIND11   --- #
# -------------------------- #

# Add PyBind11
find_package(Python3 COMPONENTS Development NumPy)
set(Pybind11_Python3_INCLUDE_DIRS ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
message("PyBind11_Python3_INCLUDE_DIRS: ${Pybind11_Python3_INCLUDE_DIRS}")
# Try to find PyBind11
find_package(pybind11 QUIET)

# If not found, add it as a subdirectory
if(NOT pybind11_FOUND)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/pybind11)
endif()
