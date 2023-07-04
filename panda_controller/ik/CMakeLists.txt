cmake_minimum_required(VERSION 3.12)
project(NJSDF_FUN)

set(CMAKE_CXX_STANDARD 11)

# Find Python and Eigen
# find_package(Python 3.7 REQUIRED COMPONENTS Development)
# Set the paths to the Python header files and libraries
set(PYTHON_INCLUDE_DIRS "/home/yoonjunheon/.local/share/ov/pkg/isaac_sim-2022.2.1/kit/python/include/python3.7m")
set(PYTHON_LIBRARIES "/home/yoonjunheon/.local/share/ov/pkg/isaac_sim-2022.2.1/kit/python/lib/libpython3.so")
set(PYTHON_LIB_DIR "/home/yoonjunheon/.local/share/ov/pkg/isaac_sim-2022.2.1/kit/python/lib/python3.7/site-packages/NJSDF")
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})

# Add your C++ source files here
set(SOURCES NJSDF.cpp)

# Create shared library in output directory
add_library(NJSDF_FUN MODULE ${SOURCES})
set_target_properties(NJSDF_FUN PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYTHON_LIB_DIR})

# Link against Python library
target_link_libraries(NJSDF_FUN PRIVATE ${Python_LIBRARIES})
