cmake_minimum_required(VERSION 3.10)

# Set the project name
project(llamascratch LANGUAGES CXX CUDA)

# Find CUDA package
find_package(CUDA REQUIRED)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the include directories
include_directories(${CMAKE_SOURCE_DIR}/external/sentencepiece/src)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})


# Collect all C++ source files in the src directory except main.cpp
file(GLOB CPP_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)
list(REMOVE_ITEM CPP_SOURCES ${CMAKE_SOURCE_DIR}/src/main.cpp)

# Specify the CUDA source file explicitly
set(CUDA_SOURCES ${CMAKE_SOURCE_DIR}/src/kernels.cu)

# Add subdirectory for external dependencies
add_subdirectory(external/sentencepiece)

# Define the main executable
add_executable(llamascratch ${CPP_SOURCES} ${CUDA_SOURCES} ${CMAKE_SOURCE_DIR}/src/main.cpp)
target_link_libraries(llamascratch sentencepiece ${CUDA_cudart_LIBRARY})

# Set the --expt-relaxed-constexpr flag globally for all CUDA files
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")


# Define the test executable
add_executable(test_entry_point ${CMAKE_SOURCE_DIR}/tests/entry_point.cpp ${CPP_SOURCES} ${CUDA_SOURCES})
target_link_libraries(test_entry_point sentencepiece ${CUDA_cudart_LIBRARY})

# Set CUDA properties for all targets
set_target_properties(llamascratch PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75")

set_target_properties(test_entry_point PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75")
