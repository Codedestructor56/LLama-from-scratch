cmake_minimum_required(VERSION 3.10)

# Set the project name
project(llamascratch LANGUAGES CXX CUDA)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the include directory
include_directories(${CMAKE_SOURCE_DIR}/external/sentencepiece/src)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Collect all source files in the src directory except main.cpp
file(GLOB CPP_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB CUDA_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cu)
list(REMOVE_ITEM CPP_SOURCES ${CMAKE_SOURCE_DIR}/src/main.cpp)

# Add subdirectory for external dependencies
add_subdirectory(external/sentencepiece)

# Define the main executable
add_executable(llamascratch ${CPP_SOURCES} ${CUDA_SOURCES} ${CMAKE_SOURCE_DIR}/src/main.cpp)
target_link_libraries(llamascratch sentencepiece)

# Define the test executable
add_executable(test_entry_point ${CMAKE_SOURCE_DIR}/tests/entry_point.cpp ${CPP_SOURCES} ${CUDA_SOURCES})
target_link_libraries(test_entry_point sentencepiece)

# Set CUDA properties
set_target_properties(llamascratch PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(test_entry_point PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
