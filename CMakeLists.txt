cmake_minimum_required(VERSION 3.10)

# Set the project name
project(llamascratch)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the include directory
include_directories(${CMAKE_SOURCE_DIR}/external/sentencepiece/src)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Collect all source files in the src directory
file(GLOB SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_subdirectory(external/sentencepiece)
# Define the executable
add_executable(llamascratch ${SOURCES})

target_link_libraries(llamascratch sentencepiece)
