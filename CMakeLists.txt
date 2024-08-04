cmake_minimum_required(VERSION 3.10)

# Set the project name
project(llamascratch)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the include directory
include_directories(${CMAKE_SOURCE_DIR}/external/sentencepiece/src)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Collect all source files in the src directory except main.cpp
file(GLOB SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)
list(REMOVE_ITEM SOURCES ${CMAKE_SOURCE_DIR}/src/main.cpp)

# Add subdirectory for external dependencies
add_subdirectory(external/sentencepiece)

# Define the main executable
add_executable(llamascratch ${SOURCES} ${CMAKE_SOURCE_DIR}/src/main.cpp)
target_link_libraries(llamascratch sentencepiece)

# Define the test executable
add_executable(test_entry_point ${CMAKE_SOURCE_DIR}/tests/entry_point.cpp ${SOURCES})
target_link_libraries(test_entry_point sentencepiece)
