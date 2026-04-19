cmake_minimum_required(VERSION 3.15)
project(DistributedNerfMaster)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# tell CMake where to look for the header files (.h)
include_directories(include)

# find the lib C++ of ZeroMQ
find_package(cppzmq REQUIRED)

# create the executable
add_executable(master_node src/master_node.cpp)

# link the executable with the lib ZeroMQ
target_link_libraries(master_node cppzmq flatbuffers)
