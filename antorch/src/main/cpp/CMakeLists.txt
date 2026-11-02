cmake_minimum_required(VERSION 3.18.1)
project(antorch_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(antorch_native SHARED anttorch_native.cpp kernels.cpp)

find_library(log-lib log)
target_link_libraries(antorch_native ${log-lib})
target_compile_options(antorch_native PRIVATE -O3 -Wall -Wextra)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_compile_options(antorch_native PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(antorch_native ${OpenMP_CXX_LIBRARIES})
endif()
