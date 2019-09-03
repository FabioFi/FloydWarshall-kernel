set(PROJECT "FloydWarshall")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.5)
FIND_PACKAGE(CUDA REQUIRED)

message("\nTest on cmake v" ${CMAKE_VERSION} "  System: " ${CMAKE_SYSTEM} "\n")

# ------------------------------------------------------------------------------
# CHECK C++ 14


include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
if (COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    add_compile_options("-std=c++14")
else()
    message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER} has no C++14 support.")
endif()

# ------------------------------------------------------------------------------

if(CUDA_VERSION_MAJOR LESS 8)
	message(FATAL_ERROR "\nRequired Cuda Toolkit >= v8.0")
endif()
# ------------------------------------------------------------------------------

add_compile_options("-O3")

include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)


list(APPEND CUDA_NVCC_FLAGS "-arch=sm_52")
list(APPEND CUDA_NVCC_FLAGS "-w")

cuda_add_executable(floydwarshall ${PROJECT_SOURCE_DIR}/kernel.cu ${SRCS})
